import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class BNNRegress(nn.Module):
    def __init__(self, precision=2):
        """
        Precision is the precision of Gaussian noise applied to the output.
        """
        super().__init__()
        self.precision = precision
        # I might actually backwards() through precisions, but it doesn't llok
        # like it right now. If that is true and I am using fixed precision
        # then I don't want to differentiate precision.
        # TODO: check to see if this kills gradients.
        # precision.detach_()  # Don't want to calculate derivative for this.

    def parameter_triples(self):
        """
        Return a sequence of (layer_input, layer_output, layer_weight) tuples,
        for use in optimizing a Bayesian posterior over weights.
        """
        raise NotImplementedError()

    def forward(self, X):
        raise NotImplementedError()

    def forward_ll(self, X, Y):
        raise NotImplementedError()
        mu = self.forward(X)
        return -0.5 * precision * sum(dyad(Y - mu, Y - mu))

    def parameters(self):
        # TODO: Hmmmm... check that super().parameters() automatically returns
        #   the Right Things?
        return super().parameters()
        raise NotImplementedError()


# meta TODO:
# check! 1. Run regular gradient descent through BasicBNNRegress on x^2 dataset.
# 2. Run noisy Adam on BasicBNNRegress with x^2 dataset. Without crashing or
#   nan.
# 3. Run noisy KFAC on BasicBNNRegress with x^2 dataset. Without crashing or
#   nan.

class BasicBNNRegress(BNNRegress):
    """
    A basic neural network, with 2 fully connected layers and a 1 dimensional
    output. Implements parameter_triples() for use by NoisyKFAC.
    """
    # TODO: Take a look at how sequential is implemented.
    def __init__(self, D, width, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.width = width

        self.W1 = nn.Linear(D, width)
        self.relu1 = nn.ReLU(width)
        self.W2 = nn.Linear(width, width)
        self.relu2 = nn.ReLU(width)
        self.W3 = nn.Linear(width, 1)

        # Holds activation inputs and outputs for each layer.
        # Used by noisy KFAC
        self.adjacent = {}

    def forward(self, X):
        # XXX: really ugly, try a helper fn. Perhaps this helper fn will
        #   have a option to decide whether or not to use adjacent[].
        a1 = X
        s2 = self.W1.forward(a1)
        self.adjacent[self.W1] = (a1, s2)

        a2 = self.relu1.forward(s2)
        s3 = self.W2.forward(a2)
        self.adjacent[self.W2] = (a2, s3)

        a3 = self.relu2.forward(s3)
        return self.W3.forward(a3)

# TODO: remove me
bnn = BasicBNNRegress(3, 3)


class NoisyAdam(Optimizer):
    # TODO: What hyperparameters do the original authors use on GitHub?
    def __init__(parameters, alpha=0.02, beta1=0.02, beta2=0.02, lamb=0.9,
            eta=0.1, damp_ext=0.01):
        super().__init__(parameters)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb = lamb
        self.eta = eta
        self.damp_ext = damp_ext
        self.damp_int = lamb/N/eta
        self.damp = self.damp_ext + self.damp_int
        self.f = torch.zeros()

    # Probably works out-of-the-box
    # def zero_grads(self):
    #     super().zero_grads()
    def step(self):
        w = self.sample(out=self.parameters)
        pass

    # I recall that in-place operations like randn_ are discouraged
    # in some scenarios involving autograd. Check out the official
    # implementation of GradientDescent or Adam does inplace operations.
    # TODO:
    def sample(self, out):
        raise NotImplementedError()
        w_new <- Normal(mu, self.lamb/N * diag((f + self.damp_int) ** -1))
        start = 0
        for w in out:
            # TODO: probably want to turn w into a view here.
            # Confirm via unit test with mu=arange and lamb=0 that w gets
            # modified as I expect.
            w.data = w_new[start:len(w)].reshape(w.data.shape)

class NoisyKFAC(Optimizer):
    def __init__(parameters, adj_dict, alpha, beta, lamb, eta, damp_ex, T_stats,
            T_inv, N):
        """
        N is the batch size.
        """
        super().__init__(parameters)
        # XXX: Might be too soon. We might want to wait to populate adj_dict
        #   until forward() call.
        for p in parameters:
            assert p in adj_dict
        self.adj_dict = adj_dict
        self.k = 0
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.eta = eta
        self.pi = 1  # Some arbitrary constant for kronkecer factorization
        self.damp_int = lamb / N / mu
        self.damp_ex = self.damp_Ex
        self.damp = self.damp_int + self.damp_ex
        self.N = N
        self.T_stats = T_stats
        self.T_inv = T_inv

    def step(self):
        raise NotImplementedError()
        if self.k % self.T_stats == 0:
            # TODO: Look at param dict instead
            for A, S, (w, a, s) in zip(A_list, S_list, param_triples):
                # TODO: What is beta_tilde?
                # TODO: How can I efficiently calculate a dyad?
                # TODO: How do I assign new data data to A? Maybe what I
                #   actually want to do is replace the entry in the list.
                #   I think it's better not to optimize memory first.
                A <- (1 - beta_tilde) * A + beta_tilde * dyad(a)
                S <- (1 - beta_tilde) * S + beta_tilde * dyad(s.grad)
        if self.k % self.T_inv == 0:
            for A, S in zip(A_list, S_list):
                # TODO: what is pi_l?
                # TODO: initialize S_inv_list
                factor = sqrt(self.lamb/self.N/self.eta)
                S_inv_list[l] <- self.lamb/self.N * inverse(
                        S + eye* 1/self.pi * factor)
                A_inv_list[l] <- inverse(A + self.pi * factor * eye)
        for l, (w, _, _) in enumerate(self.param_triples):
            V[l] <- w.grad - self.damp_int * w
        # TODO: Initialize mu list
        for l, mu in enumerate(self.mu_list):
            mu_list <- mu_list + self.alpha * A_inv_list[l] @ V[l] @ \
                    S_inv_list[l]

        # Now update weights, foo
        for l, w_new in enumerate(self.sample()):
            w_list[l] <- w_new

    def sample(self):
        """ Sample a sequence of weight vectors, to replace the weight vectors
        in param_triples"""
        raise NotImplementedError()
        for l in range(L):
            M = self.mu_list[l]
            S1 = self.lamb/self.N * self.A_inv_list[l]
            S2 = self.S_inv_list[l]
            yield mv_gauss.rvs(M, S1, S2)


    # TODO: If I pass parameters to super__init__(), then I don't think I need
    # to override zero_grads. Confirm via unit test.
    #def zero_grads(self):

# Later TODO:
# 1. Try gradient clipping.
