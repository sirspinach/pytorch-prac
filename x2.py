import torch
from torch import nn
from collections import OrderedDict


K = 6
ff = nn.Sequential(nn.Linear(1, K), nn.LeakyReLU(K), nn.Linear(K, K),
        nn.LeakyReLU(K), nn.Linear(K, 1))

def init_weights(m):
    return
    if type(m) is nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, 0.01)
ff.apply(init_weights)

N = 2000
X = torch.randn(N, 1)
Y = X**2

optimizer = torch.optim.SGD(ff.parameters(), 0.003, momentum=0.3)
loss_fn = nn.MSELoss()
for i in range(5000):
    optimizer.zero_grad()
    loss = loss_fn(ff.forward(X), Y)
    loss.backward()
    optimizer.step()
    print("MSE Error: {:.4f}".format(loss))

# Now plot our predictions
@torch.no_grad()
def visualize():
    import numpy as np
    from matplotlib import pyplot as plt
    X_test = torch.linspace(-5, 5).unsqueeze(1)
    with torch.no_grad():
        Y_hat = ff.forward(X_test).detach()
    plt.scatter(X_test.numpy(), Y_hat.numpy(), s=4, label="predicted")
    plt.scatter(X_test.numpy(), X_test.numpy()**2, s=4, label="actual")
    plt.legend()
    plt.show()
visualize()
