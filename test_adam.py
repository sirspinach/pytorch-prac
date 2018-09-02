import torch
from x2 import train_on_x2
from adam import BasicBNNRegress
def test_basic_bnn_regress_x2():
    bnn = BasicBNNRegress(D=1, width=3)
    train_on_x2(bnn, epochs=100, verbose=False)

    # zero = torch.zeros(1, 1)
    # print(bnn.forward(zero))
    # train_on_x2(bnn, epochs=2000, verbose=False)
    # print(bnn.forward(zero))
    # train_on_x2(bnn, epochs=2000, verbose=False)
    # print(bnn.forward(zero))

def test_noisy_adam():
    bnn = BasicBNNRegress(D=1, width=3)
    nadam = NoisyAdam(bnn.parameters(), )
    train_on_x2(bnn, optim=nadam, epochs=100, verbose=False)
