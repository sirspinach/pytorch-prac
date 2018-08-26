import torch, torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import numpy as np
import sklearn.preprocessing as pre
import pandas as pd

def data_from_xls(filename, test_proportion=.1):
    """

    Parse data from an xls file into scaled training and testing sets.
    Assumes that the final column is the output.

    Params:
    filename (string) -- The xls filename.
    test_proportion (float) -- A number between 0.0 and 1.0 indicating the
        proportion of data to reserve for testing.

    Returns:
    train_set (TensorDataset) -- The normalized training data, as
        (input, output) pairs.
    test_set (TensorDataset) -- The normalized testing data, as
        (input, output) pairs.
    scaler_x (sklearn.preprocessing.StandardScale) -- The scaling transform
        fitted to the training inputs. Can be used to unscale inputs.
    scaler_x (sklearn.preprocessing.StandardScale) -- The scaling transform
        fitted to the training outputs. Can be used to unscale outputs.
    """
    data = pd.ExcelFile("data/Concrete_Data.xls").parse().values.astype(
            np.float32)
    N = len(data)
    train, test = np.split(data, [int(-N * test_proportion)])

    train_X, train_Y = np.split(train, [-1], 1)
    test_X, test_Y = np.split(test, [-1], 1)

    scaler_X, scaler_Y = pre.StandardScaler(), pre.StandardScaler()
    train_X = torch.from_numpy(scaler_X.fit_transform(train_X))
    train_Y = torch.from_numpy(scaler_Y.fit_transform(train_Y))
    test_X = torch.from_numpy(scaler_X.transform(test_X))
    test_Y = torch.from_numpy(scaler_Y.transform(test_Y))

    return TensorDataset(train_X, train_Y), TensorDataset(test_X, test_Y), \
            scaler_X, scaler_Y

train_set, test_set, scaler_x, scaler_y = data_from_xls(
     "data/Concrete_Data.xls")

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True)

N = len(train_set)
D = len(train_set[0][0])
net = nn.Sequential(nn.Linear(D, D), nn.LeakyReLU(D), nn.Linear(D, D),
        nn.LeakyReLU(D), nn.Linear(D, 1))
loss_fn = nn.MSELoss()

def train(net, test_interval=25):
    optimizer = torch.optim.SGD(net.parameters(), 0.001, momentum=0.5)
    for epoch in range(1000):
        accum_loss = 0
        for X, Y in iter(train_loader):
            optimizer.zero_grad()
            loss = loss_fn(net.forward(X), Y)
            # print("MSE: {:.4f}".format(loss.item()))
            if np.isnan(loss.item()):
                raise Exception()
            loss.backward()
            accum_loss += loss.item()
            optimizer.step()
        print("epoch={} loss={:.3f}".format(epoch, accum_loss/N))
        if epoch % test_interval == test_interval - 1:
            test(net)

def test(net):
    accum_loss = 0
    with torch.no_grad():
        for X, Y in test_loader:
            Y_hat = net.forward(X)
            accum_loss += loss_fn(Y_hat, Y).item()
    print("test MSE: {}".format(accum_loss / len(test_set)))

train(net)
