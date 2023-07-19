import torch
import torch.nn as nn
import pennylane as qml
from math import pi

n_qubits = 4
n_layers = 1
n_class = 2
n_features = 1024
image_x_y_dim = 32
kernel_size = n_qubits
stride = 2

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(pi * inputs[i], i)
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CNOT(wires=[(i + 1) % n_qubits, i])
            qml.RZ(weights[l, i + (n_qubits - 1)], wires=i)
            qml.CNOT(wires=[(i + 1) % n_qubits, i])
        for i in range(n_qubits):
            qml.RX(weights[l, n_qubits - 1 + i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


class Quanv2d(nn.Module):
    def __init__(self, kernel_size=None, stride=None):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers, 2 * n_qubits)}
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method='best')
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        assert len(X.shape) == 4
        XL = []
        for i in range(0, X.shape[2] - 2, stride):
            for j in range(0, X.shape[3] - 2, stride):
                XL.append(self.ql1(torch.flatten(X[:, :, i:i + kernel_size, j:j + kernel_size], start_dim=1)))
        X = torch.cat(XL, dim=1)
        return X


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.ql1 = Quanv2d(kernel_size=kernel_size, stride=stride)

        self.fc1 = nn.Linear(900, 450)
        self.fc2 = nn.Linear(450, 64)
        self.fc3 = nn.Linear(64, 49)
        self.fc4 = nn.Linear(49, 2)

    def forward(self, X):
        bs = X.shape[0]
        X = X.view(bs, 1, image_x_y_dim, image_x_y_dim)
        X = self.ql1(X)

        X = self.fc1(X)
        # X = self.lr1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        return X
