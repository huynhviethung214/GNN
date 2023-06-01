import numpy as np
import torch as t
from functorch import functionalize

from modules import forward, calculateGradients, sigmoid
from torch.autograd import Variable


batch = 64
size = [2, 2]
Nin = size[0] * size[1]
Nh = 1
Nout = 1

N = Nin + Nh + Nout

inputIdc = np.array([i for i in range(Nin)])
hiddenIdc = np.array([i + Nin for i in range(Nh)])
outputIdc = np.array([i + Nin + Nh for i in range(Nout)])

f = sigmoid

W = np.random.random((N, N))
np.fill_diagonal(W, 0)

B = np.random.random((N,))
P = np.random.random((N, 3))
R = np.random.random((N, 1))

I = np.zeros((N, 1))
O = np.zeros((N, 1))

u = np.ones(size).flatten()
u = u.reshape((-1, 1))

v = np.ones(size).flatten()
v = v.reshape((-1, 1))
v[0] = 0.

# I, O = forward(W, I, O, P, R, B,
#                inputIdc, hiddenIdc, outputIdc, u, f)
# print(O[outputIdc])

device = t.device('cuda')
u = Variable(t.tensor(u, dtype=t.float32, device=device))
v = Variable(t.tensor(v, dtype=t.float32, device=device), requires_grad=True)


# a = t.tensor([[[1, 1, 1],
#               [1, 1, 1],
#               [1, 1, 1]],
#               [[2, 2, 2],
#                [2, 2, 2],
#                [2, 2, 2]],
#               [[3, 3, 3],
#               [3, 3, 3],
#               [3, 3, 3]]])
# b = t.tensor([[4, 4, 4],
#               [4, 4, 4],
#               [4, 4, 4]])
# print(a * b)


def sigmoidGPU(x):
    return 1. / (1. + t.exp(-x))


def mag(u, v):
    return t.sqrt(t.sum((u - v) ** 2, dim=1)).unsqueeze(dim=1)


t.autograd.set_detect_anomaly(True)


class GeneralLayer(t.nn.Module):
    def __init__(self, W, B, P, R, N):
        super().__init__()
        self.O = t.zeros((N, 1), dtype=t.float32, device=device, requires_grad=True)
        self.W = t.tensor(W, dtype=t.float32, device=device, requires_grad=True)

        self.C = self.W.clone()
        self.C[t.where(self.C != 0.)] = 1.

        self.B = t.tensor(B, dtype=t.float32, device=device, requires_grad=True)
        self.P = t.tensor(P, dtype=t.float32, device=device, requires_grad=True)
        self.R = t.tensor(R, dtype=t.float32, device=device, requires_grad=True)

        self.padMat = t.zeros((self.O.size(0) - Nin, 1), device=device)
        self.vm = t.zeros(self.O.size(), requires_grad=True, dtype=t.float32, device=device)

    def _forward(self, x):
        x = t.vstack((x, self.padMat))
        self.O = self.O + x

        for hiddenIdx in hiddenIdc:
            self.vm = self.vm * 0.
            self.vm = self.vm + mag(self.P * self.C[:, hiddenIdx].unsqueeze(dim=1),
                                    self.P[hiddenIdx])
            self.vm[hiddenIdx] = self.vm[hiddenIdx] + 1.

            self.O[hiddenIdx] = self.O[hiddenIdx] \
                                + sigmoidGPU(t.sum((self.O * self.R
                                                    * self.W[:, hiddenIdx]
                                                    .unsqueeze(dim=1)) / self.vm)
                                             + self.B[hiddenIdx])

        for outputIdx in outputIdc:
            self.vm = self.vm * 0.
            self.vm = self.vm + mag(self.P * self.C[:, outputIdx].unsqueeze(dim=1),
                                    self.P[outputIdx])
            self.vm[outputIdx] = self.vm[outputIdx] + 1.

            self.O[outputIdx] = self.O[outputIdx] \
                                + sigmoidGPU(t.sum((self.O * self.R
                                                    * self.W[:, outputIdx]
                                                    .unsqueeze(dim=1)) / self.vm)
                                             + self.B[outputIdx])

        return self.O[outputIdc]

    def forward(self, x):
        ret = t.zeros((Nout,),
                      dtype=t.float32,
                      device=device,
                      requires_grad=True)
        ret = ret + functionalize(self._forward)(x)
        return ret


model = GeneralLayer(W, B, P, R, N).to(device)
p = model(u)
print(p)

loss = t.sum((v - p) ** 2) / 2
print(loss)
loss.backward()

print(model.W.grad)
print(model.B.grad)
print(model.P.grad)
print(model.R.grad)
