from typing import Optional, Callable

import numpy as np
import torch
import torch as t

from functorch import functionalize
from numba import njit, prange


@njit(fastmath=True, parallel=True)
def removeRandomConnections(inputIdc, hiddenIdc, outputIdc,
                            inputPerNode, outputPerNode, C):
    for i in prange(inputIdc.shape[0]):
        inputIdx = inputIdc[i]
        nodeIdc = list(np.where(C[inputIdx, :] == 1)[0])

        if len(nodeIdc) > outputPerNode:
            for r in range(len(nodeIdc) - outputPerNode):
                randint = np.random.randint(0, len(nodeIdc))
                C[inputIdx, nodeIdc[randint]] = 0
                nodeIdc.pop(randint)

    for i in prange(hiddenIdc.shape[0]):
        hiddenIdx = hiddenIdc[i]
        nodeIdc = list(np.where(C[hiddenIdx, :] == 1)[0])

        if len(nodeIdc) > outputPerNode:
            for r in range(len(nodeIdc) - outputPerNode):
                randint = np.random.randint(0, len(nodeIdc))
                C[hiddenIdx, nodeIdc[randint]] = 0
                nodeIdc.pop(randint)

    for i in prange(hiddenIdc.shape[0]):
        hiddenIdx = hiddenIdc[i]
        nodeIdc = list(np.where(C[:, hiddenIdx] == 1)[0])

        if len(nodeIdc) > inputPerNode:
            for r in range(len(nodeIdc) - inputPerNode):
                randint = np.random.randint(0, len(nodeIdc))
                C[nodeIdc[randint], hiddenIdx] = 0
                nodeIdc.pop(randint)

    for i in prange(outputIdc.shape[0]):
        outputIdx = outputIdc[i]
        connections = list(np.where(C[:, outputIdx] == 1)[0])

        if len(connections) > inputPerNode:
            for r in range(len(connections) - inputPerNode):
                randint = np.random.randint(0, len(connections))
                C[connections[randint], outputIdx] = 0
                connections.pop(randint)

    return C


@njit()
def enableConnections(C, idc0, idc1):
    for idx0 in idc0:
        for idx1 in idc1:
            C[idx0, idx1] = 1.


@njit(fastmath=True, parallel=True)
def initialize(inputPerNode, outputPerNode, N,
               Nin, Nout,
               inputIdc, hiddenIdc, outputIdc):
    W = np.random.uniform(-np.sqrt(6 / (Nin + Nout)),
                          np.sqrt(6 / (Nin + Nout)), (N, N))  # Xavier's Weight Initialization

    # Initialize the connection matrix
    while True:
        C = np.ones((N, N))  # Connections Matrix

        # turn off connection(s) that is violating the constraint (see above)
        np.fill_diagonal(C, 0)

        for i in outputIdc:
            C[i, :] = 0

        for i in inputIdc:
            C[:, i] = 0

        # Fix two-way connections
        for i in range(N):
            for j in range(N):
                # if (i not in hiddenIdc or j not in hiddenIdc) and (i != j):
                if (i != j) and (C[i, j] == C[j, i] == 1):
                    rand = np.random.randint(0, 1)
                    if rand == 0:
                        C[i, j] = 0
                    else:
                        C[j, i] = 0

        C = removeRandomConnections(inputIdc, hiddenIdc, outputIdc,
                                    inputPerNode, outputPerNode, C)

        # If path exist then break the loop
        if pathExist(C, list(inputIdc)):
            return W * C, C


# Check if there is any path that lead from input node(s) to output node(s) using BFS
@njit()
def pathExist(C, inputIdc):
    queue = inputIdc.copy()

    while len(queue) > 0:
        e = queue[0]
        queue.pop(0)

        toIdc = np.where(C[e, :] == 1)[0]

        # toIdc represent the connections from i-th node to j-th node
        if toIdc.size > 0:
            for toIdx in toIdc:
                queue.append(toIdx)
        else:
            return True
    return False


# Activation Function
def sigmoid(x):
    return 1. / (1. + t.exp(-x))  # Sigmoid


# Loss Function
def g(ps, ts, Nout):
    return t.sum((ps - ts) ** 2, dim=1) / Nout


class GeneralNeuralNetwork(t.nn.Module):
    def __init__(self, width, height, depth,
                 hiddenZoneOffset, outputZoneOffset,
                 minBias, maxBias, maxRadius, minRadius,
                 maxInputPerNode, minInputPerNode,
                 maxOutputPerNode, minOutputPerNode,
                 batch, Nin, Nh, Nout, device):
        super().__init__()
        self.device = device

        self.f = sigmoid
        self.batch = batch

        self.width = width
        self.height = height
        self.depth = depth

        self.hiddenZoneOffset = hiddenZoneOffset
        self.outputZoneOffset = outputZoneOffset

        self.minBias = minBias
        self.maxBias = maxBias

        self.minRadius = minRadius
        self.maxRadius = maxRadius

        self.maxInputPerNode = maxInputPerNode
        self.minInputPerNode = minInputPerNode

        self.maxOutputPerNode = maxOutputPerNode
        self.minOutputPerNode = minOutputPerNode

        self.Nin = Nin
        self.Nh = Nh
        self.Nout = Nout
        self.N = Nin + Nh + Nout

        self.O = t.zeros((self.batch, self.N),
                         dtype=t.float32,
                         device=self.device)

        self.W = np.zeros((self.N, self.N),
                          dtype=np.float32)
        self.C = np.zeros_like(self.W)

        self.P = np.zeros((self.N, 3),
                          dtype=np.float32)

        self.B = np.random.uniform(-self.minBias, -self.maxBias, self.N)

        self.R = np.zeros((self.N, 1),
                          dtype=np.float32)

        self.padMat = t.zeros((batch, Nh + Nout), device=device)
        self.vm = t.zeros(self.R.shape,
                          dtype=t.float32,
                          device=device)

        self.inputIdc = np.array([i for i in range(self.Nin)], dtype=np.int64).reshape((self.Nin,))
        self.hiddenIdc = np.array([i + self.Nin for i in range(self.Nh)], dtype=np.int64).reshape(
            (self.Nh,))  # NOQA
        self.outputIdc = np.array([i + self.Nin + self.Nh for i in range(self.Nout)],
                                  dtype=np.int64).reshape(
            (self.Nout,))
        self.nodeIdc = np.concatenate([self.inputIdc, self.hiddenIdc, self.outputIdc], axis=0)

        self.construct_network()
        self.tensorize()

    def isOverlap(self, idx, v):
        assert self.outputZoneOffset > 0 and self.hiddenZoneOffset > 0

        for nodeIdx in self.nodeIdc:
            if idx != nodeIdx and self.magnitudeNumpy(v, self.P[nodeIdx]) <= 0.8:
                return True
        return False

    def drawVecSamples(self, zone: str = 'h'):
        start, stopWidth, stopHeight, stopDepth = 0, self.width, self.height, self.depth

        if zone != 'i':
            if zone == 'h':
                start = self.hiddenZoneOffset

            elif zone == 'o':
                start = self.hiddenZoneOffset + self.outputZoneOffset

            stopWidth = self.width - start
            stopHeight = self.height - start
            stopDepth = self.depth - start

        x = np.random.choice(np.arange(
            start=start, stop=stopWidth, dtype=np.double
        ), size=1)[0]

        y = np.random.choice(np.arange(
            start=start, stop=stopHeight, dtype=np.double
        ), size=1)[0]

        z = np.random.choice(np.arange(
            start=start, stop=stopDepth, dtype=np.double
        ), size=1)[0]

        return np.array([x, y, z], dtype=np.double)

    def getNodePosition(self, nodeIdx, zone: str = 'h'):
        v = self.drawVecSamples(zone)
        while self.isOverlap(nodeIdx, v):
            v = self.drawVecSamples(zone)
        return v

    def construct_network(self):
        print(f'Initialize Input Nodes')
        for nodeIdx in self.inputIdc:
            self.R[nodeIdx, 0] = np.random.randint(self.minRadius, self.maxRadius, 1)[0]
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='i')

        print(f'Initialize Hidden Nodes')
        for nodeIdx in self.hiddenIdc:
            self.R[nodeIdx, 0] = np.random.randint(self.minRadius, self.maxRadius, 1)[0]
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='h')

        print(f'Initialize Output Nodes')
        for nodeIdx in self.outputIdc:
            self.R[nodeIdx, 0] = 1
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='o')

        print('Initialize Weighted Matrix')
        if self.minInputPerNode != self.maxInputPerNode:
            numberOfInputConnection = np.random.randint(self.minInputPerNode, self.maxInputPerNode, 1)[0]
        else:
            numberOfInputConnection = self.minInputPerNode

        if self.minOutputPerNode != self.maxOutputPerNode:
            numberOfOutputConnection = np.random.randint(self.minOutputPerNode, self.maxOutputPerNode, 1)[0]
        else:
            numberOfOutputConnection = self.minOutputPerNode

        self.W, self.C = initialize(numberOfInputConnection,
                                    numberOfOutputConnection,
                                    self.N, self.Nin, self.Nout,
                                    self.inputIdc, self.hiddenIdc, self.outputIdc)

        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0]) \
                                       + self.N \
                                       + (self.N * 3)

        self.numberOfUsableParams = self.N ** 2 \
                                    + self.N \
                                    + (self.N * 3)

        print(f'+\tParams Usage: {(self.numberOfTrainableParams / self.numberOfUsableParams) * 100:.2f}%')
        print(f'+\tNumber of Trainable Parameters: {self.numberOfTrainableParams}')
        print(f'+\tNumber of Usable Parameters: {self.numberOfUsableParams}')

    def tensorize(self):
        self.W = t.nn.Parameter(
            t.tensor(self.W,
                     dtype=t.float32,
                     device=self.device).share_memory_(),
            requires_grad=True
        )

        self.B = t.nn.Parameter(
            t.tensor(self.B,
                     dtype=t.float32,
                     device=self.device).share_memory_(),
            requires_grad=True
        )

        self.P = t.nn.Parameter(
            t.tensor(self.P,
                     dtype=t.float32,
                     device=self.device).share_memory_(),
            requires_grad=True
        )

        self.R = t.nn.Parameter(
            t.tensor(self.R,
                     dtype=t.float32,
                     device=self.device).share_memory_(),
            requires_grad=True
        )

        self.C = t.tensor(self.C,
                          dtype=t.float32,
                          device=self.device).share_memory_()

    @staticmethod
    def magnitudeTorch(u, v):
        return t.sqrt(t.sum((u - v) ** 2, dim=1)).unsqueeze(dim=1)

    @staticmethod
    def magnitudeNumpy(u, v):
        return np.sqrt(np.sum((u - v) ** 2))

    def _forward(self, x):
        x = t.hstack((x, self.padMat))
        placeholder = self.O.clone()
        self.O = placeholder + x

        for hiddenIdx in self.hiddenIdc:
            self.vm = self.vm * 0.
            self.vm = self.vm + self.magnitudeTorch(self.P * self.C[:, hiddenIdx]
                                                    .unsqueeze(dim=1),
                                                    self.P[hiddenIdx])
            self.vm[hiddenIdx] = self.vm[hiddenIdx] + 1.

            placeholder = self.O.clone()
            matmul = placeholder @ (self.R * self.W[:, hiddenIdx]
                                    .unsqueeze(dim=1) / self.vm)

            placeholder = self.O[:, hiddenIdx].clone()
            self.O[:, hiddenIdx] = placeholder \
                                   + t.sigmoid(matmul + self.B[hiddenIdx]) \
                                       .squeeze(dim=1)

        for outputIdx in self.outputIdc:
            self.vm = self.vm * 0.
            self.vm = self.vm + self.magnitudeTorch(self.P * self.C[:, outputIdx]
                                                    .unsqueeze(dim=1),
                                                    self.P[outputIdx])
            self.vm[outputIdx] = self.vm[outputIdx] + 1.

            placeholder = self.O.clone()
            matmul = placeholder @ (self.R * self.W[:, outputIdx]
                                    .unsqueeze(dim=1) / self.vm)

            placeholder = self.O[:, outputIdx].clone()
            self.O[:, outputIdx] = placeholder \
                                   + t.sigmoid(matmul + self.B[outputIdx]) \
                                       .squeeze(dim=1)

        return self.O[:, self.outputIdc]

    def forward(self, x):
        return functionalize(self._forward)(x)


class SGD(t.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = ...):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * p.grad.data
