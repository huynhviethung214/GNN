import json
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, vectorize, types
from tqdm import tqdm

from modules.modules import magnitude, initialize, sigmoid
from modules.modules import forward as forwardCPU


@vectorize(['float32(float32)'], target='cuda')
def sigmoid(x):
    return 1. / (1. + math.exp(-x))


@vectorize(['float32(float32)'], target='cuda')
def sigmoidPrime(x):
    return (1. / (1. + math.exp(-x))) * (1. - (1. / (1. + math.exp(-x))))


@vectorize(['float32(float32, float32, int32)'], target='cuda')
def msePrime(u, v, Nout):
    return (2 / Nout) * (u - v)


@cuda.jit()
def calculateInputOfNeurons(batchSize, W, P, R, I, O, idx):
    x, y = cuda.grid(2)

    if x < batchSize and y < W.shape[0]:
        if W[y, idx] != 0. and y != idx:
            # Calculate synapse's length
            vm = math.sqrt(
                (P[idx, 0] - P[y, 0]) ** 2
                + (P[idx, 1] - P[y, 1]) ** 2
                + (P[idx, 2] - P[y, 2]) ** 2)

            # if vm == 0.:
            #     vm = 1.

            # if idx == 4 and x == 0:
            #     print('I', y, idx, O[x, y], W[y, idx], R[y, 0], vm)

            # print('I', y, idx, O[x, y], W[y, idx], R[y, 0], vm)
            I[x, idx] += (O[x, y] * W[y, idx] * R[y, 0]) / vm


@cuda.jit()
def calculateOutputOfNeurons(batchSize, O, I, B, idx):
    x = cuda.grid(1)

    if x < batchSize:
        # if idx == 4 and x == 0:
        #     print('O', x, idx, I[x, idx], B[idx])
        O[x, idx] = 1. / (1. + math.exp(-(I[x, idx] + B[idx])))


@cuda.jit()
def zero_io(batchSize, I, O):
    x, y = cuda.grid(2)

    if x < batchSize and y < O.shape[1]:
        I[x, y] = 0.
        O[x, y] = 0.


@cuda.jit()
def zero_grads(gradW, gradB, gradP, gradR, lossWRTHiddenOutput):
    x, y, z = cuda.grid(3)

    if x < gradW.shape[0] and y < gradW.shape[1] and z < gradW.shape[2]:
        gradW[x, y, z] = 0.
        gradB[x, y] = 0.

        gradP[x, y, 0] = 0.
        gradP[x, y, 1] = 0.
        gradP[x, y, 2] = 0.

        gradR[x, y, 0] = 0.
        lossWRTHiddenOutput[x, y] = 0.


@cuda.jit()
def gradBOutputToHidden(idx, outIdx, I, O, v, Nout, gradB):
    x = cuda.grid(1)

    if x < gradB.shape[0]:
        # if not x:
        #     print(x, idx, outIdx, I[x, idx], O[x, idx], v[x, outIdx], Nout)
        gradB[x, idx] = msePrime(O[x, idx], v[x, idx]) * sigmoidPrime(I[x, idx])

        # MSE Prime * Sigmoid Prime
        # gradB[x, idx] += (2. / Nout) * (O[x, idx] - v[x, outIdx]) \
        #                  * ((1. / (1. + math.exp(-I[x, idx])))
        #                     * (1. - (1. / (1. + math.exp(-I[x, idx])))))


@cuda.jit()
def remainGradsOutputToHidden(idx, Ch, O, W, P, R, gradW, gradB,
                              gradP, gradR, lossWRTHiddenOutput, C):
    x, y = cuda.grid(2)

    if x < gradW.shape[0] and y < W.shape[0]:
        # print(x, y, idx, W[y, idx])
        if W[y, idx] != 0. and y != idx:
            # Calculate synapse's length
            vm = math.sqrt((P[idx, 0] - P[y, 0]) ** 2
                           + (P[idx, 1] - P[y, 1]) ** 2
                           + (P[idx, 2] - P[y, 2]) ** 2)

            # if vm == 0.:
            #     vm = 1.

            # Ch[y, idx] determine whether if y is a Hidden Neuron or not
            lossWRTHiddenOutput[x, y] += (W[y, idx]
                                          * gradB[x, idx]
                                          * R[y, 0]
                                          * Ch[y, idx]
                                          / vm) \
                                         + (lossWRTHiddenOutput[x, y + 1]
                                            * Ch[y + 1, idx])
            # print(lossWRTHiddenOutput[x, y])

            gradW[x, y, idx] += O[x, y] \
                                * gradB[x, idx] \
                                * R[y, 0] \
                                * C[y, idx] \
                                / vm

            gradR[x, y, 0] += O[x, y] \
                              * gradB[x, idx] \
                              * W[y, idx] \
                              / vm

            # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
            x0 = P[idx, 0]
            y0 = P[idx, 1]
            z0 = P[idx, 2]

            x1 = P[y, 0]
            y1 = P[y, 1]
            z1 = P[y, 2]

            gradP[x, idx, 0] += O[x, y] \
                                * gradB[x, idx] \
                                * W[y, idx] \
                                * (R[y, 0]
                                   * -(x0 - x1)
                                   / (vm ** 3))

            gradP[x, idx, 1] += O[x, y] \
                                * gradB[x, idx] \
                                * W[y, idx] \
                                * (R[y, 0]
                                   * -(y0 - y1)
                                   / (vm ** 3))

            gradP[x, idx, 2] += O[x, y] \
                                * gradB[x, idx] \
                                * W[y, idx] \
                                * (R[y, 0]
                                   * -(z0 - z1)
                                   / (vm ** 3))

            gradP[x, y, 0] += O[x, y] \
                              * gradB[x, idx] \
                              * W[y, idx] \
                              * (R[y, 0]
                                 * (x0 - x1)
                                 / (vm ** 3))

            gradP[x, y, 1] += O[x, y] \
                              * gradB[x, idx] \
                              * W[y, idx] \
                              * (R[y, 0]
                                 * (y0 - y1)
                                 / (vm ** 3))

            gradP[x, y, 2] += O[x, y] \
                              * gradB[x, idx] \
                              * W[y, idx] \
                              * (R[y, 0]
                                 * (z0 - z1)
                                 / (vm ** 3))


@cuda.jit()
def remainGradsHiddenToInput(idx, O, W, P, R, gradW, gradB,
                             gradP, gradR, C):
    x, y = cuda.grid(2)

    if x < gradW.shape[0] and y < W.shape[0]:
        # Calculate synapse's length
        vm = math.sqrt((P[idx, 0] - P[y, 0]) ** 2
                       + (P[idx, 1] - P[y, 1]) ** 2
                       + (P[idx, 2] - P[y, 2]) ** 2)

        if vm == 0.:
            vm = 1.

        gradW[x, y, idx] += O[x, y] \
                            * gradB[x, idx] \
                            * R[y, 0] \
                            * C[y, idx] \
                            / vm

        gradR[x, y, 0] += O[x, y] \
                          * gradB[x, idx] \
                          * W[y, idx] \
                          / vm

        # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
        x0 = P[idx, 0]
        y0 = P[idx, 1]
        z0 = P[idx, 2]

        x1 = P[y, 0]
        y1 = P[y, 1]
        z1 = P[y, 2]

        gradP[x, idx, 0] += O[x, y] \
                            * gradB[x, idx] \
                            * W[y, idx] \
                            * (R[y, 0]
                               * -(x0 - x1)
                               / (vm ** 3))

        gradP[x, idx, 1] += O[x, y] \
                            * gradB[x, idx] \
                            * W[y, idx] \
                            * (R[y, 0]
                               * -(y0 - y1)
                               / (vm ** 3))

        gradP[x, idx, 2] += O[x, y] \
                            * gradB[x, idx] \
                            * W[y, idx] \
                            * (R[y, 0]
                               * -(z0 - z1)
                               / (vm ** 3))

        gradP[x, y, 0] += O[x, y] \
                          * gradB[x, idx] \
                          * W[y, idx] \
                          * (R[y, 0]
                             * (x0 - x1)
                             / (vm ** 3))

        gradP[x, y, 1] += O[x, y] \
                          * gradB[x, idx] \
                          * W[y, idx] \
                          * (R[y, 0]
                             * (y0 - y1)
                             / (vm ** 3))

        gradP[x, y, 2] += O[x, y] \
                          * gradB[x, idx] \
                          * W[y, idx] \
                          * (R[y, 0]
                             * (z0 - z1)
                             / (vm ** 3))


@cuda.jit()
def gradBHiddenToInput(batchSize, idx, I, lossWRTHiddenOutput, out):
    x = cuda.grid(1)

    if x < batchSize:
        out[x, idx] = lossWRTHiddenOutput[x, idx] \
                      * (1. / (1. + math.exp(-I[x, idx]))) \
                      * (1. - (1. / (1. + math.exp(-I[x, idx]))))  # Sigmoid Prime


class Network:
    def __init__(self,
                 train: bool = True, minBias: float = 0.4, maxBias: float = 0.8,
                 decay: float = 0.1, etaW: float = 1e-3, etaR: float = 1e-3,
                 etaP: float = 1e-3, etaB: float = 1e-3, minRadius: int = 10,
                 maxRadius: int = 20, frequency: int = 5, nInputs: int = 784,
                 nHiddens: int = 10, nOutputs: int = 10,  # NOQA
                 epochs: int = 10, datasetCap: int = 1000,
                 width: int = 20, height: int = 20, depth: int = 20,
                 hiddenZoneOffset: int = 400, outputZoneOffset: int = 400,
                 maxInputPerNode: int = 2, minInputPerNode: int = 1,
                 maxOutputPerNode: int = 20, minOutputPerNode: int = 1,
                 datasetName: str = 'mnist', activationFunc: str = 'sigmoid',
                 lossFunc: str = 'mse', batch: int = 800):  # NOQA
        assert lossFunc != '' and activationFunc != '', 'Please specify which type of loss function ' \
                                                        '/ activation function to use.'
        self.f = None
        self.g = None

        if train:
            self.losses = []
            self.batch = batch

            self.fPrime = None
            self.gPrime = None

            self.activationFunc = activationFunc
            self.lossFunc = lossFunc

            self.hiddenZoneOffset = hiddenZoneOffset
            self.outputZoneOffset = outputZoneOffset

            self.nInputs = nInputs
            self.nHiddens = nHiddens  # NOQA
            self.nOutputs = nOutputs
            self.N = nInputs + nHiddens + nOutputs  # Number of Nodes

            self.maxInputPerNode = maxInputPerNode
            self.minInputPerNode = minInputPerNode

            self.maxOutputPerNode = maxOutputPerNode
            self.minOutputPerNode = minOutputPerNode

            self.etaW = etaW
            self.etaB = etaB
            self.etaR = etaR
            self.etaP = etaP

            self.minBias = minBias
            self.maxBias = maxBias
            self.decay = decay

            self.width = width
            self.depth = depth
            self.height = height

            self.minRadius = minRadius
            self.maxRadius = maxRadius

            self.epochs = epochs
            self.frequency = frequency

            self.datasetCap = datasetCap
            self.datasetName = datasetName

            self.numberOfTrainableParams = 0
            self.numberOfUsableParams = 0

            self.W = None
            self.C = None
            self.Ch = None
            self.B = None
            self.P = None
            self.R = None

            self.gradU = None
            self.gradB = None
            self.gradR = None
            self.gradP = None

            self.I, self.O = None, None
            self.lossWRTHiddenOutput = None

            self.inputIdc, self.hiddenIdc, self.outputIdc = None, None, None
            self.nodeIdc = None

            self.createMatrices()
            self.construct_network()

    def createMatrices(self):
        self.W = np.zeros((self.N, self.N),
                          dtype=np.float64)
        self.Ch = np.zeros((self.N, self.N),
                           dtype=np.float64)
        self.P = np.zeros((self.N, 3),
                          dtype=np.float64)
        self.B = np.random.uniform(-self.minBias, -self.maxBias, self.N)

        self.R = np.zeros((self.N, 1),
                          dtype=np.float64)

        self.I = np.zeros((self.batch, self.N), dtype=np.float64)
        self.O = np.zeros((self.batch, self.N), dtype=np.float64)

        self.gradB = np.zeros((self.batch, self.N), dtype=np.float64)
        self.gradU = np.zeros((self.batch, self.N, self.N),
                              dtype=np.float64)
        self.gradR = np.zeros((self.batch, self.N, 1),
                              dtype=np.float64)
        self.gradP = np.zeros((self.batch, self.N, 3),
                              dtype=np.float64)
        self.lossWRTHiddenOutput = np.zeros((self.batch, self.N), dtype=np.float64)

        self.inputIdc = np.array([i for i in range(self.nInputs)], dtype=np.int64).reshape((self.nInputs,))
        self.hiddenIdc = np.array([i + self.nInputs for i in range(self.nHiddens)], dtype=np.int64).reshape(
            (self.nHiddens,))  # NOQA
        self.outputIdc = np.array([i + self.nInputs + self.nHiddens for i in range(self.nOutputs)],
                                  dtype=np.int64).reshape(
            (self.nOutputs,))
        self.nodeIdc = np.concatenate([self.inputIdc, self.hiddenIdc, self.outputIdc], axis=0)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for inputIdx in self.inputIdc:
            ax.plot(self.P[inputIdx, 0], self.P[inputIdx, 1], self.P[inputIdx, 2],
                    marker='o', color='r')

        for hiddenIdx in self.hiddenIdc:
            ax.plot(self.P[hiddenIdx, 0], self.P[hiddenIdx, 1], self.P[hiddenIdx, 2],
                    marker='o', color='g')

        for outputIdx in self.outputIdc:
            ax.plot(self.P[outputIdx, 0], self.P[outputIdx, 1], self.P[outputIdx, 2],
                    marker='o', color='b')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def save_config(self, trainingTime, accuracy, simulationFolderPath):
        configs = {
            'nInputs': self.nInputs,
            'nHiddens': self.nHiddens,  # NOQA
            'nOutputs': self.nOutputs,
            'N': self.nInputs + self.nHiddens + self.nOutputs,
            'maxInputPerNode': self.maxInputPerNode,
            'minInputPerNode': self.minInputPerNode,
            'maxOutputPerNode': self.maxOutputPerNode,
            'minOutputPerNode': self.minOutputPerNode,
            'etaW': self.etaW,
            'etaB': self.etaB,
            'etaR': self.etaR,
            'etaP': self.etaP,
            'decay': self.decay,
            'width': self.width,
            'depth': self.depth,
            'height': self.height,
            'minRadius': self.minRadius,
            'maxRadius': self.maxRadius,
            'minBias': self.minBias,
            'maxBias': self.maxBias,
            'epochs': self.epochs,
            'hiddenZoneOffset': self.hiddenZoneOffset,
            'outputZoneOffset': self.outputZoneOffset,
            'numberOfTrainableParams': int(self.numberOfTrainableParams),
            'numberOfUsableParams': int(self.numberOfUsableParams),
            'frequency': self.frequency,
            'datasetCap': self.datasetCap,
            'dataset': self.datasetName,
            'activationFunc': self.activationFunc,
            'trainingTime': trainingTime,
            'accuracy': accuracy,
            'losses': self.losses
        }

        with open(f'./{simulationFolderPath}/configs.json', 'w') as f:
            json.dump(configs, f, indent=4)

    def save_result(self, simulationFolderPath):
        np.save(f'./{simulationFolderPath}/w.npy', self.W)
        np.save(f'./{simulationFolderPath}/b.npy', self.B)
        np.save(f'./{simulationFolderPath}/r.npy', self.R)
        np.save(f'./{simulationFolderPath}/p.npy', self.P)

    def load_config(self, modelFolderPath):
        assert os.path.exists(f'./{modelFolderPath}/configs.json'), \
            f'./{modelFolderPath}/configs.json cannot be found'

        with open(f'./{modelFolderPath}/configs.json', 'r') as f:
            for key, value in json.load(f).items():
                setattr(self, key, value)

    def load_simulation(self, fp):
        print(f'Load Simulation From {fp}')
        assert os.path.exists(f'./{fp}'), f'./{fp} cannot be found'
        self.load_config(f'./{fp}')
        self.createMatrices()

        print('Load Matrices')
        self.W = np.load(f'./{fp}/w.npy')
        self.B = np.load(f'./{fp}/b.npy')
        self.R = np.load(f'./{fp}/r.npy')
        self.P = np.load(f'./{fp}/p.npy')

    def isOverlap(self, idx, v, zone: str = 'i'):
        assert self.outputZoneOffset > 0 and self.hiddenZoneOffset > 0

        for nodeIdx in self.nodeIdc:
            if idx != nodeIdx and magnitude(v, self.P[nodeIdx]) <= 0.8:
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
        while self.isOverlap(nodeIdx, v, zone):
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
                                    self.N, self.nInputs, self.nOutputs,
                                    self.inputIdc, self.hiddenIdc, self.outputIdc)

        self.Ch = np.copy(self.W)
        self.Ch[:, :self.nInputs] = 0.
        self.Ch[:, self.nInputs + self.nHiddens:] = 0.
        self.Ch[self.Ch != 0] = 1.

        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0]) \
                                       + self.N \
                                       + (self.N * 3)

        self.numberOfUsableParams = self.N ** 2 \
                                    + self.N \
                                    + (self.N * 3)

        self.W = cuda.to_device(self.W)
        self.C = cuda.to_device(self.C)
        self.Ch = cuda.to_device(self.Ch)
        self.B = cuda.to_device(self.B)
        self.P = cuda.to_device(self.P)
        self.R = cuda.to_device(self.R)

        self.I = cuda.to_device(self.I)
        self.O = cuda.to_device(self.O)

        self.gradU = cuda.to_device(self.gradU)
        self.gradB = cuda.to_device(self.gradB)
        self.gradP = cuda.to_device(self.gradP)
        self.gradR = cuda.to_device(self.gradR)
        self.lossWRTHiddenOutput = cuda.to_device(self.lossWRTHiddenOutput)

        print(f'+\tParams Usage: {(self.numberOfTrainableParams / self.numberOfUsableParams) * 100:.2f}%')
        print(f'+\tNumber of Trainable Parameters: {self.numberOfTrainableParams}')
        print(f'+\tNumber of Usable Parameters: {self.numberOfUsableParams}')

    def backward(self, v, epoch):
        for outputIdx in self.outputIdc:
            outIdx = outputIdx - (self.nInputs + self.nHiddens)
            gradBOutputToHidden[(32, 32), (32, 32)](self.batch, outputIdx,
                                                    outIdx, self.I, self.O,
                                                    v, self.nOutputs, self.gradB)

            remainGradsOutputToHidden[(32, 32), (32, 32)](outputIdx, self.Ch, self.O,
                                                          self.W, self.P, self.R,
                                                          self.gradU, self.gradB, self.gradP,
                                                          self.gradR, self.lossWRTHiddenOutput,
                                                          self.C)

        for hiddenIdx in self.hiddenIdc[::-1]:
            gradBHiddenToInput[(32, 32), (32, 32)](self.batch, hiddenIdx, self.I,
                                                   self.lossWRTHiddenOutput, self.gradB)
            remainGradsHiddenToInput[(32, 32), (32, 32)](hiddenIdx, self.O, self.W,
                                                         self.P, self.R, self.gradU,
                                                         self.gradB, self.gradP, self.gradR,
                                                         self.C)

        self.W = self.W.copy_to_host()
        self.B = self.B.copy_to_host()
        self.P = self.P.copy_to_host()
        self.R = self.R.copy_to_host()

        # print(f'W is nan: {np.isnan(self.W).any()}')
        # print(f'B is nan: {np.isnan(self.B).any()}')
        # print(f'P is nan: {np.isnan(self.P).any()}')
        # print(f'R is nan: {np.isnan(self.R).any()}')

        if (epoch + 1) % self.frequency == 0:
            self.etaW *= self.decay
            self.etaB *= self.decay
            self.etaP *= self.decay
            self.etaR *= self.decay

        self.W += -self.etaW * np.sum(self.gradU.copy_to_host(), axis=0) \
                  / self.batch

        self.B += -self.etaB * np.sum(self.gradB.copy_to_host(), axis=0) \
                  / self.batch

        self.P += -self.etaP * np.sum(self.gradP.copy_to_host(), axis=0) \
                  / self.batch

        self.R += -self.etaR * np.sum(self.gradR.copy_to_host(), axis=0) \
                  / self.batch

        self.W = cuda.to_device(self.W)
        self.B = cuda.to_device(self.B)
        self.P = cuda.to_device(self.P)
        self.R = cuda.to_device(self.R)

    def save_weight_image_per_epoch(self, epoch, simulationFolderPath):
        if not os.path.exists(f'{simulationFolderPath}/graphs'):
            os.mkdir(f'{simulationFolderPath}/graphs')
            os.mkdir(f'{simulationFolderPath}/graphs/W')

        WForPlot = self.W.copy()
        WForPlot = np.abs(WForPlot)

        plt.imsave(f'./{simulationFolderPath}/graphs/W/w{epoch}.jpg',
                   WForPlot, cmap='hot')

    def record_output_of_hidden_neurons(self, simulationFolderPath, label):
        if not os.path.exists(f'{simulationFolderPath}/graphs/O'):
            os.mkdir(f'{simulationFolderPath}/graphs/O')

        O = self.O.copy()
        O = O[self.nInputs: self.nInputs + self.nHiddens] \
            .reshape(self.nHiddens // 2, 2).T
        plt.imsave(f'./{simulationFolderPath}/graphs/O/{label}.jpg', O)

    def zero_grads(self):
        zero_grads[(32, 32), (16, 16)](self.gradU, self.gradB,
                                       self.gradP, self.gradR,
                                       self.lossWRTHiddenOutput)

    def zero_io(self):
        zero_io[(32, 32), (16, 16)](self.batch, self.I, self.O)

    def forward(self):
        for hiddenIdx in self.hiddenIdc:
            calculateInputOfNeurons[(32, 32), (32, 32)](self.batch, self.W,
                                                        self.P, self.R,
                                                        self.I, self.O, hiddenIdx)
            calculateOutputOfNeurons[(32, 32), (32, 32)](self.batch, self.O,
                                                         self.I, self.B, hiddenIdx)

        for outputIdx in self.outputIdc:
            calculateInputOfNeurons[(32, 32), (32, 32)](self.batch, self.W,
                                                        self.P, self.R,
                                                        self.I, self.O, outputIdx)
            calculateOutputOfNeurons[(32, 32), (32, 32)](self.batch, self.O,
                                                         self.I, self.B, outputIdx)

    def predict(self, u):
        W = self.W.copy_to_host()
        B = self.B.copy_to_host()
        P = self.P.copy_to_host()
        R = self.R.copy_to_host()

        I = np.zeros((self.N,), dtype=np.float32)
        O = np.zeros((self.N,), dtype=np.float32)

        I, O = forwardCPU(W, I, O, P, R, B,
                          self.inputIdc, self.hiddenIdc, self.outputIdc,
                          u, sigmoid)

        return O[self.nInputs + self.nHiddens:]

    def train(self, epochs, trainingSet):
        timeStart = time.time()
        for epochIdx in range(epochs):
            for batchIdx in tqdm(range(len(trainingSet[0])),
                                 desc=f'Epoch: {epochIdx} / {epochs}',
                                 leave=False):
                self.zero_grads()
                # self.zero_io()

                xs, ys = trainingSet[0][batchIdx], trainingSet[1][batchIdx]
                ys = cuda.to_device(ys)

                self.I = cuda.to_device(xs)
                self.O = cuda.to_device(self.I.copy_to_host())

                self.forward()
                cuda.synchronize()
                # predicts = self.O[:, min(self.outputIdc):].copy_to_host()
                self.backward(ys, epochIdx)
                cuda.synchronize()

        timeEnd = time.time()
        print(f'Total Forward + Backward Time: {timeEnd - timeStart} (s)')
