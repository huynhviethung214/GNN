import json
import os

import numpy as np
import matplotlib.pyplot as plt

from modules.modules import forward, calculateGradients, magnitude, \
    initialize, sigmoidPrime, reluPrime, sigmoid, relu, tanhPrime, tanh, cos, sin, mse, msePrime, crossEntropy, \
    crossEntropyPrime, absPrime, absFunc


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
                 lossFunc: str = 'mse'):  # NOQA
        assert lossFunc != '' and activationFunc != '', 'Please specify which type of loss function ' \
                                                        '/ activation function to use.'
        self.f = None
        self.g = None

        if train:
            self.losses = []

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

            self.getLossFunction()
            self.getActivationFunction()

    def createMatrices(self):
        self.W = np.zeros((self.N, self.N),
                          dtype=np.float64)  # Use to store size of every node in the self
        self.P = np.zeros((self.N, 3),
                          dtype=np.float64)  # Use to store position (x, y) of node in the self
        self.B = np.random.uniform(-self.minBias, -self.maxBias, self.N)

        self.R = np.zeros((self.N, 1),
                          dtype=np.float64)  # Use to store AoE of every node in the self
        self.I = np.zeros((self.N,), dtype=np.float64)
        self.O = np.zeros((self.N,), dtype=np.float64)

        self.gradB = np.zeros((self.N,), dtype=np.float64)
        self.gradU = np.zeros((self.N, self.N),
                              dtype=np.float64)  # Gradient U use to update W
        self.gradR = np.zeros((self.N, 1),
                              dtype=np.float64)  # Gradient S use to update the size of each node
        self.gradP = np.zeros((self.N, 3),
                              dtype=np.float64)  # Gradient P use to update position of each node
        self.lossWRTHiddenOutput = np.zeros((self.N,), dtype=np.float64)

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

        # if zone == 'i':
        #     zoneNodeIdc = self.inputIdc
        # elif zone == 'h':
        #     zoneNodeIdc = self.hiddenIdc
        # else:
        #     zoneNodeIdc = self.outputIdc

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

        self.W, _ = initialize(numberOfInputConnection,
                               numberOfOutputConnection,
                               self.N, self.nInputs, self.nOutputs,
                               self.inputIdc, self.hiddenIdc, self.outputIdc)

        # self.W = fixSynapses(self.W, self.P, self.maxRadius, self.nodeIdc)

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

    def getActivationFunction(self):
        if self.activationFunc == 'sigmoid':
            self.fPrime = sigmoidPrime
            self.f = sigmoid

        elif self.activationFunc == 'relu':
            self.fPrime = reluPrime
            self.f = relu

        elif self.activationFunc == 'tanh':
            self.fPrime = tanhPrime
            self.f = tanh

        elif self.activationFunc == 'sin':
            self.fPrime = cos
            self.f = sin

    def getLossFunction(self):
        if self.lossFunc == 'mse':
            self.gPrime = msePrime
            self.g = mse

        elif self.lossFunc == 'ce':
            self.gPrime = crossEntropyPrime
            self.g = crossEntropy

        elif self.lossFunc == 'abs':
            self.gPrime = absPrime
            self.g = absFunc

    def zeros_io(self):
        self.I *= 0
        self.O *= 0

    def zeros_grad(self):
        self.gradU *= 0
        self.gradB *= 0
        self.gradR *= 0
        self.gradP *= 0
        self.lossWRTHiddenOutput *= 0

    def update_params(self, v, etaW, etaB, etaP, etaR):
        self.gradU, self.gradB, self.gradR, self.gradP = calculateGradients(
            v, self.W, self.I,
            self.O, self.P, self.R,
            self.gradU, self.gradR, self.gradP, self.gradB,
            self.nInputs, self.nHiddens, self.nOutputs,
            self.hiddenIdc, self.outputIdc,
            self.fPrime, self.gPrime, self.lossWRTHiddenOutput
        )

        self.W += -etaW * self.gradU
        self.B += -etaB * self.gradB
        self.P += -etaP * self.gradP
        self.R += -etaR * self.gradR

    def save_weight_image_per_epoch(self, epoch, simulationFolderPath):
        if not os.path.exists(f'{simulationFolderPath}/graphs'):
            os.mkdir(f'{simulationFolderPath}/graphs')
            os.mkdir(f'{simulationFolderPath}/graphs/W')

        # if (epoch + 1) % self.frequency == 0:
        WForPlot = self.W.copy()
        WForPlot = np.abs(WForPlot)

        # try:
        #     plt.imsave(f'./{simulationFolderPath}/graphs/W/w{epoch}.jpg',
        #                WForPlot[:, self.nInputs: self.nInputs + self.nHiddens].T
        #                , cmap='hot')
        # except Exception:
        #     plt.imsave(f'./{simulationFolderPath}/graphs/W/w{epoch}.jpg',
        #                WForPlot, cmap='hot')

        plt.imsave(f'./{simulationFolderPath}/graphs/W/w{epoch}.jpg',
                   WForPlot, cmap='hot')

    def record_output_of_hidden_neurons(self, simulationFolderPath, label):
        if not os.path.exists(f'{simulationFolderPath}/graphs/O'):
            os.mkdir(f'{simulationFolderPath}/graphs/O')

        O = self.O.copy()
        O = O[self.nInputs: self.nInputs + self.nHiddens] \
            .reshape(self.nHiddens // 2, 2).T
        plt.imsave(f'./{simulationFolderPath}/graphs/O/{label}.jpg', O)

    def predict(self, u):
        self.zeros_io()
        self.I, self.O = forward(self.W, self.I, self.O, self.P, self.R, self.B,
                                 self.inputIdc, self.hiddenIdc, self.outputIdc, u, self.f)

        return self.O[self.nInputs + self.nHiddens:].reshape((self.nOutputs,))
