import json
import os

import numpy as np
import matplotlib.pyplot as plt

from newInitializationMethod import forward, calculateGradients, magnitude, \
    initialize, sigmoidPrime, \
    reluPrime, sigmoid, relu, tanhPrime, tanh, cos, sin


class Network:
    def __init__(self,
                 train: bool = True, bias: float = 1, decay: float = 0.1,
                 etaW: float = 1e-3, etaR: float = 1e-3, etaP: float = 1e-3,
                 minRadius: int = 10, maxRadius: int = 20, frequency: int = 5,
                 nInputs: int = 784, nHiddens: int = 10, nOutputs: int = 10,  # NOQA
                 epochs: int = 10, datasetCap: int = 1000,
                 width: int = 20, height: int = 20, depth: int = 20,
                 hiddenZoneOffset: int = 400, outputZoneOffset: int = 400,
                 maxInputPerNode: int = 2, minInputPerNode: int = 1,
                 maxOutputPerNode: int = 20, minOutputPerNode: int = 1,
                 datasetName: str = 'mnist', activationFunc: str = 'sigmoid',
                 lossFunc: str = 'cross_entropy'):  # NOQA
        if train:
            self.fPrime = None
            self.f = None

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
            self.etaR = etaR
            self.etaP = etaP

            self.bias = bias
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

            self.createMatrices()
            self.construct_network()

    def createMatrices(self):
        self.W = np.zeros((self.N, self.N),
                          dtype=np.float64)  # Use to store size of every node in the self
        self.P = np.zeros((self.N, 3),
                          dtype=np.float64)  # Use to store position (x, y) of node in the self

        self.R = np.zeros((self.N, 1),
                          dtype=np.float64)  # Use to store AoE of every node in the self
        self.I = np.zeros((self.N,), dtype=np.float64)
        self.O = np.zeros((self.N,), dtype=np.float64)

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
            'etaR': self.etaR,
            'etaP': self.etaP,
            'bias': self.bias,
            'decay': self.decay,
            'width': self.width,
            'depth': self.depth,
            'height': self.height,
            'minRadius': self.minRadius,
            'maxRadius': self.maxRadius,
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
            'accuracy': accuracy
        }

        with open(f'./{simulationFolderPath}/configs.json', 'w') as f:
            json.dump(configs, f, indent=4)

    def save_result(self, simulationFolderPath):
        np.save(f'./{simulationFolderPath}/w.npy', self.W)
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
        self.R = np.load(f'./{fp}/r.npy')
        self.P = np.load(f'./{fp}/p.npy')

    def isOverlap(self, idx, v, zone: str = 'i'):
        assert self.outputZoneOffset > 0 and self.hiddenZoneOffset > 0

        if zone == 'i':
            zoneNodeIdc = self.inputIdc
        elif zone == 'h':
            zoneNodeIdc = self.hiddenIdc
        else:
            zoneNodeIdc = self.outputIdc

        for nodeIdx in zoneNodeIdc:
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
        self.W, _ = initialize(np.random.randint(self.minInputPerNode, self.maxInputPerNode, 1)[0],
                               np.random.randint(self.minOutputPerNode, self.maxOutputPerNode, 1)[0],
                               self.N, self.nInputs, self.nOutputs,
                               self.inputIdc, self.hiddenIdc, self.outputIdc)

        # self.W = fixSynapses(self.W, self.P, self.maxRadius, self.nodeIdc)

        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0]) \
                                       + self.R.shape[0] \
                                       + self.P.shape[0]

        self.numberOfUsableParams = np.square(self.N) \
                                    - self.N \
                                    - self.nInputs * self.N \
                                    - self.nOutputs * (self.N - self.nInputs) \
                                    + self.R.shape[0] \
                                    + self.P.shape[0]

        print(f'Params Usage: {(self.numberOfTrainableParams / self.numberOfUsableParams) * 100:.2f}%')
        print(f'Number of Trainable Parameters: {self.numberOfTrainableParams}')
        print(f'Number of Usable Parameters: {self.numberOfUsableParams}')

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

    def zeros_io(self):
        self.I *= 0
        self.O *= 0

    def zeros_grad(self):
        self.gradU *= 0
        self.gradR *= 0
        self.gradP *= 0
        self.lossWRTHiddenOutput *= 0

    def update_params(self, loss, etaW, etaP, etaR):
        self.gradU, self.gradR, self.gradP = calculateGradients(
            loss, self.W, self.I,
            self.O, self.P, self.R,
            self.gradU, self.gradR, self.gradP,
            self.nInputs, self.nHiddens, self.nOutputs,
            self.hiddenIdc, self.outputIdc,
            self.fPrime, self.lossWRTHiddenOutput
        )

        self.W += -etaW * self.gradU
        self.P += -etaP * self.gradP
        self.R += -etaR * self.gradR

    def save_weight_image_per_epoch(self, epoch, simulationFolderPath):
        if not os.path.exists(f'{simulationFolderPath}/graphs'):
            os.mkdir(f'{simulationFolderPath}/graphs')

        if (epoch + 1) % self.frequency == 0:
            WForPlot = self.W.copy()
            WForPlot = np.abs(WForPlot)

            plt.imsave(f'./{simulationFolderPath}/graphs/w{epoch}.jpg',
                       WForPlot[:, self.nInputs: self.nInputs + self.nHiddens].T
                       , cmap='hot')

    def predict(self, u,
                inputPerNetwork, hiddenPerNetwork, outputPerNetwork,
                numSubNetworks: int = 3):
        self.getActivationFunction()
        self.zeros_io()

        self.I, self.O = forward(self.W, self.I, self.O, self.P, self.R,
                                 self.nodeIdc, self.inputIdc,
                                 self.hiddenIdc, self.outputIdc,
                                 u, self.bias, self.f, numSubNetworks,
                                 inputPerNetwork, hiddenPerNetwork, outputPerNetwork)

        return self.O[min(self.outputIdc):].reshape((self.nOutputs,))
