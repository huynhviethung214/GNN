import json
import os

import numpy as np
import matplotlib.pyplot as plt

from modules.modules import forwardInput, calculateGradients, magnitude, \
    initialize, sigmoidPrime, reluPrime, sigmoid, relu, tanhPrime, tanh, cos, sin, mse, msePrime, crossEntropy, \
    crossEntropyPrime, mae, maePrime, butterworth, butterworthPrime, forwardOutput, \
    recallMemory, saveMemory, forwardHidden, calculateGradientBetweenStates


class Network:
    def __init__(self,
                 train: bool = True, etaR: float = 1e-3,
                 etaW: float = 1e-3, etaD: float = 1e-3,
                 etaP: float = 1e-3, etaB: float = 1e-3,
                 frequency: int = 5, decay: float = 0.1,
                 minBias: float = 0.4, maxBias: float = 0.8,
                 minRadius: float = 10, maxRadius: float = 20,
                 minDivision: float = 0.1, maxDivision: float = 10,
                 Nin: int = 784, Nh: int = 10, Nout: int = 10,  # NOQA
                 epochs: int = 10, datasetCap: int = 1000, numberOfStates: int = 1,
                 width: float = 20, height: float = 20, depth: float = 20,
                 hiddenZoneOffset: float = 400, outputZoneOffset: float = 400,
                 maxInputPerNode: int = 2, minInputPerNode: int = 1,
                 maxOutputPerNode: int = 20, minOutputPerNode: int = 1,
                 datasetName: str = 'mnist', activationFunc: str = 'sigmoid',
                 enableInputActivation: bool = False, enableLoss: bool = True,
                 O: tuple = (0, 0, 0), lossFunc: str = 'mse', nRanges: int = 1,
                 nClasses: int = 0, imageShape: list = None, inputShape: list = None,
                 stride: list = None, imagesPerStackedImage: int = 0):  # NOQA
        assert lossFunc != '' and activationFunc != '', 'Please specify which type of loss function ' \
                                                        '/ activation function to use.'
        self.f = None
        self.g = None

        if train:
            self.losses = []
            self.O = O

            self.fPrime = None
            self.gPrime = None

            self.activationFunc = activationFunc
            self.enableInputActivation = enableInputActivation
            self.enableLoss = enableLoss

            self.lossFunc = lossFunc

            self.hiddenZoneOffset = hiddenZoneOffset
            self.outputZoneOffset = outputZoneOffset

            self.inputShape = inputShape
            self.imageShape = imageShape
            self.stride = stride
            self.imagesPerStackedImage = imagesPerStackedImage

            self.nClasses = nClasses
            self.Nin = Nin
            self.Nh = Nh  # NOQA
            self.Nout = Nout
            self.N = Nin + Nh + Nout  # Number of Nodes
            self.nRanges = nRanges

            self.maxInputPerNode = maxInputPerNode
            self.minInputPerNode = minInputPerNode

            self.maxOutputPerNode = maxOutputPerNode
            self.minOutputPerNode = minOutputPerNode

            self.etaW = etaW
            self.etaD = etaD
            self.etaB = etaB
            self.etaR = etaR
            self.etaP = etaP

            self.decay = decay

            self.width = width
            self.depth = depth
            self.height = height

            self.minRadius = minRadius
            self.maxRadius = maxRadius

            self.minBias = minBias
            self.maxBias = maxBias

            self.minDivision = minDivision
            self.maxDivision = maxDivision

            self.epochs = epochs
            self.frequency = frequency

            self.datasetCap = datasetCap
            self.datasetName = datasetName

            self.numberOfTrainableParams = 0
            self.numberOfUsableParams = 0

            self.inputPerNode = 0
            self.outputPerNode = 0

            self.wE = 0
            self.dE = 0
            self.bE = 0
            self.pE = 0
            self.rE = 0

            self.numberOfStates = numberOfStates
            self.C = None
            self.Ch = None

            # self.temp = np.zeros((self.Nh,))

            self.W = None
            self.Wh = None
            self.Rh = None
            self.D = None
            self.B = None
            self.P = None
            self.R = None

            self.locks = np.zeros((self.N,))

            self.gradW = None
            self.gradWh = None
            self.gradRh = None
            self.gradD = None
            self.gradB = None
            self.gradR = None
            self.gradP = None

            self.I, self.O = None, None
            self.Ih, self.Oh = None, None
            self.stateFlags = None

            self.lossWRTHiddenOutput = None
            self.lossWRTInputOutput = None
            self.lossWRTOutputOutput = None

            self.inputIdc, self.hiddenIdc, self.outputIdc = None, None, None
            self.nodeIdc = None

            self.createMatrices()
            self.construct_network()

            self.getLossFunction()
            self.getActivationFunction()

    def createMatrices(self):
        self.W = np.zeros((self.N, self.N),
                          dtype=np.float64)  # Use to store size of every node in the self

        # self.Wh = np.zeros((self.Nh, self.Nh), dtype=np.float64)

        self.P = np.zeros((self.N, 3),
                          dtype=np.float64)  # Use to store position (x, y) of node in the self

        self.B = np.random.uniform(self.minBias, self.maxBias, (self.N,))

        self.D = np.random.uniform(self.minDivision, self.maxDivision, (self.N,))
        self.D[self.D <= 0.] = self.minDivision

        # self.Rh = np.zeros((self.numberOfStates, self.N), dtype=np.float64)

        self.R = np.zeros((self.N, 1),
                          dtype=np.float64)  # Use to store AoE of every node in the self

        self.I = np.zeros((self.N,), dtype=np.float64)
        self.O = np.zeros((self.N,), dtype=np.float64)
        self.Oh = np.zeros((self.numberOfStates, self.N), dtype=np.float64)
        self.Ih = np.zeros((self.numberOfStates, self.N), dtype=np.float64)

        self.gradW = np.zeros((self.N, self.N),
                              dtype=np.float64)  # Gradient U use to update W
        self.gradD = np.zeros((self.N,), dtype=np.float64)
        self.gradB = np.zeros((self.N,), dtype=np.float64)
        self.gradP = np.zeros((self.N, 3),
                              dtype=np.float64)  # Gradient P use to update position of each node
        self.gradR = np.zeros((self.N, 1),
                              dtype=np.float64)  # Gradient S use to update the size of each node
        self.lossWRTInputOutput = np.zeros((self.Nin,), dtype=np.float64)
        self.lossWRTHiddenOutput = np.zeros((self.numberOfStates, self.Nh),
                                            dtype=np.float64)
        self.lossWRTOutputOutput = np.zeros((self.Nout,), dtype=np.float64)

        # # Hold lower and upper limit of a single hidden block
        # self.inputLim = np.zeros((self.nRanges, 2), dtype=np.int32)
        # self.hiddenLim = np.zeros((self.nRanges, 2), dtype=np.int32)
        # self.outputLim = np.zeros((self.nRanges, 2), dtype=np.int32)
        #
        # # Number of Input / Hidden Node per Block
        # assert self.Nin % self.nRanges == 0. or self.Nh % self.nRanges == 0., \
        #     'Both Nin / Nh must be divisible by number of nRanges'
        # nodePerInputLim = self.Nin / self.nRanges
        # nodePerHiddenLim = self.Nh / self.nRanges
        # nodePerOutputLim = self.Nout / self.nRanges
        #
        # for limIndex in range(self.nRanges):
        #     self.inputLim[limIndex, 0] = limIndex * nodePerInputLim
        #     self.inputLim[limIndex, 1] = (limIndex + 1) * nodePerInputLim
        #
        #     self.hiddenLim[limIndex, 0] = limIndex * nodePerHiddenLim + self.Nin
        #     self.hiddenLim[limIndex, 1] = (limIndex + 1) * nodePerHiddenLim + self.Nin
        #
        #     self.outputLim[limIndex, 0] = limIndex * nodePerOutputLim + self.Nin + self.Nh
        #     self.outputLim[limIndex, 1] = (limIndex + 1) * nodePerOutputLim + self.Nin + self.Nh

        self.inputIdc = np.array([i for i in range(self.Nin)], dtype=np.int64).reshape((self.Nin,))
        self.hiddenIdc = np.array([i + self.Nin for i in range(self.Nh)], dtype=np.int64).reshape(
            (self.Nh,))  # NOQA
        self.outputIdc = np.array([i + self.Nin + self.Nh for i in range(self.Nout)],
                                  dtype=np.int64).reshape(
            (self.Nout,))
        self.nodeIdc = np.concatenate([self.inputIdc, self.hiddenIdc, self.outputIdc], axis=0)

    def plot(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        #
        # for inputIdx in self.inputIdc:
        #     ax.plot(self.P[inputIdx, 0], self.P[inputIdx, 1], self.P[inputIdx, 2],
        #             marker='o', color='r')
        #
        # for hiddenIdx in self.hiddenIdc:
        #     ax.plot(self.P[hiddenIdx, 0], self.P[hiddenIdx, 1], self.P[hiddenIdx, 2],
        #             marker='o', color='g')
        #
        # for outputIdx in self.outputIdc:
        #     ax.plot(self.P[outputIdx, 0], self.P[outputIdx, 1], self.P[outputIdx, 2],
        #             marker='o', color='b')

        plt.title(f'Nin: {self.Nin} | Nh: {self.Nh} | Nout: {self.Nout}')
        plt.plot([0, self.N], [self.Nin - 1, self.Nin - 1], lw=1, color='black')
        plt.plot([0, self.N], [self.Nin + self.Nh - 1, self.Nin + self.Nh - 1], lw=1, color='black')
        plt.plot([self.Nin - 1, self.Nin - 1], [0, self.N], lw=1, color='black')
        plt.plot([self.Nin + self.Nh - 1, self.Nin + self.Nh - 1], [0, self.N], lw=1, color='black')
        plt.imshow(self.W)
        plt.colorbar()
        plt.show()

        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

    def save_config(self, trainingTime, accuracy, simulationFolderPath):
        with open(f'./{simulationFolderPath}/configs.json', 'w') as f:
            configs = {
                'Nin': self.Nin,
                'Nh': self.Nh,
                'Nout': self.Nout,
                'N': self.Nin + self.Nh + self.Nout,
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
                'minDivision': self.minDivision,
                'maxDivision': self.maxDivision,
                'numberOfStates': self.numberOfStates,
                'enableInputActivation': self.enableInputActivation,
                'enableLoss': self.enableLoss,
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
                'imagesPerStackedImage': self.imagesPerStackedImage,
                'inputShape': self.inputShape,
                'imageShape': self.imageShape,
                'stride': self.stride,
                'losses': self.losses
            }
            json.dump(configs, f, indent=4)

    def save_result(self, simulationFolderPath):
        np.save(f'{simulationFolderPath}/w.npy', self.W)
        np.save(f'{simulationFolderPath}/wh.npy', self.Wh)
        np.save(f'{simulationFolderPath}/ch.npy', self.Ch)
        np.save(f'{simulationFolderPath}/d.npy', self.D)
        np.save(f'{simulationFolderPath}/b.npy', self.B)
        np.save(f'{simulationFolderPath}/p.npy', self.P)
        np.save(f'{simulationFolderPath}/r.npy', self.R)

    def load_config(self, modelFolderPath):
        assert os.path.exists(f'./{modelFolderPath}/configs.json'), \
            f'./{modelFolderPath}/configs.json cannot be found'

        with open(f'./{modelFolderPath}/configs.json', 'r') as f:
            for key, value in json.load(f).items():
                setattr(self, key, value)

    def load_simulation(self, fp):
        print(f'Load Simulation From {fp}')
        assert os.path.exists(f'{fp}'), f'{fp} cannot be found'
        self.load_config(f'{fp}')
        self.createMatrices()

        print('Load Matrices')
        self.W = np.load(f'{fp}/w.npy')
        self.Wh = np.load(f'{fp}/wh.npy')
        self.Ch = np.load(f'{fp}/ch.npy')
        self.D = np.load(f'{fp}/d.npy')
        self.B = np.load(f'{fp}/b.npy')
        self.P = np.load(f'{fp}/p.npy')
        self.R = np.load(f'{fp}/r.npy')

        self.C = np.copy(self.W)
        self.C[self.C != 0.] = 1.

    def isOverlap(self, idx, v):
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

        x = np.random.uniform(self.O[0] + start, self.O[0] + stopWidth, 1)[0]
        y = np.random.uniform(self.O[1] + start, self.O[1] + stopHeight, 1)[0]
        z = np.random.uniform(self.O[2] + start, self.O[2] + stopDepth, 1)[0]

        return np.array([x, y, z], dtype=np.double)

    def getNodePosition(self, nodeIdx, zone: str = 'h'):
        v = self.drawVecSamples(zone)
        while self.isOverlap(nodeIdx, v):
            v = self.drawVecSamples(zone)

        return v

    def add_hidden_neuron(self, n: int = 0):
        if not n:
            n = np.random.randint(self.Nh // 4, self.Nh, 1)[0]
        # oldNh = self.Nh
        randomNodeIdc = np.random.choice(self.hiddenIdc, n)

        for randomNodeIdx in randomNodeIdc:
            maxHiddenIndex = np.max(self.hiddenIdc)
            self.hiddenIdc = np.append(self.hiddenIdc, maxHiddenIndex + 1)
            self.locks = np.append(self.locks, 0.)
            # self.temp = np.append(self.temp, 0.)
            self.outputIdc += 1

            Wr = np.copy(self.W[randomNodeIdx, :]).reshape((1, self.N))
            Wr[0, :self.Nin] = 0.
            # Wr[0, np.random.randint(0, self.N, (self.Nh - 1) // 2)] = 0.
            # Wr *= 0.

            Wc = np.copy(self.W[:, randomNodeIdx])
            # Wc[np.random.randint(0, self.N + 1, (self.Nh - 1) // 2)] = 0.
            Wc[self.Nin + self.Nh:] = 0.
            lastRows = Wc[randomNodeIdx + 1:]
            Wc = np.hstack((Wc[:randomNodeIdx + 1], [0]))
            Wc = np.hstack((Wc, lastRows)).reshape((self.N + 1, 1))

            D = np.copy(self.D[randomNodeIdx]) + 0.05
            B = np.copy(self.B[randomNodeIdx])
            P = np.copy(self.P[randomNodeIdx]) + 50.
            R = np.copy(self.R[randomNodeIdx])

            lastCols = self.W[maxHiddenIndex + 1:, :]
            self.W = np.vstack((self.W[:maxHiddenIndex + 1, :], Wr))
            self.W = np.vstack((self.W, lastCols))

            lastRows = self.W[:, maxHiddenIndex + 1:]
            self.W = np.hstack((self.W[:, :maxHiddenIndex + 1], Wc))
            self.W = np.hstack((self.W, lastRows))

            np.fill_diagonal(self.W, 0.)
            self.W[self.Nin + self.Nh:] = 0.
            self.W[:, :self.Nin + 1] = 0.

            for key in self.Wh.keys():
                Wr = np.copy(self.Wh[key][randomNodeIdx, :]).reshape((1, self.N))
                Wr[0, :self.Nin] = 0.

                Wc = np.copy(self.Wh[key][:, randomNodeIdx])
                Wc[self.Nin + self.Nh:] = 0.
                lastRows = Wc[randomNodeIdx + 1:]
                Wc = np.hstack((Wc[:randomNodeIdx + 1], [0]))
                Wc = np.hstack((Wc, lastRows)).reshape((self.N + 1, 1))

                lastCols = self.Wh[key][maxHiddenIndex + 1:, :]
                self.Wh[key] = np.vstack((self.Wh[key][:maxHiddenIndex + 1, :], Wr))
                self.Wh[key] = np.vstack((self.Wh[key], lastCols))

                lastRows = self.Wh[key][:, maxHiddenIndex + 1:]
                self.Wh[key] = np.hstack((self.Wh[key][:, :maxHiddenIndex + 1], Wc))
                self.Wh[key] = np.hstack((self.Wh[key], lastRows))

                np.fill_diagonal(self.Wh[key], 0.)
                self.Wh[key][self.Nin + self.Nh:] = 0.
                self.Wh[key][:, :self.Nin + 1] = 0.

            self.C = np.copy(self.W)
            self.C[self.C != 0.] = 1.

            lastCols = self.D[maxHiddenIndex + 1:]
            self.D = np.hstack((self.D[:maxHiddenIndex + 1], D))
            self.D = np.hstack((self.D, lastCols))

            lastCols = self.B[maxHiddenIndex + 1:]
            self.B = np.hstack((self.B[:maxHiddenIndex + 1], B))
            self.B = np.hstack((self.B, lastCols))

            lastCols = self.P[maxHiddenIndex + 1:, :]
            self.P = np.vstack((self.P[:maxHiddenIndex + 1, :], P))
            self.P = np.vstack((self.P, lastCols))

            lastCols = self.R[maxHiddenIndex + 1:, :]
            self.R = np.vstack((self.R[:maxHiddenIndex + 1, :], R))
            self.R = np.vstack((self.R, lastCols))

            # Update Experience Matrices
            # Wr = np.zeros((1, self.N))
            # Wc = np.zeros((self.N + 1, 1))
            #
            # bZeros = np.zeros((1,))
            # pZeros = np.zeros((1, 3))
            # rZeros = np.zeros((1, 1))
            #
            # lastRows = self.wE[maxHiddenIndex + 1:, :]
            # self.wE = np.vstack((self.wE[:maxHiddenIndex + 1, :], Wr))
            # self.wE = np.vstack((self.wE, lastRows))
            #
            # lastCols = self.wE[:, maxHiddenIndex + 1:]
            # self.wE = np.hstack((self.wE[:, :maxHiddenIndex + 1], Wc))
            # self.wE = np.hstack((self.wE, lastCols))
            #
            # lastRows = self.bE[maxHiddenIndex + 1:]
            # self.bE = np.hstack((self.bE[:maxHiddenIndex + 1], bZeros))
            # self.bE = np.hstack((self.bE, lastRows))
            #
            # lastRows = self.pE[maxHiddenIndex + 1:, :]
            # self.pE = np.vstack((self.pE[:maxHiddenIndex + 1, :], pZeros))
            # self.pE = np.vstack((self.pE, lastRows))
            #
            # lastRows = self.rE[maxHiddenIndex + 1:, :]
            # self.rE = np.vstack((self.rE[:maxHiddenIndex + 1, :], rZeros))
            # self.rE = np.vstack((self.rE, lastRows))

            self.N += 1
            self.Nh += 1
            self.Ih = np.zeros((self.numberOfStates, self.N))
            self.Oh = np.zeros((self.numberOfStates, self.N))

            # self.I = np.zeros((self.N,))
            # self.O = np.zeros((self.N,))
            # print(len(self.Wh.keys()))
            for key in self.Wh.keys():
                self.gradWh[key] = np.zeros_like(self.Wh[key])

            self.gradW = np.zeros((self.N, self.N))
            self.gradD = np.zeros((self.N,))
            self.gradB = np.zeros((self.N,))
            self.gradP = np.zeros((self.N, 3))
            self.gradR = np.zeros((self.N, 1))
            self.lossWRTHiddenOutput = np.zeros((self.numberOfStates, self.Nh))
            # self.lossWRTOutputOutput = np.zeros((self.Nout,))
            self.nodeIdc = np.concatenate([self.inputIdc, self.hiddenIdc, self.outputIdc])

        return n

    def construct_network(self):
        # print(f'Initialize Input Nodes')
        for nodeIdx in self.inputIdc:
            self.R[nodeIdx, 0] = np.random.uniform(self.minRadius, self.maxRadius, 1)[0]
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='i')

        # print(f'Initialize Hidden Nodes')
        for nodeIdx in self.hiddenIdc:
            self.R[nodeIdx, 0] = np.random.uniform(self.minRadius, self.maxRadius, 1)[0]
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='h')

        # print(f'Initialize Output Nodes')
        for nodeIdx in self.outputIdc:
            self.R[nodeIdx, 0] = np.random.uniform(self.minRadius, self.maxRadius, 1)[0]
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='o')

        # print('Initialize Weighted Matrix')
        if self.minInputPerNode != self.maxInputPerNode:
            self.inputPerNode = np.random.randint(self.minInputPerNode, self.maxInputPerNode, 1)[0]
        else:
            self.inputPerNode = self.minInputPerNode

        if self.minOutputPerNode != self.maxOutputPerNode:
            self.outputPerNode = np.random.randint(self.minOutputPerNode, self.maxOutputPerNode, 1)[0]
        else:
            self.outputPerNode = self.minOutputPerNode

        self.W, self.C = initialize(self.inputPerNode, self.outputPerNode,
                                    self.N, self.Nin, self.Nout,
                                    self.inputIdc, self.hiddenIdc, self.outputIdc)

        # self.Ch = np.zeros((self.numberOfStates, self.numberOfStates))
        # self.Ch[0, 1] = 1.
        # self.Ch[1, 0] = 1.
        # self.Wh = np.zeros((self.numberOfStates, self.numberOfStates,
        #                     self.N, self.N))

        # self.Rh = np.random.uniform(0., 1., (self.numberOfStates, self.N))
        # self.stateFlags = np.zeros((self.numberOfStates,))

        self.Ch = np.zeros((self.numberOfStates, self.numberOfStates))
        for i in range(self.numberOfStates):
            for j in range(self.numberOfStates):
                if j > i:
                    self.Ch[i, j] = 1
                    break

        self.Wh = {}
        self.gradWh = {}
        for stateIdx in range(self.numberOfStates):
            byStateIdc = np.where(self.Ch[:, stateIdx] == 1)[0]
            for byStateIdx in byStateIdc:
                self.Wh.update({
                    f'{byStateIdx}{stateIdx}':
                        np.random.uniform(-np.sqrt(6 / (self.Nin * self.Nout)),
                                          +np.sqrt(6 / (self.Nin * self.Nout)),
                                          (self.N, self.N))
                })

                self.gradWh.update({
                    f'{byStateIdx}{stateIdx}': np.zeros((self.N, self.N))
                })

        # for stateIdx in range(self.numberOfStates):
        #     byStateIdc = np.where(self.Ch[:, stateIdx] == 1)[0]
        #     for byStateIdx in byStateIdc:
        #         self.Wh[byStateIdx, stateIdx] \
        #             = np.random.uniform(-np.sqrt(6 / (self.Nin * self.Nout)),
        #                                 +np.sqrt(6 / (self.Nin * self.Nout)),
        #                                 (self.N, self.N))
        #         self.Wh[byStateIdx, stateIdx, :, :self.Nin] = 0.
        #         self.Wh[byStateIdx, stateIdx, self.Nin + self.Nh:] = 0.
        #         np.fill_diagonal(self.Wh[byStateIdx, stateIdx], 0.)
        #
        # self.gradWh = np.zeros_like(self.Wh)
        # self.gradRh = np.zeros_like(self.Rh)

        # for i, lim0 in enumerate(self.inputLim):
        #     for j, lim1 in enumerate(self.hiddenLim):
        #         if i != j:
        #             self.C[lim0[0]: lim0[1], lim1[0]: lim1[1]] = 0.
        #             self.C[lim1[0]: lim1[1], lim0[0]: lim0[1]] = 0.
        #
        # for i, lim0 in enumerate(self.hiddenLim):
        #     for j, lim1 in enumerate(self.hiddenLim):
        #         if i < j:
        #             # self.C[lim0[0]: lim0[1], lim1[0]: lim1[1]] = 0.
        #             self.C[lim1[0]: lim1[1], lim0[0]: lim0[1]] = 0.
        #
        # for i, lim0 in enumerate(self.hiddenLim):
        #     for j, lim1 in enumerate(self.outputLim):
        #         if i != j:
        #             self.C[lim0[0]: lim0[1], lim1[0]: lim1[1]] = 0.
        #             self.C[lim1[0]: lim1[1], lim0[0]: lim0[1]] = 0.
        #
        # self.C[:self.Nin, self.Nin + self.Nh:] = 0.
        # self.W *= self.C
        # self.zippedLims = zip(self.inputLim, self.hiddenLim, self.outputLim)

        # self.halfPortionInputIndex = self.Nin // 2
        # self.halfPortionHiddenIndex = self.Nh // 2
        #
        # self.W[:self.halfPortionInputIndex,
        # self.Nin + self.halfPortionHiddenIndex: self.Nin + self.Nh] = 0.
        #
        # self.W[self.halfPortionInputIndex: self.Nin,
        # self.Nin: self.Nin + self.halfPortionHiddenIndex] = 0.
        #
        # # self.W[self.Nin: self.Nin + self.halfPortionHiddenIndex,
        # # self.Nin + self.halfPortionHiddenIndex: self.Nin + self.Nh] = 0.
        #
        # self.W[self.Nin + self.halfPortionHiddenIndex: self.Nin + self.Nh,
        # self.Nin: self.Nin + self.halfPortionHiddenIndex] = 0.

    def get_params_usage(self):
        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0])

        self.numberOfUsableParams = self.N ** 2 \
                                    - self.N * self.Nin \
                                    - self.Nout * (self.N - self.Nin)

        print(f'+\tParams Usage: {(self.numberOfTrainableParams / self.numberOfUsableParams) * 100:.2f}%')
        print(f'+\tNumber of Trainable Parameters: {self.numberOfTrainableParams}')
        print(f'+\tNumber of Usable Parameters: {self.numberOfUsableParams}')

    def getActivationFunction(self):
        if self.activationFunc == 'sigmoid':
            self.fPrime = sigmoidPrime
            self.f = sigmoid

        elif self.activationFunc == 'bwf':
            self.fPrime = butterworthPrime
            self.f = butterworth

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

        elif self.lossFunc == 'bce':
            self.gPrime = crossEntropyPrime
            self.g = crossEntropy

        elif self.lossFunc == 'mae':
            self.gPrime = maePrime
            self.g = mae

    def zeros_io(self):
        self.I *= 0.
        self.O *= 0.

        self.Ih *= 0.
        self.Oh *= 0.

    def zeros_grad(self):
        self.gradW *= 0.
        # print(len(self.gradWh.keys()))
        for key in self.gradWh.keys():
            self.gradWh[key] *= 0.

        self.gradD *= 0.
        self.gradB *= 0.
        self.gradP *= 0.
        self.gradR *= 0.
        self.lossWRTHiddenOutput *= 0.
        self.lossWRTInputOutput *= 0.

    def backward(self, v, etaW, etaD, etaB, etaP, etaR,
                 lossWRTOutputOutput: np.ndarray = np.zeros((1,)),
                 a: np.ndarray = np.ones((1,)), alpha: float = 0.4):
        self.zeros_grad()

        if a.all() != 1.:
            self.lossWRTHiddenOutput += a

        for stateIdx in range(self.numberOfStates - 1, -1, -1):
            byStateIdc = np.where(self.Ch[:, stateIdx] == 1)[0]
            target = v[stateIdx * self.Nout: (stateIdx + 1) * self.Nout]

            calculateGradients(
                target, self.W, self.D, self.C, self.B, self.P,
                self.R, self.I, self.Ih, self.O, self.Oh, self.gradW,
                self.gradD, self.gradR, self.gradP, self.gradB,
                self.Nin, self.Nh, self.Nout, self.inputIdc,
                self.hiddenIdc, self.outputIdc, self.f, self.fPrime, self.gPrime,
                self.lossWRTInputOutput, self.lossWRTHiddenOutput[stateIdx],
                lossWRTOutputOutput, alpha, self.locks, self.enableInputActivation,
                self.enableLoss, stateIdx
            )
            # print(len(byStateIdc))
            for byStateIdx in byStateIdc:
                calculateGradientBetweenStates(
                    self.gradWh[f'{byStateIdx}{stateIdx}'],
                    self.gradR, self.gradB, self.W, self.D,
                    self.B, self.Wh[f'{byStateIdx}{stateIdx}'],
                    self.Ih, self.Oh, self.R, self.nodeIdc,
                    self.N, self.Nin, self.Nh, stateIdx, byStateIdx,
                    self.lossWRTHiddenOutput, self.fPrime
                )

        # etaW += self.wE * self.gradW
        # etaD += self.dE * self.gradD
        # etaB += self.bE * self.gradB
        # etaP += self.pE * self.gradP
        # etaR += self.rE * self.gradR

        self.W += -etaW * self.gradW
        self.W = np.clip(self.W, -1., 1.)
        # print(len(self.Wh.keys()))
        for key in self.Wh.keys():
            self.Wh[key] += -etaW * self.gradWh[key]
            self.Wh[key] = np.clip(self.Wh[key], -1., 1.)

        # self.Rh += -etaR * self.gradRh
        # self.Rh = np.clip(self.Rh, self.minRadius, self.maxRadius)

        self.B += -etaB * self.gradB
        self.B = np.clip(self.B, self.minBias, self.maxBias)

        self.D += -etaD * self.gradD
        # self.D[self.D <= 0.] = self.minDivision
        self.D = np.clip(self.D, self.minDivision, self.maxDivision)

        self.P += -etaP * self.gradP

        self.R += -etaR * self.gradR
        # self.R[self.R <= 0.] = (self.maxRadius + self.minRadius) / 2
        self.R = np.clip(self.R, self.minRadius, self.maxRadius)

        # Applying decay factor lambda for previous experiences.
        # i.e ((m0 * lambda + m1) * lambda + m2) * lambda + ... + mk) * lambda
        # self.wE *= np.exp(-decayFactor * t)
        # self.dE *= np.exp(-decayFactor * t)
        # self.bE *= np.exp(-decayFactor * t)
        # self.pE *= np.exp(-decayFactor * t)
        # self.rE *= np.exp(-decayFactor * t)

        # Adding new experiences
        # self.wE += self.gradW
        # self.dE += self.gradD
        # self.bE += self.gradB
        # self.pE += self.gradP
        # self.rE += self.gradR

    def predict(self, us, a: np.ndarray = np.ones((1,))):
        self.zeros_io()
        if a.all() != 1.:
            self.I[self.Nin: self.Nin + self.Nh] += a

        for stateIdx in range(self.numberOfStates):
            if stateIdx < self.numberOfStates:
                u = us[0][stateIdx * self.Nin: (stateIdx + 1) * self.Nin]
            else:
                u = us[0][self.Nin + self.Nh:]

            forwardInput(self.D, self.Ih[stateIdx], self.Oh[stateIdx], self.B,
                         self.inputIdc, u, self.f, self.enableInputActivation)

            byStateIdc = np.where(self.Ch[:, stateIdx] == 1)[0]
            # print(len(byStateIdc))
            for byStateIdx in byStateIdc:
                recallMemory(stateIdx, byStateIdx, self.Ih, self.Oh,
                             self.Wh[f'{byStateIdx}{stateIdx}'], self.R, self.Nin, self.Nh, self.Nout,
                             self.hiddenIdc, self.outputIdc, self.stateFlags,
                             self.numberOfStates)

            forwardHidden(self.Nh, self.W, self.C, self.D, self.B, self.P, self.R,
                          self.Ih[stateIdx], self.Oh[stateIdx], self.hiddenIdc,
                          self.f)

            forwardOutput(self.Nout, self.outputIdc, self.W, self.C, self.D,
                          self.B, self.P, self.R, self.Ih[stateIdx],
                          self.Oh[stateIdx], self.f)

        return np.copy(self.Oh[:, self.Nin + self.Nh:])

    def save_weight_image_per_epoch(self, epoch, simulationFolderPath):
        if not os.path.exists(f'{simulationFolderPath}/graphs'):
            os.mkdir(f'{simulationFolderPath}/graphs')
            os.mkdir(f'{simulationFolderPath}/graphs/W')

        if (epoch + 1) % self.frequency == 0:
            WForPlot = self.W.copy()
            WForPlot = np.abs(WForPlot)

            plt.imsave(f'./{simulationFolderPath}/graphs/W/w{epoch}.jpg',
                       WForPlot, cmap='hot')

    def record_output_of_hidden_neurons(self, simulationFolderPath, label):
        if not os.path.exists(f'{simulationFolderPath}/graphs/O'):
            os.mkdir(f'{simulationFolderPath}/graphs/O')

        O = self.O.copy()
        O = O[self.Nin: self.Nin + self.Nh] \
            .reshape(self.Nh // 2, 2).T
        plt.imsave(f'./{simulationFolderPath}/graphs/O/{label}.jpg', O)
