import json
import os
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import random
import time
import cv2

from time import sleep

from numba import cuda

from lib4WithCuda import forward, calculateGradients, magnitude, \
    g, initialize
from tqdm import tqdm


class Network:
    def __init__(self,
                 train: bool = True, bias: float = 1, decay: float = 0.1,
                 etaW: float = 1e-3, etaR: float = 1e-3, etaP: float = 1e-3,
                 minRadius: int = 10, maxRadius: int = 20, frequency: int = 5,
                 nInputs: int = 784, nHiddens: int = 10, nOutputs: int = 10,  # NOQA
                 trainingDatasetX: list = None, trainingDatasetY: list = None,
                 evaluatingDatasetX: list = None, evaluatingDatasetY: list = None,
                 epochs: int = 10, datasetCap: int = 1000, batchSize: int = 32,
                 width: int = 20, height: int = 20, depth: int = 20,
                 hiddenZoneOffset: int = 400, outputZoneOffset: int = 400,
                 maxInputPerNode: int = 2, minInputPerNode: int = 1,
                 maxOutputPerNode: int = 20, minOutputPerNode: int = 1,
                 save: bool = True, datasetName: str = 'MNIST', configFilePath: str = '',
                 activationFunc: str = 'sigmoid', lossFunc: str = 'mse'):  # NOQA
        # assert (rOffset < minRadius) and (minRadius < maxRadius), 'Radius\' Offset must be less than Minimum Radius'

        self.lossPerEpoch = []

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

        self.bias = np.array([bias], dtype=np.float64)
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

        self.trainingDatasetX = trainingDatasetX
        self.trainingDatasetY = trainingDatasetY

        self.evaluatingDatasetX = evaluatingDatasetX
        self.evaluatingDatasetY = evaluatingDatasetY

        # self.batchSize = batchSize

        self.lossWRTHiddenOutput = np.zeros((self.N,), dtype=np.float64)
        self.W = np.zeros((self.N, self.N),
                          dtype=np.float64)  # Use to store size of every node in the network
        self.P = np.zeros((self.N, 3),
                          dtype=np.float64)  # Use to store position (x, y) of node in the network

        self.R = np.zeros((self.N, 1),
                          dtype=np.float64)  # Use to store size of every node in the network
        self.I = np.zeros((self.N,), dtype=np.float64)
        self.O = np.zeros((self.N,), dtype=np.float64)

        self.gradU = np.zeros((self.N, self.N),
                              dtype=np.float64)  # Gradient U use to update W
        self.gradR = np.zeros((self.N, 1),
                              dtype=np.float64)  # Gradient S use to update the size of each node
        self.gradP = np.zeros((self.N, 3),
                              dtype=np.float64)  # Gradient P use to update position of each node

        self.inputIdc = np.array([i for i in range(nInputs)], dtype=np.int64).reshape((self.nInputs, 1))
        self.hiddenIdc = np.array([i + nInputs for i in range(nHiddens)], dtype=np.int64).reshape(
            (self.nHiddens, 1))  # NOQA
        self.outputIdc = np.array([i + nInputs + nHiddens for i in range(nOutputs)], dtype=np.int64).reshape(
            (self.nOutputs, 1))
        self.nodeIdc = np.concatenate([self.inputIdc, self.hiddenIdc, self.outputIdc], axis=0)

        self.numberOfTrainableParams = 0
        self.numberOfUsableParams = 0

        if train:
            self.simulationIdx = 0
            while True:
                try:
                    self.filename = f'3d_nodes_simulation_{self.simulationIdx}'
                    self.dirName = f'records/{self.filename}'

                    os.mkdir(f'./{self.dirName}')
                    break

                except FileExistsError:
                    self.simulationIdx += 1
            print(f'Using file\'s name: {self.filename}')

            self.save = save
            self.configFilePath = configFilePath

            self.construct_network()

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

    def save_config(self, trainingTime, accuracy):
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
            'bias': int(self.bias),
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
            'losses': self.lossPerEpoch,
            'accuracy': accuracy
        }

        with open(f'./{self.dirName}/configs.json', 'w') as f:
            json.dump(configs, f, indent=4)

    def save_result(self):
        with open(f'./{self.dirName}/w.npy', 'wb') as fW:
            np.save(fW, self.W)

        with open(f'./{self.dirName}/r.npy', 'wb') as fR:
            np.save(fR, self.R)

        with open(f'./{self.dirName}/p.npy', 'wb') as fP:
            np.save(fP, self.P)

    def load_config(self):
        assert os.path.exists(f'./{self.configFilePath}'), f'./{self.configFilePath} cannot be found'
        with open(f'./{self.configFilePath}', 'r') as f:
            for key, value in json.load(f).items():
                setattr(self, key, value)

    def load_simulation(self, fp):
        print(f'Load Simulation From {fp}')
        assert os.path.exists(f'./{fp}'), f'./{fp} cannot be found'

        print('Load Matrices')
        with open(f'./{fp}/w.npy', 'rb') as fW:
            self.W = np.load(fW)

        with open(f'./{fp}/r.npy', 'rb') as fR:
            self.R = np.load(fR)

        with open(f'./{fp}/p.npy', 'rb') as fP:
            self.P = np.load(fP)

        self.configFilePath = f'./{fp}/configs.json'
        self.load_config()

    def isInHiddenZone(self, vec):
        if (self.hiddenZoneOffset < vec[0] < self.width - self.hiddenZoneOffset) and \
                (self.hiddenZoneOffset < vec[1] < self.height - self.hiddenZoneOffset) and \
                (self.hiddenZoneOffset < vec[2] < self.depth - self.hiddenZoneOffset):
            return True
        return False

    def isInOutputZone(self, vec):
        _outputZoneOffset = self.hiddenZoneOffset + self.outputZoneOffset
        if (_outputZoneOffset < vec[0] < self.width - _outputZoneOffset) and \
                (_outputZoneOffset < vec[1] < self.height - _outputZoneOffset) and \
                (_outputZoneOffset < vec[2] < self.depth - _outputZoneOffset):
            return True
        return False

    def isOverlap(self, idx, v, zone: str = 'i'):
        assert self.outputZoneOffset > 0 and self.hiddenZoneOffset > 0

        if zone == 'i':
            zoneNodeIdc = self.inputIdc[0]
        elif zone == 'h':
            zoneNodeIdc = self.hiddenIdc[0]
        else:
            zoneNodeIdc = self.outputIdc[0]

        for nodeIdx in zoneNodeIdc:
            # print(v.shape, self.P[nodeIdx].shape)
            if idx != nodeIdx and magnitude(v, self.P[nodeIdx]) <= 0.8:
                return True
        return False

    def drawVecSamples(self, zone: str = 'h'):
        start, stopWidth, stopHeight, stopDepth = 0, self.width, self.height, self.depth

        if zone == 'h':
            start = self.hiddenZoneOffset
            stopWidth = self.width - start
            stopHeight = self.height - start
            stopDepth = self.depth - start

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
            # self.R[nodeIdx, 0] = 1
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='i')

        print(f'Initialize Hidden Nodes')
        for nodeIdx in self.hiddenIdc:
            self.R[nodeIdx, 0] = np.random.randint(self.minRadius, self.maxRadius, 1)[0]
            # self.R[nodeIdx, 0] = 1
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='h')

        print(f'Initialize Output Nodes')
        for nodeIdx in self.outputIdc:
            # self.R[nodeIdx, 0] = np.random.randint(self.minRadius, self.maxRadius, 1)[0]
            self.R[nodeIdx, 0] = 1
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='o')

        print('Initialize Weighted Matrix')
        self.W, _ = initialize(np.random.randint(self.minInputPerNode, self.maxInputPerNode, 1)[0],
                               np.random.randint(self.minOutputPerNode, self.maxOutputPerNode, 1)[0],
                               self.N, self.nInputs, self.nHiddens, self.nOutputs)

        # Find cycles in Hidden Layer
        loops = []
        for h0 in self.hiddenIdc:
            closed = []
            queue = [h0]

            while len(queue) > 0:
                n0 = queue.pop(0)
                connectedNodes = np.where(self.W[n0, :] == 1)[0]

                for node in connectedNodes:
                    if node != h0 and node not in closed:
                        queue.append(node)
                closed.append(n0)

        # Remove cycles in Hidden Layer
        if len(loops) > 0:
            for loop in loops:
                if len(loop) > 1:
                    if np.random.randint(0, 1) <= 0.5:
                        self.W[loop[-2], loop[-1]] = 0
                    else:
                        self.W[loop[-1], loop[-2]] = 0

        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0])
        self.numberOfUsableParams = np.square(self.N) \
                                    - self.N \
                                    - self.nInputs * self.N \
                                    - self.nOutputs * (self.N - self.nInputs)
        print(f'Params Usage: {(self.numberOfTrainableParams / self.numberOfUsableParams) * 100:.2f}%')
        print(f'Number of Trainable Parameters: {self.numberOfTrainableParams}')
        print(f'Number of Usable Parameters: {self.numberOfUsableParams}')

    # def getActivationFunction(self):
    #     if self.activationFunc == 'sigmoid':
    #         self.fPrime = sigmoidPrime
    #         self.f = sigmoid
    #
    #     elif self.activationFunc == 'relu':
    #         self.fPrime = reluPrime
    #         self.f = relu
    #
    #     elif self.activationFunc == 'absolute_sin':
    #         self.fPrime = sin
    #         self.f = cos
    #
    #     elif self.activationFunc == 'tanh':
    #         self.fPrime = tanh
    #         self.f = tanhPrime

    def train(self) -> None:
        # self.getActivationFunction()
        sleep(1)

        for epoch in range(self.epochs):
            etaW = np.copy(self.etaW)
            etaP = np.copy(self.etaP)
            etaR = np.copy(self.etaR)
            loss = 0

            if (epoch + 1) % self.frequency == 0:
                etaW *= self.decay
                etaP *= self.decay
                etaR *= self.decay

            for u, v in tqdm(zip(self.trainingDatasetX, self.trainingDatasetY),
                             desc=f'Number of trainable parameters: {self.numberOfTrainableParams:<5}, '
                                  f'Number of usable parameters: {self.numberOfUsableParams:<5}, '
                                  f'Epoch: {epoch + 1:<2} / {self.epochs:<2}',
                             colour='green',
                             leave=False,
                             total=len(self.trainingDatasetX)):
                self.I.fill(0)
                self.O.fill(0)

                I = cuda.to_device(self.I)
                O = cuda.to_device(self.O)

                W = cuda.to_device(self.W)
                P = cuda.to_device(self.P)
                R = cuda.to_device(self.R)
                bias = cuda.to_device(self.bias)

                self.I, self.O = forward(W, I, O, P, R,
                                         self.outputIdc, self.hiddenIdc,
                                         u, bias)
                predict = np.ascontiguousarray(self.O[min(self.outputIdc[:, 0]):])
                predict = cuda.to_device(predict)

                assert not np.isnan(predict).any()

                I = cuda.to_device(self.I)
                O = cuda.to_device(self.O)

                self.gradU.fill(0)
                self.gradP.fill(0)
                self.gradR.fill(0)
                self.lossWRTHiddenOutput.fill(0)

                gradU = cuda.to_device(self.gradU)
                gradP = cuda.to_device(self.gradP)
                gradR = cuda.to_device(self.gradR)
                lossWRTHiddenOutput = cuda.to_device(self.lossWRTHiddenOutput)

                self.gradU, self.gradR, self.gradP = calculateGradients(predict, v,
                                                                        W, I, O, P, R,
                                                                        gradU, gradR, gradP,
                                                                        self.nInputs, self.nHiddens, self.nOutputs,
                                                                        self.hiddenIdc, self.outputIdc,
                                                                        lossWRTHiddenOutput)

                assert not np.isnan(self.gradU).any()
                assert not np.isnan(self.gradP).any()
                assert not np.isnan(self.gradR).any()

                self.W += -etaW * self.gradU
                self.P += -etaP * self.gradP
                self.R += -etaR * self.gradR

                loss += g(predict.copy_to_host(), v)

            self.lossPerEpoch.append(loss)

    def evaluate(self) -> float:
        # self.getActivationFunction()
        score = 0

        for u, v in tqdm(zip(self.evaluatingDatasetX, self.evaluatingDatasetY),
                         desc=f'Number of trainable parameters: {self.numberOfTrainableParams:<5}, '
                              f'Number of usable parameters: {self.numberOfUsableParams:<5}',
                         colour='green',
                         leave=False):
            # us = self.evaluatingDatasetX[self.batchSize * batchIdx: self.batchSize * (batchIdx + 1)]
            # us = np.array(us).reshape(self.batchSize, self.nInputs)
            #
            # vs = self.evaluatingDatasetY[self.batchSize * batchIdx: self.batchSize * (batchIdx + 1)]
            # vs = np.array(vs).reshape(self.batchSize, self.nOutputs)
            #
            # us = cuda.to_device(us)
            # vs = cuda.to_device(vs)

            score += len(np.where((np.argmax(self.predict(u), axis=1)
                                   == np.argmax(v, axis=1)) == True)[0])

        return float(f'{(score / len(self.evaluatingDatasetX)) * 100:.2f}')

    def predict(self, u):
        # self.getActivationFunction()
        I = cuda.to_device(self.I.fill(0))
        O = cuda.to_device(self.O.fill(0))

        W = cuda.to_device(self.W)
        P = cuda.to_device(self.P)
        R = cuda.to_device(self.R)
        bias = cuda.to_device(self.bias)

        _, O = forward(W, I, O, P, R,
                       self.outputIdc, self.hiddenIdc,
                       u, bias)
        # print(min(self.outputIdc[:, 0]))
        predicts = O[min(self.outputIdc[:, 0]):]
        # print(O.shape)

        return predicts


if __name__ == '__main__':
    nInputs = 28 * 28
    nHiddens = 300  # NOQA
    nOutputs = 10
    N = nInputs + nHiddens + nOutputs

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    train = True
    # train = False
    datasetCap = 256 * 20
    batchSize = 256
    ratio = 1 / 2

    classPerTrainingDataset = np.zeros((nOutputs,), dtype=np.int64)
    classPerTransformedTrainingDataset = np.zeros((nOutputs,), dtype=np.int64)

    evaluatingDatasetX = []
    evaluatingDatasetY = []

    trainingDatasetX = []
    trainingDatasetY = []

    validationSetX = []
    validationSetY = []

    if train:
        # trainingMNIST = torchvision.datasets.EMNIST('./data', train=True, download=True, split='balanced')
        # evaluatingMNIST = torchvision.datasets.EMNIST('./data', train=False, split='balanced')

        trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=True)
        transformedTrainingMNIST = torchvision.datasets.MNIST('./data', train=True,
                                                              transform=torchvision.transforms.Compose([
                                                                  torchvision.transforms.RandomResizedCrop(
                                                                      (28, 28), scale=(0.5, 0.5)
                                                                  ),
                                                                  torchvision.transforms.RandomRotation(20)
                                                              ]))
        evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)

        # trainingMNIST = torchvision.datasets.QMNIST('./data', what='train', download=True)
        # evaluatingMNIST = torchvision.datasets.QMNIST('./data', what='nist', download=True)

        # Preprocessing Training Data
        for u, v in tqdm(trainingMNIST, desc='Preprocessing Training Data: ', colour='green'):
            if classPerTrainingDataset[v] + 1 \
                    <= (datasetCap * ratio) // nOutputs:
                # Target output vector v
                t = np.zeros((nOutputs,), np.float64)
                t[v] = 1

                if np.max(u) > 255. or np.max(u) <= 0.:
                    continue
                u = np.array(u) / np.max(u)

                trainingDatasetX.append(u.flatten())
                trainingDatasetY.append(t)

                classPerTrainingDataset[v] += 1
            else:
                t = np.zeros((nOutputs,), np.float64)
                t[v] = 1

                if np.max(u) > 255. or np.max(u) <= 0.:
                    continue

                u = np.array(u) / np.max(u)
                validationSetX.append(u.flatten())
                validationSetY.append(t)

        for u, v in tqdm(transformedTrainingMNIST, desc='Preprocessing Transformed Training Data: ', colour='green'):
            if classPerTransformedTrainingDataset[v] + 1 \
                    <= (datasetCap * (1 - ratio)) // nOutputs:
                # Target output vector v
                t = np.zeros((nOutputs,), np.float64)
                t[v] = 1

                if classPerTransformedTrainingDataset[v] + 1 == 10:
                    cv2.imwrite(f'./training_samples/{v}.jpg', np.array(u))

                if np.max(u) > 255. or np.max(u) <= 0.:
                    continue
                u = np.array(u) / np.max(u)

                trainingDatasetX.append((u).flatten())
                trainingDatasetY.append(t)

                classPerTransformedTrainingDataset[v] += 1

        # Preprocessing Evaluating Data
        for u, v in tqdm(evaluatingMNIST, desc='Preprocessing Evaluating Data: ', colour='green'):
            # Target output vector v
            t = np.zeros((nOutputs,), np.float64)
            t[v] = 1

            u = np.array(u) / np.max(u)

            evaluatingDatasetX.append(u.flatten())
            evaluatingDatasetY.append(t)

        # evaluatingDatasetX.extend(validationSetX)
        # evaluatingDatasetY.extend(validationSetY)

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        trainingDatasetX=trainingDatasetX, trainingDatasetY=trainingDatasetY,
        evaluatingDatasetX=evaluatingDatasetX, evaluatingDatasetY=evaluatingDatasetY,
        epochs=30, datasetCap=datasetCap, frequency=8,
        minRadius=500, maxRadius=1000, save=True, batchSize=batchSize,
        etaW=8e-2, etaP=1e-2, etaR=1e-2, decay=0.1,
        width=8000, height=8000, depth=8000, bias=-0.5,
        hiddenZoneOffset=3000, outputZoneOffset=500,
        maxInputPerNode=maxInputPerNode, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode, minOutputPerNode=maxOutputPerNode // 2,
        activationFunc='sigmoid', datasetName='MNIST', train=train
    )

    if train:
        cuda.synchronize()
        startTime = time.perf_counter()
        network.train()
        accuracy = network.evaluate()
        print(f'Accuracy: {accuracy}%')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')

        network.save_result()
        network.save_config(trainingTime=totalTime,
                            accuracy=accuracy)
        network.plot()
    else:
        network.load_simulation('records/3d_nodes_simulation_49')

    fp = './test1.jpg'
    u = cv2.imread(fp)
    u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

    stream = cuda.stream()
    u = np.array(u) / np.max(u)
    u = (u.flatten()).reshape(1, nInputs)
    u = cuda.to_device(u)

    print(f'Testing {fp}')
    predict = network.predict(u, stream)
    print(predict, np.sum(predict))
    print(f'Prediction: {np.argmax(predict)}')
