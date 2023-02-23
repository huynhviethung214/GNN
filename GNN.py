import json
import os
import torch

import numpy as np
import torchvision
import matplotlib.pyplot as plt
import random
import time
import cv2

from time import sleep

import torchvision.transforms

from modules import forward, calculateGradients, magnitude, \
    g, initialize, sigmoidPrime, \
    reluPrime, sigmoid, relu
from tqdm import tqdm


class Network:
    def __init__(self,
                 train: bool = True, bias: float = 1, decay: float = 0.1,
                 etaW: float = 1e-3, etaR: float = 1e-3, etaP: float = 1e-3,
                 minRadius: int = 10, maxRadius: int = 20, frequency: int = 5,
                 nInputs: int = 784, nHiddens: int = 10, nOutputs: int = 10,  # NOQA
                 trainingDataset: list = None, evaluatingDataset: list = None, epochs: int = 10,
                 datasetCap: int = 1000, width: int = 20, height: int = 20, depth: int = 20,
                 hiddenZoneOffset: int = 400, outputZoneOffset: int = 400,
                 maxInputPerNode: int = 2, minInputPerNode: int = 1,
                 maxOutputPerNode: int = 20, minOutputPerNode: int = 1,
                 save: bool = True, datasetName: str = 'MNIST', configFilePath: str = '',
                 activationFunc: str = 'sigmoid', lossFunc: str = 'cross_entropy'):  # NOQA
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
        self.trainingDataset = trainingDataset
        self.evaluatingDataset = evaluatingDataset

        self.confusionMatrix = np.zeros((self.nOutputs, self.nOutputs), dtype=np.float64)

        self.lossWRTHiddenOutput = np.zeros((self.N,), dtype=np.float64)
        self.W = np.zeros((self.N, self.N),
                          dtype=np.float64)  # Use to store size of every node in the network
        self.P = np.zeros((self.N, 3),
                          dtype=np.float64)  # Use to store position (x, y) of node in the network

        self.R = np.zeros((self.N, 1),
                          dtype=np.float64)  # Use to store size of every node in the network
        self.I = np.zeros((self.N, self.N), dtype=np.float64)
        self.O = np.zeros((self.N, self.N), dtype=np.float64)

        self.gradU = np.zeros((self.N, self.N),
                              dtype=np.float64)  # Gradient U use to update W
        self.gradR = np.zeros((self.N, 1),
                              dtype=np.float64)  # Gradient S use to update the size of each node
        self.gradP = np.zeros((self.N, 3),
                              dtype=np.float64)  # Gradient P use to update position of each node

        self.inputIdc = np.array([i for i in range(nInputs)], dtype=np.int64).reshape((self.nInputs,))
        self.hiddenIdc = np.array([i + nInputs for i in range(nHiddens)], dtype=np.int64).reshape(
            (self.nHiddens,))  # NOQA
        self.outputIdc = np.array([i + nInputs + nHiddens for i in range(nOutputs)], dtype=np.int64).reshape(
            (self.nOutputs,))
        self.nodeIdc = np.concatenate([self.inputIdc, self.hiddenIdc, self.outputIdc], axis=0)

        self.numberOfTrainableParams = 0
        self.numberOfUsableParams = 0

        if train:
            self.simulationIdx = 0
            while True:
                try:
                    self.filename = f'3d_nodes_simulation_{self.simulationIdx}'
                    self.dirName = f'records/{self.filename}'
                    self.wFolders = f'{self.dirName}/w_graphs'

                    os.mkdir(f'./{self.dirName}')
                    os.mkdir(f'./{self.wFolders}')
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

        with open(f'./{self.dirName}/confusion_matrix.npy', 'wb') as fCM:
            np.save(fCM, self.confusionMatrix)

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

        with open(f'./{fp}/confusion_matrix.npy', 'rb') as fCM:
            self.confusionMatrix = np.load(fCM)

        self.configFilePath = f'./{fp}/configs.json'
        self.load_config()

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

        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0])
        self.numberOfUsableParams = np.square(self.N) \
                                    - self.N \
                                    - self.nInputs * self.N \
                                    - self.nOutputs * (self.N - self.nInputs)
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

    def getPrecisionForEachClass(self):
        precisionVector = np.zeros((self.nOutputs, 1))
        for outputIdx in range(self.outputIdc.shape[0]):
            precisionVector[outputIdx] = self.confusionMatrix[outputIdx, outputIdx] \
                                         / np.sum(self.confusionMatrix[:, outputIdx])

        return precisionVector * 100.

    def train(self) -> None:
        self.getActivationFunction()
        sleep(1)

        for epoch in range(self.epochs):
            etaW = np.copy(self.etaW)
            etaP = np.copy(self.etaP)
            etaR = np.copy(self.etaR)
            loss = 0

            WForPlot = self.W.copy()
            WForPlot = np.abs(WForPlot)
            plt.imsave(f'./{self.wFolders}/w{epoch}.jpg',
                       WForPlot[:, nInputs: nInputs + nHiddens].T
                       , cmap='hot')

            if (epoch + 1) % self.frequency == 0:
                etaW *= self.decay
                etaP *= self.decay
                etaR *= self.decay

            for u, v in tqdm(self.trainingDataset[:self.datasetCap],
                             desc=f'Number of trainable parameters: {self.numberOfTrainableParams:<5}, '
                                  f'Number of usable parameters: {self.numberOfUsableParams:<5}, '
                                  f'Epoch: {epoch + 1:<2} / {self.epochs:<2}',
                             colour='green',
                             leave=False,
                             total=self.datasetCap):
                self.I.fill(0)
                self.O.fill(0)

                self.I, self.O = forward(self.W, self.I, self.O, self.P, self.R,
                                         self.nodeIdc, self.inputIdc,
                                         self.hiddenIdc, self.outputIdc,
                                         u, self.bias, self.f)
                predict = np.diag(self.O)[min(self.outputIdc):].reshape((self.nOutputs,))

                self.gradU.fill(0)
                self.gradR.fill(0)
                self.gradP.fill(0)
                self.lossWRTHiddenOutput.fill(0)

                self.gradU, self.gradR, self.gradP = calculateGradients(predict, v, self.W, self.I,
                                                                        self.O, self.P, self.R,
                                                                        self.gradU, self.gradR, self.gradP,
                                                                        self.nInputs, self.nHiddens, self.nOutputs,
                                                                        self.hiddenIdc, self.outputIdc,
                                                                        self.fPrime, self.lossWRTHiddenOutput)
                self.W += -etaW * self.gradU
                self.P += -etaP * self.gradP
                self.R += -etaR * self.gradR
                loss += g(predict, v, self.nOutputs)

            self.lossPerEpoch.append(loss / self.datasetCap)

    def evaluate(self):
        self.getActivationFunction()

        for u, v in tqdm(self.evaluatingDataset,
                         desc=f'Number of trainable parameters: {self.numberOfTrainableParams:<5}, '
                              f'Number of usable parameters: {self.numberOfUsableParams:<5}',
                         colour='green',
                         leave=False):
            predict = np.argmax(self.predict(u))
            target = np.argmax(v)

            self.confusionMatrix[target, predict] += 1

        return self.confusionMatrix

    def predict(self, u):
        self.getActivationFunction()
        self.I.fill(0)
        self.O.fill(0)

        _, O = forward(self.W, self.I, self.O, self.P, self.R,
                       self.nodeIdc, self.inputIdc,
                       self.hiddenIdc, self.outputIdc,
                       u, self.bias, self.f)
        predict = np.diag(self.O)[min(self.outputIdc):].reshape((self.nOutputs,))

        return predict


if __name__ == '__main__':
    nInputs = 28 * 28
    nHiddens = 500  # NOQA
    nOutputs = 10
    N = nInputs + nHiddens + nOutputs

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    train = True
    EPOCHS = 24
    datasetCap = 8000
    ratio = 1 / 2
    record_no = 63

    classPerTrainingDataset = np.zeros((nOutputs,), dtype=np.int64)
    classPerTransformedTrainingDataset = np.zeros((nOutputs,), dtype=np.int64)
    evaluatingDataset = []
    trainingDataset = []
    validationSet = []

    if train:
        trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=True)
        evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)

        # Preprocessing Training Data
        for u, v in tqdm(trainingMNIST, desc='Preprocessing Training Data: ', colour='green'):
            if classPerTrainingDataset[v] + 1 \
                    <= datasetCap // nOutputs:
                # Target output vector v
                t = np.zeros((nOutputs,), dtype=np.float64)
                t[v] = 1

                if np.max(u) == 0.:
                    continue

                if classPerTransformedTrainingDataset[v] + 1 == 10:
                    cv2.imwrite(f'./training_samples/{v}.jpg', np.array(u))

                u = np.array(u) / np.max(u)
                u[u != 0] = 1.

                trainingDataset.append([(u).flatten(), t])
                classPerTrainingDataset[v] += 1

        # Preprocessing Evaluating Data
        for u, v in tqdm(evaluatingMNIST, desc='Preprocessing Evaluating Data: ', colour='green'):
            # Target output vector v
            t = np.zeros((nOutputs,), dtype=np.float64)
            t[v] = 1

            u = np.array(u) / np.max(u)
            u[u != 0] = 1.

            evaluatingDataset.append([(u).flatten(), t])

    for _ in range(100):
        random.shuffle(trainingDataset)
        random.shuffle(evaluatingDataset)

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        trainingDataset=trainingDataset, evaluatingDataset=evaluatingDataset,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=6,
        minRadius=500, maxRadius=800, save=True,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=0.1,
        width=8000, height=8000, depth=8000, bias=-0.5,
        hiddenZoneOffset=3000, outputZoneOffset=500,
        maxInputPerNode=maxInputPerNode, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode, minOutputPerNode=maxOutputPerNode // 2,
        activationFunc='sigmoid', datasetName='MNIST', train=train
    )

    if train:
        startTime = time.perf_counter()
        network.train()
        confusionMatrix = network.evaluate()
        sumOfDiag = np.sum(np.diag(confusionMatrix))
        accuracy = sumOfDiag * 100. / len(evaluatingDataset)
        print(f'Accuracy: {accuracy}%')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')
        print(f'Precision:\n{network.getPrecisionForEachClass()}')

        network.save_result()
        network.save_config(trainingTime=totalTime,
                            accuracy=accuracy)
        network.plot()
    else:
        network.load_simulation(f'records/3d_nodes_simulation_{record_no}')

        fp = './test1.jpg'
        u = cv2.imread(fp)
        u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

        u = np.array(u) / np.max(u)
        u = u.flatten()

        print(f'Testing {fp}')
        predict = network.predict(u)
        print(f'Prediction: {np.argmax(predict)}')
