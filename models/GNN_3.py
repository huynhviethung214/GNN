import os
import time
import json
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from modules.modules_3 import forward, calculateGradients, magnitude, initialize
from tqdm import tqdm


PI = np.round(np.pi, 4)


class Network:
    def __init__(self,
                 train: bool = True,
                 decay: float = 0.1, decayFrequency: int = 5,
                 etaW: float = 1e-3, etaR: float = 1e-3, etaP: float = 1e-3,
                 etaA: float = 1e-3, etaL: float = 1e-3, etaF: float = 1e-3,
                 etaPHI: float = 1e-3,
                 minA: float = 0.3, maxA: float = 0.4,
                 minL: float = 2, maxL: float = 4,
                 minF: float = 200, maxF: float = 400,
                 minPHI: float = -np.pi / 2, maxPHI: float = np.pi / 2,
                 minRadius: int = 10, maxRadius: int = 20,
                 nInputs: int = 784, nHiddens: int = 10, nOutputs: int = 10,  # NOQA
                 epochs: int = 10, datasetCap: int = 1000,
                 width: int = 20, height: int = 20, depth: int = 20,
                 hiddenZoneOffset: int = 400, outputZoneOffset: int = 400,
                 maxInputPerNode: int = 2, minInputPerNode: int = 1,
                 maxOutputPerNode: int = 20, minOutputPerNode: int = 1,
                 datasetName: str = 'mnist', lossFunc: str = 'mse'):  # NOQA
        if train:
            self.fPrime = None
            self.f = None

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

            self.etaA = etaA
            self.etaL = etaL
            self.etaF = etaF
            self.etaPHI = etaPHI

            self.width = width
            self.depth = depth
            self.height = height

            self.minRadius = minRadius
            self.maxRadius = maxRadius

            self.minA = minA
            self.maxA = maxA

            self.minF = minF
            self.maxF = maxF

            self.minL = minL
            self.maxL = maxL

            self.minPHI = minPHI
            self.maxPHI = maxPHI

            self.epochs = epochs
            self.decay = decay
            self.decayFrequency = decayFrequency

            self.datasetCap = datasetCap
            self.datasetName = datasetName

            self.numberOfTrainableParams = 0
            self.numberOfUsableParams = 0

            self.createMatrices()
            self.construct_network()

    def createMatrices(self):
        self.W = np.zeros((self.N, self.N), dtype=np.float64)
        self.P = np.zeros((self.N, 3), dtype=np.float64)
        self.R = np.zeros((self.N,), dtype=np.float64)

        self.A = np.random.uniform(self.minA, self.maxA, (self.N,))
        self.L = np.random.uniform(self.minL, self.maxL, (self.N,))
        self.F = np.random.uniform(self.minF, self.maxF, (self.N,))
        self.PHI = np.random.choice(np.arange(self.minPHI, self.maxPHI, PI / 8), (self.N,))

        self.I = np.zeros((self.N,), dtype=np.float64)
        self.O = np.zeros((self.N,), dtype=np.float64)

        self.gradW = np.zeros((self.N, self.N), dtype=np.float64)
        self.gradP = np.zeros((self.N, 3), dtype=np.float64)
        self.gradR = np.zeros((self.N,), dtype=np.float64)

        self.gradA = np.zeros((self.N,), dtype=np.float64)
        self.gradL = np.zeros((self.N,), dtype=np.float64)
        self.gradF = np.zeros((self.N,), dtype=np.float64)
        self.gradPHI = np.zeros((self.N,))
        self.lossWRTHiddenOutput = np.zeros((self.N,), dtype=np.float64)

        self.inputIdc = np.array([i for i in range(self.nInputs)], dtype=np.int64).reshape((self.nInputs,))
        self.hiddenIdc = np.array([i + self.nInputs for i in range(self.nHiddens)], dtype=np.int64).reshape(
            (self.nHiddens,))  # NOQA
        self.outputIdc = np.array([i + self.nInputs + self.nHiddens for i in range(self.nOutputs)],
                                  dtype=np.int64).reshape((self.nOutputs,))
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
            'decayFrequency': self.decayFrequency,
            'datasetCap': self.datasetCap,
            'dataset': self.datasetName,
            'trainingTime': trainingTime,
            'accuracy': accuracy
        }

        with open(f'./{simulationFolderPath}/configs.json', 'w') as f:
            json.dump(configs, f, indent=4)

    def save_result(self, simulationFolderPath):
        np.save(f'./{simulationFolderPath}/C.npy', self.C)
        np.save(f'./{simulationFolderPath}/A.npy', self.A)
        np.save(f'./{simulationFolderPath}/L.npy', self.L)
        np.save(f'./{simulationFolderPath}/F.npy', self.F)
        np.save(f'./{simulationFolderPath}/PHI.npy', self.PHI)

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
        self.C = np.load(f'./{fp}/C.npy')
        self.A = np.load(f'./{fp}/A.npy')
        self.L = np.load(f'./{fp}/L.npy')
        self.F = np.load(f'./{fp}/F.npy')
        self.PHI = np.load(f'./{fp}/PHI.npy')

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
            self.R[nodeIdx] = np.random.randint(self.minRadius, self.maxRadius, 1)[0]
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='i')

        print(f'Initialize Hidden Nodes')
        for nodeIdx in self.hiddenIdc:
            self.R[nodeIdx] = np.random.randint(self.minRadius, self.maxRadius, 1)[0]
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='h')

        print(f'Initialize Output Nodes')
        for nodeIdx in self.outputIdc:
            self.R[nodeIdx] = 1
            self.P[nodeIdx] = self.getNodePosition(nodeIdx, zone='o')

        print('Initialize Weighted Matrix')
        self.W, _ = initialize(np.random.randint(self.minInputPerNode, self.maxInputPerNode, 1)[0],
                               np.random.randint(self.minOutputPerNode, self.maxOutputPerNode, 1)[0],
                               self.N, self.nInputs, self.nOutputs,
                               self.inputIdc, self.hiddenIdc, self.outputIdc)

        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0]) \
                                       + self.R.shape[0] \
                                       + self.P.shape[0] \
                                       + self.A.shape[0] \
                                       + self.L.shape[0] \
                                       + self.F.shape[0] \
                                       + self.PHI.shape[0]

        self.numberOfUsableParams = np.square(self.N) \
                                    + self.R.shape[0] \
                                    + self.P.shape[0] \
                                    + self.A.shape[0] \
                                    + self.L.shape[0] \
                                    + self.F.shape[0] \
                                    + self.PHI.shape[0]
        print(f'Params Usage: {(self.numberOfTrainableParams / self.numberOfUsableParams) * 100:.2f}%')
        print(f'Number of Trainable Parameters: {self.numberOfTrainableParams}')
        print(f'Number of Usable Parameters: {self.numberOfUsableParams}')

    def zeros_io(self):
        self.I *= 0
        self.O *= 0

    def zeros_grad(self):
        self.gradA *= 0
        self.gradL *= 0
        self.gradF *= 0
        self.gradPHI *= 0

        self.gradW *= 0
        self.gradR *= 0
        self.gradP *= 0

        self.lossWRTHiddenOutput *= 0

    def update_params(self, v, t, etaW, etaP, etaR, etaA, etaL, etaF, etaPHI):
        self.gradW, self.gradR, self.gradP, \
        self.gradA, self.gradL, self.gradF, self.gradPHI = calculateGradients(
            v, t,
            self.A, self.L, self.F, self.PHI,
            self.W, self.R, self.P, self.I, self.O,
            self.outputIdc, self.nOutputs,
            self.hiddenIdc, self.nHiddens,
            self.gradA, self.gradL, self.gradF, self.gradPHI,
            self.gradW, self.gradP, self.gradR,
            self.lossWRTHiddenOutput, self.nInputs
        )

        self.W += -etaW * self.gradW
        self.P += -etaP * self.gradP
        self.R += -etaR * self.gradR

        self.A += -etaA * self.gradA
        self.L += -etaL * self.gradL
        self.F += -etaF * self.gradF
        self.PHI += -etaPHI * self.gradPHI

    def save_weight_image_per_epoch(self, epoch, simulationFolderPath):
        if not os.path.exists(f'{simulationFolderPath}/graphs'):
            os.mkdir(f'{simulationFolderPath}/graphs')

        if (epoch + 1) % self.decayFrequency == 0:
            WForPlot = self.W.copy()
            WForPlot = np.abs(WForPlot)

            plt.imsave(f'./{simulationFolderPath}/graphs/w{epoch}.jpg',
                       WForPlot[:, self.nInputs: self.nInputs + self.nHiddens].T
                       , cmap='hot')

    def predict(self, u, t):
        self.zeros_io()

        self.I, self.O = forward(t, self.A, self.L, self.F, self.PHI,
                                 self.W, self.P, self.R, self.I, self.O, u,
                                 self.inputIdc, self.hiddenIdc, self.outputIdc)

        return np.abs(self.O[min(self.outputIdc):].reshape((self.nOutputs,)))


def train(network0, trainingDataset, nFrames):
    time.sleep(1)

    for epoch in range(network0.epochs):
        etaW = network0.etaW
        etaP = network0.etaP
        etaR = network0.etaR

        etaA = network0.etaA
        etaL = network0.etaL
        etaF = network0.etaF
        etaPHI = network0.etaPHI

        if (epoch + 1) % network0.decayFrequency == 0:
            etaW *= network0.decay
            etaP *= network0.decay
            etaR *= network0.decay

            etaA *= network0.decay
            etaL *= network0.decay
            etaF *= network0.decay
            etaPHI *= network0.decay

        for u, v in tqdm(trainingDataset,
                              desc=f'Training | Epoch: {epoch + 1:<2} / {network0.epochs:<2}',
                              colour='green',
                              leave=False,
                              total=network0.datasetCap):
            for frame in range(nFrames):
                network0.zeros_grad()
                network0.zeros_io()

                timeStep = round(time.perf_counter(), 2)
                network0.predict(u[0], timeStep)

                # errorRate = network1.predict(np.concatenate((u[0], v)).flatten(),
                #                              timeStep + 1e-3)[0]

                network0.update_params(v, timeStep,
                                       etaW, etaP, etaR,
                                       etaA, etaL, etaF, etaPHI)


def evaluate(network, evaluatingDataset, nFrames: int = 1):
    confusionMatrix = np.zeros((network.nOutputs, network.nOutputs),
                               dtype=np.int64)

    for i, (u, v) in tqdm(enumerate(evaluatingDataset),
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False):
        # timeStep = round(time.perf_counter(), 2)
        predict = network.predict(u[0], i)
        predict = np.argmax(predict)
        target = np.argmax(v)
        confusionMatrix[target, predict] += 1

    return confusionMatrix


if __name__ == '__main__':
    nInputs = 28 * 28
    nHiddens = 80  # NOQA
    nOutputs = 10
    nFrames = 60

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    EPOCHS = 4
    datasetCap = nOutputs * 800
    record_no = 85
    nTest = 100

    trainingClasses = np.zeros((nOutputs,), dtype=np.int64)
    evaluatingClasses = np.zeros((nOutputs,), dtype=np.int64)

    evaluatingDataset = []
    trainingDataset = []
    validationSet = []

    if isTrain:
        trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=True)
        evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)

        # Preprocessing Training Data
        for u, v in tqdm(trainingMNIST,
                         desc='Preprocessing Training Data: ',
                         colour='green'):
            if trainingClasses[v] < datasetCap // nOutputs:
                # Target output vector v
                t = np.zeros((nOutputs,), dtype=np.float64)
                t[v] = 1

                u = np.array(u)
                u = u / np.max(u)

                trainingDataset.append([u.reshape(1, -1), t])
                trainingClasses[v] += 1

        # Preprocessing Evaluating Data
        for u, v in tqdm(evaluatingMNIST,
                         desc='Preprocessing Evaluating Data: ',
                         colour='green'):
            # Target output vector v
            t = np.zeros((nOutputs,), dtype=np.float64)
            t[v] = 1

            u = np.array(u)
            u = u / np.max(u)

            evaluatingDataset.append([u.reshape(1, -1), t])
            evaluatingClasses[v] += 1

    network0 = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, decayFrequency=2,
        minRadius=400, maxRadius=800,
        minPHI=-PI, maxPHI=PI,
        minA=0.5, maxA=0.8,
        minF=20, maxF=40,
        minL=2, maxL=8,
        etaW=1e-2, etaP=1e-2, etaR=1e-2, decay=0.1,
        etaA=1e-2, etaL=1e-2, etaF=1e-2, etaPHI=1e-2,
        width=2000, height=2000, depth=2000,
        hiddenZoneOffset=500, outputZoneOffset=400,
        maxInputPerNode=maxInputPerNode // 6, minInputPerNode=maxInputPerNode // 8,
        maxOutputPerNode=maxOutputPerNode // 6, minOutputPerNode=maxOutputPerNode // 8,
        datasetName='mnist', train=isTrain
    )

    if isTrain:
        startTime = time.perf_counter()  # Start timer
        train(network0, trainingDataset, nFrames)

        confusionMatrix = evaluate(network0, evaluatingDataset, nFrames)

        diag = np.diag(confusionMatrix)
        print(diag)

        sumOfDiag = np.sum(diag)
        accuracy = sumOfDiag * 100. / len(evaluatingDataset)

        totalTime = time.perf_counter() - startTime  # End timer

        print(f'Accuracy: {accuracy}%')
        print(f'Total Established Time: {totalTime} (sec)')

        network0.plot()
