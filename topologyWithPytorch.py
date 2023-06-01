import json
import os
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import sleep
from lib4WithPytorch import forward, initialize, gradients, g
from torch.profiler import profile, record_function, ProfilerActivity

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Network:
    def __init__(self,
                 bias: float = 1, decay: float = 0.1,
                 etaW: float = 1e-3, etaR: float = 1e-3, etaP: float = 1e-3,
                 minRadius: int = 10, maxRadius: int = 20, frequency: int = 5,
                 nInputs: int = 784, nHiddens: int = 10, nOutputs: int = 10,  # NOQA
                 trainingBatches: list = None, evaluatingBatches: list = None,
                 datasetCap: int = 1000, batchSize: int = 64, epochs: int = 10,
                 width: int = 20, height: int = 20, depth: int = 20,
                 hiddenZoneOffset: int = 600, outputZoneOffset: int = 200,
                 maxInputPerNode: int = 2, minInputPerNode: int = 1,
                 maxOutputPerNode: int = 20, minOutputPerNode: int = 1,
                 save: bool = True, datasetName: str = 'MNIST', configFilePath: str = ''):  # NOQA
        # assert (rOffset < minRadius) and (minRadius < maxRadius), 'Radius\' Offset must be less than Minimum Radius'

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
        # self.rOffset = rOffset
        self.frequency = frequency

        self.batchSize = batchSize
        self.datasetCap = datasetCap
        self.datasetName = datasetName

        self.trainingBatches = trainingBatches
        self.evaluatingBatches = evaluatingBatches

        self.simulationIdx = 0
        while True:
            try:
                self.filename = f'pytorch_3d_nodes_simulation_{self.simulationIdx}'
                self.dirName = f'records/{self.filename}'

                os.mkdir(f'./{self.dirName}')
                break

            except FileExistsError:
                self.simulationIdx += 1
        print(f'Using file\'s name: {self.filename}')

        self.save = save
        self.configFilePath = configFilePath
        self.lossPerEpoch = []
        self.accuracy = 0

        # I and O matrix for training dataset
        self.I = torch.ones((self.batchSize, self.N),
                            dtype=torch.float64,
                            device=DEVICE)
        self.I.share_memory_()

        self.O = torch.ones((self.batchSize, self.N),
                            dtype=torch.float64,
                            device=DEVICE)
        self.O.share_memory_()

        self.C = np.zeros((self.N, self.N),
                          dtype=np.float64)

        # Tensors for forward pass
        self.W = np.zeros((self.N, self.N),
                          dtype=np.float64)  # Use to store size of every nodes in the network

        self.P = np.zeros((self.N, 3),
                          dtype=np.float64)  # Use to store position (x, y) of node in the network

        self.R = np.zeros((self.N, 1),
                          dtype=np.float64)  # Use to store size of every nodes in the network

        # Gradients Tensor for backward pass
        # Gradient U use to update W
        self.gradU = torch.zeros((self.batchSize, self.N, self.N),
                                 dtype=torch.float64,
                                 device=DEVICE)
        self.gradU.share_memory_()

        # Gradient S use to update the size of each node
        self.gradR = torch.zeros((self.batchSize, self.N),
                                 dtype=torch.float64,
                                 device=DEVICE)
        self.gradR.share_memory_()

        # Gradient P use to update position of each node
        self.gradP = torch.zeros((self.batchSize, self.N, 3),
                                 dtype=torch.float64,
                                 device=DEVICE)
        self.gradP.share_memory_()

        self.lossWRTHiddenOutput = torch.zeros((self.batchSize, self.N),
                                               dtype=torch.float64,
                                               device=DEVICE)
        self.lossWRTHiddenOutput.share_memory_()

        self.inputIdc = np.array([i for i in range(nInputs)],
                                 dtype=np.int64).reshape((self.nInputs,))

        self.hiddenIdc = np.array([i + nInputs for i in range(nHiddens)],
                                  dtype=np.int64).reshape((self.nHiddens,))  # NOQA

        self.outputIdc = np.array([i + nInputs + nHiddens for i in range(nOutputs)],
                                  dtype=np.int64).reshape((self.nOutputs,))

        self.nodeIdc = np.concatenate([self.inputIdc, self.hiddenIdc, self.outputIdc],
                                      axis=0)

        self.numberOfTrainableParams = 0
        self.numberOfUsableParams = 0

        self.construct_network()

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        P = self.P.cpu().numpy()

        for inputIdx in self.inputIdc:
            ax.plot(P[inputIdx, 0], P[inputIdx, 1], P[inputIdx, 2],
                    marker='o', color='r')

        for hiddenIdx in self.hiddenIdc:
            ax.plot(P[hiddenIdx, 0], P[hiddenIdx, 1], P[hiddenIdx, 2],
                    marker='o', color='g')

        for outputIdx in self.outputIdc:
            ax.plot(P[outputIdx, 0], P[outputIdx, 1], P[outputIdx, 2],
                    marker='o', color='b')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def save_config(self):
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
            # 'rOffset': self.rOffset,
            'frequency': self.frequency,
            'datasetCap': self.datasetCap,
            'dataset': self.datasetName,
            'batchSize': self.batchSize,
            'losses': self.lossPerEpoch,
            'accuracy': self.accuracy
        }

        with open(f'./{self.dirName}/configs.json', 'w') as f:
            json.dump(configs, f, indent=4)

    def save_result(self):
        np.save(f'./{self.dirName}/w.npy', self.W.cpu().numpy())
        np.save(f'./{self.dirName}/r.npy', self.R.cpu().numpy())
        np.save(f'./{self.dirName}/p.npy', self.P.cpu().numpy())

    def load_config(self):
        assert os.path.exists(f'./{self.configFilePath}'), f'./{self.configFilePath} cannot be found'
        with open(f'./{self.configFilePath}', 'r') as f:
            for key, value in json.load(f).items():
                setattr(self, key, value)

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

    def getVectorMagnitude(self, u, v):
        return np.sqrt(np.sum((u - v) ** 2))

    def isOverlap(self, idx, v, zone: str = 'i'):
        assert self.outputZoneOffset > 0 and self.hiddenZoneOffset > 0

        if zone == 'i':
            zoneNodeIdc = self.inputIdc
        elif zone == 'h':
            zoneNodeIdc = self.hiddenIdc
        else:
            zoneNodeIdc = self.outputIdc

        for nodeIdx in zoneNodeIdc:
            if idx != nodeIdx and self.getVectorMagnitude(v, self.P[nodeIdx]) <= 0.8:
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
            start=start, stop=stopWidth, dtype=np.float64
        ), size=1)[0]

        y = np.random.choice(np.arange(
            start=start, stop=stopHeight, dtype=np.float64
        ), size=1)[0]

        z = np.random.choice(np.arange(
            start=start, stop=stopDepth, dtype=np.float64
        ), size=1)[0]

        return np.array([x, y, z], dtype=np.float64)

    def getNodePosition(self, nodeIdx, zone: str = 'h'):
        v = self.drawVecSamples(zone)
        while self.isOverlap(nodeIdx, v, zone):
            v = self.drawVecSamples(zone)

        return v

    def construct_network(self):
        # Add Nodes
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
        np.fill_diagonal(self.W, 0)

        self.W[self.outputIdc] = 0
        self.W[:, self.inputIdc] = 0

        # Remove loops in Hidden Layer
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

        if len(loops) > 0:
            for loop in loops:
                if len(loop) > 1:
                    self.W[loop[-2], loop[-1]] = 0

        print('\nInfo: ')
        self.numberOfTrainableParams = len(np.where(self.W != 0)[0])
        self.numberOfUsableParams = np.square(self.N) \
                                    - self.N \
                                    - self.nInputs * self.N \
                                    - self.nOutputs * (self.N - self.nInputs)
        print(f'Params Usage: {(self.numberOfTrainableParams / self.numberOfUsableParams) * 100:.2f}%')
        print(f'Number of Trainable Parameters: {self.numberOfTrainableParams}')
        print(f'Number of Usable Parameters: {self.numberOfUsableParams}')

        self.W = torch.tensor(self.W,
                              dtype=torch.float64,
                              device=DEVICE)
        self.W.share_memory_()

        self.P = torch.tensor(self.P,
                              dtype=torch.float64,
                              device=DEVICE)
        self.P.share_memory_()

        self.R = torch.tensor(self.R,
                              dtype=torch.float64,
                              device=DEVICE)
        self.R.share_memory_()

        # self.getConnectionMatrix()
        self.C = self.W.clone()
        self.C[self.C != 0.] = 1.
        self.C.share_memory_()

        assert self.W.is_shared(), 'W is not shared'
        assert self.P.is_shared(), 'P is not shared'
        assert self.R.is_shared(), 'R is not shared'

    # def getConnectionMatrix(self):
    #     self.C = torch.as_tensor(self.W)
    #     self.C.requires_grad = False
    #     self.C[self.C != 0] = 1
    #
    #     assert self.C.is_shared(), 'C is not shared'

    def train(self) -> None:
        # torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            sleep(1)
            # torch.cuda.synchronize()

            for epoch in range(self.epochs):
                etaW = self.etaW
                etaP = self.etaP
                etaR = self.etaR
                loss = 0

                if (epoch + 1) % self.frequency == 0:
                    etaW *= self.decay
                    etaP *= self.decay
                    etaR *= self.decay

                for batch in tqdm(self.trainingBatches,
                                  desc=f'Number of trainable parameters: {self.numberOfTrainableParams:<5}, '
                                       f'Number of usable parameters: {self.numberOfUsableParams:<5}, '
                                       f'Epoch {epoch:<2} / {self.epochs:<2}',
                                  colour='green',
                                  leave=False):
                    us, vs = batch[0], batch[1]
                    # us = np.array(self.trainingDatasetX[batchIdx * self.batchSize:
                    #                                     self.batchSize * (batchIdx + 1)]) \
                    #     .reshape((self.batchSize, self.nInputs))
                    # us = torch.tensor(us,
                    #                   dtype=torch.float64,
                    #                   device=DEVICE)
                    #
                    # vs = np.array(self.trainingDatasetY[batchIdx * self.batchSize:
                    #                                     self.batchSize * (batchIdx + 1)]) \
                    #     .reshape((self.batchSize, self.nOutputs))
                    # vs = torch.tensor(vs,
                    #                   dtype=torch.float64,
                    #                   device=DEVICE)

                    # self.I = torch.fill_(self.I, 0)
                    # self.O = torch.fill_(self.O, 0)
                    # self.O[:, self.inputIdc] = us.clone()
                    # self.O = us.clone()
                    # self.I = us.clone()

                    self.O, self.I = forward(self.W, us.clone(),
                                             us.clone(), self.P, self.R,
                                             self.hiddenIdc, self.outputIdc,
                                             self.bias)
                    predicts = self.O[:, min(self.outputIdc):]

                    # assert not torch.isnan(self.I).any()
                    # assert not torch.isnan(self.O).any()

                    self.gradU = torch.fill_(self.gradU, 0)
                    self.gradP = torch.fill_(self.gradP, 0)
                    self.gradR = torch.fill_(self.gradR, 0)

                    self.gradU, self.gradP, self.gradR = gradients(
                        predicts, vs, self.C,
                        self.W, self.I, self.O, self.P, self.R,
                        self.gradU, self.gradR, self.gradP,
                        self.nInputs, self.nHiddens,
                        self.nOutputs, self.hiddenIdc, self.outputIdc,
                        self.lossWRTHiddenOutput
                    )

                    # assert not torch.isnan(self.gradU).any()
                    # assert not torch.isnan(self.gradR).any()
                    # assert not torch.isnan(self.gradR).any()

                    print(self.gradU[0, 0])

                    self.W += -etaW * torch.sum(self.gradU, dim=0) / self.batchSize
                    self.P += -etaP * torch.sum(self.gradP, dim=0) / self.batchSize
                    self.R += (-etaR
                               * torch.sum(self.gradR, dim=0)
                               / self.batchSize).view(-1, 1)
                    loss += g(predicts, vs, nOutputs)

            self.lossPerEpoch.append(float(loss / self.datasetCap))

    def evaluate(self) -> float:
        score = 0
        acc = 0

        for batch in tqdm(self.evaluatingBatches,
                          desc=f'Number of trainable parameters: {self.numberOfTrainableParams:<5}, '
                               f'Number of usable parameters: {self.numberOfUsableParams:<5}, '
                               f'Accuracy: {acc}',
                          colour='green',
                          leave=False):
            # us = np.array(self.evaluatingDatasetX[batchIdx * self.batchSize: self.batchSize * (batchIdx + 1)]) \
            #     .reshape((self.batchSize, self.nInputs))
            # us = torch.tensor(us)
            #
            # vs = np.array(self.evaluatingDatasetY[batchIdx * self.batchSize: self.batchSize * (batchIdx + 1)]) \
            #     .reshape((self.batchSize, self.nOutputs))
            # vs = torch.tensor(vs, device=DEVICE)

            # self.I = torch.fill_(self.I, 0)
            # self.O = torch.fill_(self.O, 0)
            # self.O[:, self.inputIdc] = us.clone()
            us, vs = batch[0], batch[1]

            self.O = us.clone()
            self.I = us.clone()

            self.O, _ = forward(self.W, self.I,
                                self.O, self.P, self.R,
                                self.hiddenIdc, self.outputIdc,
                                self.bias)
            predicts = self.O[:, min(self.outputIdc):].to(DEVICE)

            score += len(torch.where(
                torch.argmax(predicts, dim=1) == torch.argmax(vs, dim=1)
            )[0])

        self.accuracy = float(
            f'{(score / (self.batchSize * len(self.evaluatingBatches))) * 100:.2f}'
        )

        return self.accuracy


if __name__ == '__main__':
    nInputs = 28 * 28
    nHiddens = 500  # NOQA
    nOutputs = 10
    N = nInputs + nHiddens + nOutputs

    datasetCap = 8000
    batchSize = 200

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    trainingMNIST = torchvision.datasets.MNIST('./data',
                                               train=True,
                                               download=False)
    trainingDatasetX = []
    trainingDatasetY = []

    for i, (u, v) in enumerate(tqdm(trainingMNIST,
                                    desc='Preprocessing Training Data: ',
                                    colour='green')):
        if i % datasetCap == 0 and i != 0:
            break

        # Target output vector v
        t = np.zeros((1, nOutputs))
        t[0, v] = 1
        trainingDatasetY.append(t)

        u = np.array(u).flatten()
        u = u / np.max(u)
        x = np.zeros((N,))
        x[:nInputs] = u
        trainingDatasetX.append(x)

    # Preprocessing Evaluating Data
    evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)
    evaluatingDatasetX = []
    evaluatingDatasetY = []

    for u, v in tqdm(evaluatingMNIST,
                     desc='Preprocessing Evaluating Data: ',
                     colour='green'):
        # Target output vector v
        t = np.zeros((1, nOutputs))
        t[0, v] = 1
        evaluatingDatasetY.append(t)

        u = np.array(u).flatten()
        u = u / np.max(u)
        x = np.zeros((N,))
        x[:nInputs] = u
        evaluatingDatasetX.append(x)

    trainingBatches = []
    evaluatingBatches = []

    for batchIdx in tqdm(range(datasetCap // batchSize),
                         desc='Batch-ing Dataset',
                         colour='green'):
        trainingBatches.append([
            torch.tensor(
                trainingDatasetX[batchSize * batchIdx: batchSize * (batchIdx + 1)],
                device=DEVICE,
                dtype=torch.float64
            ).view(batchSize, N),
            torch.tensor(
                trainingDatasetY[batchSize * batchIdx: batchSize * (batchIdx + 1)],
                device=DEVICE,
                dtype=torch.float64
            ).view(batchSize, nOutputs),
        ])

        evaluatingBatches.append([
            torch.tensor(
                evaluatingDatasetX[batchSize * batchIdx: batchSize * (batchIdx + 1)],
                device=DEVICE,
                dtype=torch.float64
            ).view(batchSize, N),
            torch.tensor(
                evaluatingDatasetY[batchSize * batchIdx: batchSize * (batchIdx + 1)],
                device=DEVICE,
                dtype=torch.float64
            ).view(batchSize, nOutputs),
        ])

    # c = list(zip(trainingDatasetX, trainingDatasetY))
    # for _ in range(100):
    #     random.shuffle(c)
    # trainingDatasetX, trainingDatasetY = zip(*c)
    #
    # c = list(zip(evaluatingDatasetX, evaluatingDatasetY))
    # for _ in range(100):
    #     random.shuffle(c)
    # evaluatingDatasetX, evaluatingDatasetY = zip(*c)

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        trainingBatches=trainingBatches, evaluatingBatches=evaluatingBatches,
        epochs=20, datasetCap=datasetCap, frequency=5,
        minRadius=800, maxRadius=1000, save=True, batchSize=batchSize,
        hiddenZoneOffset=4000, outputZoneOffset=500,
        width=10000, height=10000, depth=10000,
        etaW=1e-2, etaP=1e-2, etaR=1e-2, decay=0.1, bias=-0.8,
        maxInputPerNode=maxInputPerNode, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode, minOutputPerNode=maxOutputPerNode // 2,
    )

    # with profile(activities=[ProfilerActivity.CPU],
    #              record_shapes=True,
    #              with_stack=True,
    #              profile_memory=True,
    #              with_modules=True) as prof:
    #     with record_function("model_inference"):
    network.train()

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5)
    #       .table(sort_by="cpu_time_total", row_limit=10))

    print(f'Accuracy: {network.evaluate()}%')
    network.save_result()
    network.save_config()
    network.plot()
