import torch
import numpy as np
import torchvision
import cv2

from tqdm import tqdm
from torch.autograd import Variable
from modules.modules_4 import GeneralNeuralNetwork

device = torch.device('cuda')

if __name__ == '__main__':
    Nin = 28 * 28
    Nh = 80  # NOQA
    Nout = 10

    maxInputPerNode = Nin + Nh
    maxOutputPerNode = Nout + Nh

    train = True
    EPOCHS = 12
    datasetCap = 8000
    batchSize = 200
    record_no = 0
    nTest = 100

    classes = np.zeros((Nout,), dtype=np.int64)
    evaluatingDatasetX = []
    evaluatingDatasetY = []

    trainingDatasetX = []
    trainingDatasetY = []

    trainingBatch = []
    evaluatingBatch = []

    model = GeneralNeuralNetwork(
        Nin=Nin, Nh=Nh, Nout=Nout, batch=batchSize, device=device,
        minRadius=1000, maxRadius=3000, width=8000, height=8000, depth=8000,
        minBias=0.1, maxBias=0.8, hiddenZoneOffset=2000, outputZoneOffset=1000,
        maxInputPerNode=maxInputPerNode // 1, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode // 1, minOutputPerNode=maxOutputPerNode // 2,
    ).to(device)

    if train:
        trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=True)
        evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)

        for u, v in tqdm(trainingMNIST,
                         desc='Preprocessing Training Data: ',
                         colour='green'):
            if classes[v] + 1 <= datasetCap // Nout:
                t = np.zeros((Nout,), dtype=np.float64)
                t[v] = 1

                if classes[v] + 1 == 100:
                    cv2.imwrite(f'./training_samples/{v}.jpg', np.array(u))

                u = np.array(u) / np.max(u)

                trainingDatasetX.append(u.flatten())
                trainingDatasetY.append(t)
                classes[v] += 1

        for u, v in tqdm(evaluatingMNIST,
                         desc='Preprocessing Evaluating Data: ',
                         colour='green'):
            t = np.zeros((Nout,), dtype=np.float64)
            t[v] = 1

            u = np.array(u) / np.max(u)

            evaluatingDatasetX.append(u.flatten())
            evaluatingDatasetY.append(t)

    trainingDatasetX = np.array(trainingDatasetX)
    trainingDatasetY = np.array(trainingDatasetY)

    evaluatingDatasetX = np.array(evaluatingDatasetX)
    evaluatingDatasetY = np.array(evaluatingDatasetY)

    # Batch-ing things up
    for batchIdx in tqdm(range(datasetCap // batchSize),
                         desc=f'Batching Training Data',
                         colour='green',
                         leave=False,
                         total=datasetCap // batchSize):
        us = torch.tensor(trainingDatasetX[batchIdx * batchSize:
                                           batchSize * (batchIdx + 1)],
                          device=device,
                          dtype=torch.float32)

        vs = torch.tensor(
            trainingDatasetY[batchIdx * batchSize:
                             batchSize * (batchIdx + 1)],
            device=device,
            dtype=torch.float32,
            requires_grad=True
        )

        trainingBatch.append([us, vs])

    for batchIdx in tqdm(range(len(evaluatingDatasetX) // batchSize),
                         desc=f'Batching Evaluating Data',
                         colour='green',
                         leave=False,
                         total=len(evaluatingDatasetX) // batchSize):
        us = torch.tensor(evaluatingDatasetX[batchIdx * batchSize:
                                             batchSize * (batchIdx + 1)],
                          device=device,
                          dtype=torch.float32)

        vs = torch.tensor(
            evaluatingDatasetY[batchIdx * batchSize:
                               batchSize * (batchIdx + 1)],
            device=device,
            dtype=torch.float32
        )

        evaluatingBatch.append([us, vs])

    if train:
        torch.autograd.set_detect_anomaly(True)
        # optimizer = SGD(model.parameters(), lr=1e-2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()

        for epoch in range(EPOCHS):
            for us, vs in tqdm(trainingBatch,
                               desc=f'{epoch} / {EPOCHS}',
                               colour='green'):
                us = Variable(us).float()
                vs = Variable(vs)

                optimizer.zero_grad()
                ps = model(us)

                # loss = (torch.sum((ps - vs) ** 2, dim=1) / Nout).sum()
                loss = criterion(ps, vs).sum()
                loss.backward()
                optimizer.step()
