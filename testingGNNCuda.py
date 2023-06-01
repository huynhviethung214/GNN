import os
import time

import cv2
import numpy as np

from models.GNNWithCuda import Network
from testing.testNumbaCuda import batchingDataset


if __name__ == '__main__':
    transformData = True

    datasetCap = 8000
    batchSize = 400
    epochs = 24
    inputShape = [28, 28]

    Nin = inputShape[0] * inputShape[1]
    Nh = 80
    Nout = 10

    N = Nin + Nh + Nout

    maxInputPerNode = Nin + Nh
    maxOutputPerNode = Nout + Nh

    trainingSet, _ = batchingDataset(Nin, Nh, Nout, datasetCap,
                                     batchSize, inputShape, transformData)

    network = Network(
        nInputs=Nin, nHiddens=Nh, nOutputs=Nout,
        epochs=epochs, datasetCap=datasetCap, frequency=8,
        minRadius=3000, maxRadius=5000, etaW=1e-0, etaP=1e-0,
        etaB=1e-0, etaR=1e-0, decay=0.1,
        width=8000, height=8000, depth=8000, minBias=0.1, maxBias=0.8,
        hiddenZoneOffset=2000, outputZoneOffset=1000,
        maxInputPerNode=maxInputPerNode // 1, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode // 1, minOutputPerNode=maxOutputPerNode // 2,
        activationFunc='sigmoid', datasetName='mnist', train=True, batch=batchSize
    )

    time.sleep(1)
    network.train(epochs, trainingSet)
    # timeStart = time.time()
    # for epochIdx in range(epochs):
    #     for batchIdx in tqdm(range(len(trainingSet[0])),
    #                          desc=f'Epoch: {epochIdx} / {epochs}',
    #                          leave=False):
    #         network.zero_grads()
    #
    #         xs, ys = trainingSet[0][batchIdx], trainingSet[1][batchIdx]
    #         ys = cuda.to_device(ys)
    #
    #         network.I = cuda.to_device(xs)
    #         network.O = cuda.to_device(np.copy(xs))
    #         predicts = network.predict()
    #
    #         network.backward(ys)
    #
    # timeEnd = time.time()
    # print(f'Total Forward + Backward Time: {timeEnd - timeStart} (s)')

    fp = './external data'
    for image in os.listdir(fp):
        imagePath = f'{fp}/{image}'
        label = image.split('.')[0]

        if ' ' in label:
            label = label.split(' ')[0]

        u = cv2.imread(imagePath)
        u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

        u = np.array(u) / np.max(u)
        u = u.flatten()

        print(f'Prediction result: {np.argmax(network.predict(u))}')

        # res = np.zeros((Nout,), dtype=np.int64)
        # for i in range(100):
        #     predict = network.predict(u)
        #     res[np.argmax(predict)] += 1
        #
        # print(f'Best Prediction: {np.argmax(res)} | Target: {label:<10} |'
        #       f' Image: {imagePath:<40}')
