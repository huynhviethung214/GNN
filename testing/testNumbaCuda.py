import copy
import math
from time import sleep

import numpy as np
import torchvision

from copy import deepcopy
from numba import cuda
from scipy.ndimage import zoom, rotate
from tqdm import tqdm
from models.GNNWithCuda import zero_io, zero_grads, calculateInputOfNeurons, calculateOutputOfNeurons, \
    gradBOutputToHidden, remainGradsOutputToHidden, gradBHiddenToInput, remainGradsHiddenToInput
from modules.modules import initialize, sigmoid, sigmoidPrime, msePrime
from modules.modules import forward as fw
from modules.modules import calculateGradients as backwardCPU


np.set_printoptions(precision=6)


def batchingDataset(Nin, Nh, Nout, datasetCap, batchSize, inputShape, transformData):
    # Preprocessing parameters
    minScalingFactor = 1.2
    maxScalingFactor = 1.4

    minRotateX = -20
    maxRotateX = 20

    P = 0.7
    numberOfPixelToJitter = 50
    minJitterBrightness = 0.3
    maxJitterBrightness = 0.6

    trainingClasses = np.zeros((Nout,), dtype=np.int64)

    trainingSet = []
    evaluatingSet = []

    trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=True)
    evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)

    # Preprocessing Training Data
    for u, v in tqdm(trainingMNIST,
                     desc='Preprocessing Training Data:',
                     colour='green'):
        # Target output vector v
        t = np.zeros((Nout,), dtype=np.float64)
        t[v] = 1

        u = np.array(u)
        u = u / 255.

        if transformData:
            if np.random.uniform(0, 1, 1)[0] <= P:
                u[np.random.randint(0, u.shape[0], numberOfPixelToJitter),
                np.random.randint(0, u.shape[1], numberOfPixelToJitter)] \
                    += np.random.uniform(minJitterBrightness, maxJitterBrightness, numberOfPixelToJitter)
                u[u > 0.6] = 1.

            if np.random.uniform(0, 1, 1)[0] <= P:
                scalingFactor = np.random.uniform(minScalingFactor, maxScalingFactor, 1)[0]
                scalingFactor = np.round(scalingFactor, 1)

                u = zoom(u, scalingFactor)

            if np.random.uniform(0, 1, 1)[0] <= P:
                u = rotate(u, np.random.randint(minRotateX, maxRotateX, 1)[0])

            if u.shape[0] != inputShape[0] and u.shape[1] != inputShape[1]:
                if u.shape[0] % 2 != 0:
                    u = u[0: u.shape[0] - 1, 0: u.shape[1] - 1]

                marginX = (u.shape[0] - inputShape[0]) // 2
                marginY = (u.shape[1] - inputShape[1]) // 2

                u = u[marginX: u.shape[0] - marginX, marginY: u.shape[1] - marginY]
                u[u > 0.4] = 1.

        if trainingClasses[v] < datasetCap // Nout:
            u = u.flatten()
            u = np.hstack((u, np.zeros((Nin + Nh + Nout))))

            trainingSet.append([u, t])
            trainingClasses[v] += 1

    # Preprocessing Evaluating Data
    for u, v in tqdm(evaluatingMNIST,
                     desc='Preprocessing Evaluating Data: ',
                     colour='green'):
        # Target output vector v
        t = np.zeros((Nout,), dtype=np.float64)
        t[v] = 1

        u = np.array(u)
        u = u / 255.

        evaluatingSet.append([u.flatten(), t])

    trainingXBatches = []
    trainingYBatches = []

    xBatch = []
    yBatch = []

    for x, y in trainingSet:
        xBatch.append(x)
        yBatch.append(y)

        if len(xBatch) == batchSize:
            trainingXBatches.append(np.array(xBatch))
            trainingYBatches.append(np.array(yBatch))

            xBatch = []
            yBatch = []

    return (trainingXBatches, trainingYBatches), evaluatingSet


def backward(batchSize, v, gradW, gradB, gradP, gradR, lossWRTHiddenOutput,
             W, C, Ch, P, R, I, O, Nin, Nh, Nout, hiddenIdc, outputIdc):
    for outputIdx in outputIdc:
        outIdx = outputIdx - (Nin + Nh)
        gradBOutputToHidden[(32, 32), (32, 32)](outputIdx, outIdx, I, O,
                                                v, Nout, gradB)
        print(gradB.copy_to_host()[0])
        # remainGradsOutputToHidden[(32, 32), (32, 32)](outputIdx,
        #                                               Ch, O, W, P, R,
        #                                               gradW, gradB, gradP, gradR,
        #                                               lossWRTHiddenOutput, C)
        # print(lossWRTHiddenOutput.copy_to_host())

    # for hiddenIdx in hiddenIdc[::-1]:
    #     gradBHiddenToInput[(32, 32), (32, 32)](batchSize, hiddenIdx, I,
    #                                            lossWRTHiddenOutput, gradB)
    #     remainGradsHiddenToInput[(32, 32), (32, 32)](hiddenIdx, O, W, P, R,
    #                                                  gradW, gradB, gradP, gradR, C)


def forward(batchSize, W, B, P, R, I, O, hiddenIdc, outputIdc):
    for hiddenIdx in hiddenIdc:
        calculateInputOfNeurons[(32, 32), (32, 32)](batchSize, W, P, R,
                                                    I, O, hiddenIdx)
        # cuda.synchronize()
        calculateOutputOfNeurons[(32, 32), (32, 32)](batchSize, O, I,
                                                     B, hiddenIdx)
        # cuda.synchronize()

    for outputIdx in outputIdc:
        calculateInputOfNeurons[(32, 32), (32, 32)](batchSize, W, P, R,
                                                    I, O, outputIdx)
        # cuda.synchronize()
        calculateOutputOfNeurons[(32, 32), (32, 32)](batchSize, O, I,
                                                     B, outputIdx)
        # cuda.synchronize()


def train(W, Ch, B, P, R, I, O, gradW, gradB, gradP, gradR, lossWRTHiddenOutput,
          Nin, Nh, Nout, hiddenIdc, outputIdc, trainingSet):
    for epochIdx in range(epochs):
        for batchIdx in tqdm(range(len(trainingSet[0])),
                             desc=f'Epoch: {epochIdx} / {epochs}',
                             leave=False):
            xs, ys = trainingSet[0][batchIdx], trainingSet[1][batchIdx]
            ys = cuda.to_device(ys)

            zero_io[(32, 32), (16, 16)](I, O)
            zero_grads[(32, 32), (16, 16)](gradW, gradB, gradP, gradR)

            forward(W, B, P, R, I, O, hiddenIdc, outputIdc)
            backward(ys, gradW, gradB, gradP, gradR, lossWRTHiddenOutput,
                     W, Ch, P, R, I, O, Nin, Nh, Nout, hiddenIdc, outputIdc)

            W = np.array(W.copy_to_host())
            B = np.array(B.copy_to_host())
            P = np.array(P.copy_to_host())
            R = np.array(R.copy_to_host())

            W = W - 0.01 * (np.sum(np.array(gradW.copy_to_host()), axis=0) / batchSize)
            B = B - 0.01 * (np.sum(np.array(gradB.copy_to_host()), axis=0) / batchSize)
            P = P - 0.01 * (np.sum(np.array(gradP.copy_to_host()), axis=0) / batchSize)
            R = R - 0.01 * (np.sum(np.array(gradR.copy_to_host()), axis=0) / batchSize)

            W = cuda.to_device(W)
            B = cuda.to_device(B)
            P = cuda.to_device(P)
            R = cuda.to_device(R)


@cuda.jit()
def sigmoidGPU(x, out):
    out[0] = 1. / (1. + math.exp(-x))


if __name__ == '__main__':
    transformData = True

    datasetCap = 8000
    batchSize = 1
    epochs = 24
    inputShape = [28, 28]

    # Nin = inputShape[0] * inputShape[1]
    # Nh = 80
    # Nout = 10

    Nin = 2
    Nh = 2
    Nout = 1

    inputIdc = np.array([i for i in range(Nin)], dtype=np.int64).flatten()
    hiddenIdc = np.array([i + Nin for i in range(Nh)], dtype=np.int64).flatten()
    outputIdc = np.array([i + Nin + Nh for i in range(Nout)], dtype=np.int64).flatten()

    N = Nin + Nh + Nout

    # Variables
    maxInputPerNode = Nin + Nh
    maxOutputPerNode = Nout + Nh

    inputPerNeuron = np.random.randint(maxInputPerNode // 2, maxInputPerNode, 1)[0]
    outputPerNeuron = np.random.randint(maxOutputPerNode // 2, maxOutputPerNode, 1)[0]
    W, C = initialize(inputPerNeuron, outputPerNeuron,
                      N, Nin, Nout, inputIdc, hiddenIdc, outputIdc)

    Ch = np.copy(C)
    Ch[:Nin] = 0.
    Ch[Nin + Nh:] = 0.
    print(f'Ch:\n{Ch}')

    B = np.random.uniform(-0.1, -0.8, (N,))
    P = np.random.uniform(0, 8000, (N, 3))
    R = np.random.uniform(2000, 6000, (N, 1))

    # I = np.zeros((batchSize, N), dtype=np.float32)
    # O = np.zeros((batchSize, N), dtype=np.float32)

    # I = cuda.to_device(I)
    # O = cuda.to_device(O)

    # # Random Generated Samples
    x = np.array(np.random.uniform(0, 1, (batchSize, Nin)), dtype=np.float32)
    # x = cuda.to_device(x)

    y = np.array(np.random.uniform(0, 1, (batchSize, Nout)), dtype=np.float32)
    # y = cuda.to_device(y)

    # Gradients
    gradW = np.zeros((N, N), dtype=np.float32)
    gradB = np.zeros((N,), dtype=np.float32)
    gradP = np.zeros((N, 3), dtype=np.float32)
    gradR = np.zeros((N, 1), dtype=np.float32)
    lossWRTHiddenOutput = np.zeros((N,), dtype=np.float32)

    I = np.zeros((N,), dtype=np.float32)
    O = np.zeros((N,), dtype=np.float32)

    print(f'\nC:\n{C}')
    print('Forward CPU')
    I, O = fw(W, I, O, P, R, B, inputIdc, hiddenIdc, outputIdc, x[0], sigmoid)
    cpuO = O
    cpuI = I
    print(f'CPU O: {cpuO}')
    print(f'CPU I: {cpuI}')

    # out = cuda.to_device(np.zeros((1,), dtype=np.float32))
    # sigmoidGPU[(16, 16), (8, 8)](0.2, out)
    # print(sigmoid(0.2) - out.copy_to_host())

    # print('Backward CPU')
    # for grad in backwardCPU(y[0], W, I, O, P, R, gradW, gradR, gradP, gradB, Nin, Nh, Nout,
    #                         hiddenIdc, outputIdc, sigmoidPrime, msePrime, lossWRTHiddenOutput):
    #     # grad[grad != 0.] = 1.
    #     # print(grad)
    #     break

    x = np.hstack((x, np.zeros((batchSize, N - Nin))))

    W = cuda.to_device(W)
    Ch = cuda.to_device(Ch)
    B = cuda.to_device(B)
    P = cuda.to_device(P)
    R = cuda.to_device(R)

    gradW = np.zeros((batchSize, N, N), dtype=np.float32)
    gradB = np.zeros((batchSize, N), dtype=np.float32)
    gradP = np.zeros((batchSize, N, 3), dtype=np.float32)
    gradR = np.zeros((batchSize, N, 1), dtype=np.float32)
    lossWRTHiddenOutput = np.zeros((batchSize, N), dtype=np.float32)

    gradW = cuda.to_device(gradW)
    gradB = cuda.to_device(gradB)
    gradP = cuda.to_device(gradP)
    gradR = cuda.to_device(gradR)
    lossWRTHiddenOutput = cuda.to_device(lossWRTHiddenOutput)

    I = cuda.to_device(copy.deepcopy(x))
    O = cuda.to_device(copy.deepcopy(x))

    y = cuda.to_device(y)

    print('\nForward GPU')
    sleep(1.)
    forward(batchSize, W, B, P, R, I, O, hiddenIdc, outputIdc)
    gpuO = O.copy_to_host()[0]
    gpuI = I.copy_to_host()[0]
    print(f'GPU O: {gpuO}')
    print(f'GPU I: {gpuI}')
    print(f'\nDeviation between cpu and gpu output:\n{cpuO - gpuO}')

    # print('Backward GPU')
    # backward(batchSize, y, gradW, gradB, gradP, gradR, lossWRTHiddenOutput,
    #          W, C, Ch, P, R, I, O, Nin, Nh, Nout, hiddenIdc, outputIdc)

    # for i in range(1):
    #     gradWGPU = gradW.copy_to_host()[i]
    #     # gradWGPU[gradWGPU != 0.] = 1.
    #     print(f'Batch {i}:\n{gradWGPU}')
    # print(gradB.copy_to_host())
    # print(gradP.copy_to_host())
    # print(gradR.copy_to_host())

    # MNIST Dataset
    # trainingSet, _ = batchingDataset(Nout, datasetCap, batchSize,
    #                                  inputShape, transformData)
    # forward(W, B, R, P, I, O, x[0], hiddenIdc, outputIdc)
    # print(O.copy_to_host())

    # W = W.copy_to_host()
    # B = B.copy_to_host()
    # P = P.copy_to_host()
    # R = R.copy_to_host()
    #
    # I = np.zeros((N,), dtype=np.float32)
    # O = np.zeros((N,), dtype=np.float32)
    # x = x.copy_to_host()

    # I, O = fw(W, I, O, P, R, B, inputIdc, hiddenIdc, outputIdc, x[0], sigmoid)
    # print(O)

    # timeStart = time.time()
    # for epochIdx in range(1):
    # zero_io[(32, 32), (16, 16)](I, O)
    # zero_grads[(32, 32), (16, 16)](gradW, gradB, gradP, gradR, lossWRTHiddenOutput)

    # print((x[3, 0] == x[0, 0]).all())
    # print(x[3, 0], x[1, 0])

    # for inputIdx in inputIdc:
    #     O[:, inputIdx] = x[:, inputIdx]
    #     I[:, inputIdx] = x[:, inputIdx]

    # I = cuda.to_device(I)
    # O = cuda.to_device(O)
    #
    # forward(W, B, R, P, I, O, x, hiddenIdc, outputIdc)
    # print((np.array(O.copy_to_host()[3, 0])
    #        == np.array(O.copy_to_host()[0, 0])).all())
    # backward(y, gradW, gradB, gradP, gradR, lossWRTHiddenOutput, W,
    #          Ch, P, R, I, O, Nin, Nh, Nout, hiddenIdc, outputIdc)

    # train(W, Ch, B, P, R, I, O, gradW, gradB, gradP, gradR, lossWRTHiddenOutput,
    #       Nin, Nh, Nout, hiddenIdc, outputIdc, trainingSet)
    # timeEnd = time.time()
    # print(f'Total Forward + Backward Time: {timeEnd - timeStart} (s)')

    # fp = '../external data'
    # I = np.zeros((1, N), dtype=np.float32)
    # O = np.zeros((1, N), dtype=np.float32)
    #
    # I = cuda.to_device(I)
    # O = cuda.to_device(O)
    #
    # for image in os.listdir(fp):
    #     imagePath = f'{fp}/{image}'
    #     label = image.split('.')[0]
    #
    #     if ' ' in label:
    #         label = label.split(' ')[0]
    #
    #     u = cv2.imread(imagePath)
    #     u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)
    #
    #     u = np.array(u) / np.max(u)
    #     u = u.flatten()
    #     u = cuda.to_device(u.reshape((1, -1)))
    #
    #     res = np.zeros((Nout,), dtype=np.int64)
    #
    #     for i in range(1):
    #         forward(W, B, R, P, I, O, u, hiddenIdc, outputIdc)
    #         predict = O[0, Nin + Nh:].copy_to_host()
    #         predict = np.array(predict).flatten()
    #         res[np.argmax(predict)] += 1
    #
    #     print(f'Best Prediction: {np.argmax(res)} | Target: {label:<10} |'
    #           f' Image: {imagePath:<40}')
