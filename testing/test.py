import copy
import cupy as cp
import numpy as np
from tqdm import tqdm

from modules.modules import initialize, forward, sigmoid, calculateGradients, sigmoidPrime, msePrime

Nin = 784
Nh = 80
Nout = 10
N = Nin + Nh + Nout
batchSize = 800

inputIdc = np.array([i for i in range(Nin)], dtype=cp.int64)
hiddenIdc = np.array([i + Nin for i in range(Nh)], dtype=cp.int64)
outputIdc = np.array([i + Nin + Nh for i in range(Nout)], dtype=cp.int64)

# Variables
maxInputPerNode = Nin + Nh
maxOutputPerNode = Nout + Nh

inputPerNeuron = np.random.randint(maxInputPerNode // 2, maxInputPerNode, 1)[0]
outputPerNeuron = np.random.randint(maxOutputPerNode // 2, maxOutputPerNode, 1)[0]
W, C = initialize(inputPerNeuron, outputPerNeuron,
                  N, Nin, Nout, inputIdc, hiddenIdc, outputIdc)

B = np.random.uniform(-0.1, -0.8, (N,))
P = np.random.uniform(0, 8000, (N, 3))
R = np.random.uniform(2000, 6000, (N, 1))

gradUCPU = np.zeros((N, N), dtype=np.float32)
gradBCPU = np.zeros((N,), dtype=np.float32)
gradPCPU = np.zeros((N, 3), dtype=np.float32)
gradRCPU = np.zeros((N, 1), dtype=np.float32)
lossWRTHiddenOutput = np.zeros((N,), dtype=np.float32)

# # Random Generated Samples
x = np.array(np.random.uniform(0, 1, (batchSize, Nin)), dtype=np.float32)
# print(x)
y = np.random.uniform(0, 1, (batchSize, Nout))
# print(f'C:\n{C}\n')

iCPU = np.zeros((N,))
oCPU = np.zeros((N,))

iCPU, oCPU = forward(W, iCPU, oCPU, P, R, B, inputIdc, hiddenIdc, outputIdc, x[0], sigmoid)
gradUCPU, gradBCPU, gradRCPU, gradPCPU = calculateGradients(y[0], W, iCPU, oCPU, P, R,
                                                            gradUCPU, gradRCPU, gradPCPU,
                                                            gradBCPU, Nin, Nh, Nout,
                                                            hiddenIdc, outputIdc,
                                                            sigmoidPrime, msePrime,
                                                            lossWRTHiddenOutput)

# print(f'CPU I:\n{I}\n')
# print(f'CPU O:\n{O}\n')
# print(f'Loss WRT Hidden Output:\n{lossWRTHiddenOutput}\n')
# print(f'Grad W:\n{gradUCPU}\n')
# print(f'Grad B:\n{gradBCPU}\n')
# print(f'Grad R:\n{gradR}\n')
# print(f'Grad P:\n{gradP}\n')

W = cp.array(W)
C = cp.array(C)
B = cp.array(B)
P = cp.array(P)
R = cp.array(R)

Ch = cp.array(copy.deepcopy(C))
Ch[:Nin] = 0.
Ch[Nin + Nh:] = 0.
print(f'Ch:\n{Ch}\n')

# nextCh = Ch[1:]
# nextCh = cp.vstack((nextCh, cp.zeros((1, N), dtype=cp.float32)))
# print(f'Next Ch:\n{nextCh}\n')

gradWGPU = cp.zeros((batchSize, N, N), dtype=cp.float32)
gradBGPU = cp.zeros((batchSize, N), dtype=cp.float32)
gradPGPU = cp.zeros((batchSize, N, 3), dtype=cp.float32)
gradRGPU = cp.zeros((batchSize, N, 1), dtype=cp.float32)
lossWRTHiddenOutputGPU = cp.zeros((batchSize, N), dtype=cp.float32)

inputIdc = cp.array(inputIdc)
hiddenIdc = cp.array(hiddenIdc)
outputIdc = cp.array(outputIdc)

x = np.hstack((x, np.zeros((batchSize, N - Nin))))
x = cp.array(x)
# print(x)
y = cp.array(y)

iGPU = cp.array(copy.deepcopy(x))
oGPU = cp.array(copy.deepcopy(x))


def mag(u, v):
    return cp.sqrt(cp.sum((u - v) ** 2, axis=1))


def f(x):
    return 1. / (1. + cp.exp(-x))


def fPrime(x):
    return f(x) * (1. - f(x))


def gPrime(p, t, Nout):
    return (2. / Nout) * (p - t)


# a = cp.array([
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 1, 0]
# ])
# b = cp.array([[0, 1, 0, 0],
#               [0, 0, 0, 0],
#               [0, 0, 0, 0]])
# print(cp.dot(a, b))
#
# shiftingMatrix = cp.zeros((batchSize, N))
# shiftingMatrix[0, 0] = 1.


for i in tqdm(range(100),
              desc=f'Test Run For 100 Epochs With Batch Size = {batchSize}'):
    for hiddenIdx in hiddenIdc:
        vm = mag(P * C[:, hiddenIdx].reshape((-1, 1)), P[hiddenIdx])

        iGPU[:, hiddenIdx] = cp.sum((oGPU * R.T * W[:, hiddenIdx].T) / vm, axis=1)
        oGPU[:, hiddenIdx] = f(iGPU[:, hiddenIdx] + B[hiddenIdx])

    for outputIdx in outputIdc:
        vm = mag(P * C[:, outputIdx].reshape((-1, 1)),
                 P[outputIdx])

        iGPU[:, outputIdx] = cp.sum((oGPU * R.T * W[:, outputIdx].T) / vm, axis=1)
        oGPU[:, outputIdx] = f(iGPU[:, outputIdx] + B[outputIdx])

    # print(f'Deviation between input of GPU and CPU:\n{cp.array(iCPU) - iGPU[0]}\n')
    # print(f'Deviation between output of GPU and CPU:\n{cp.array(oCPU) - oGPU[0]}\n')

    for outputIdx in outputIdc:
        outIdx = outputIdx - (Nin + Nh)

        gradBGPU[:, outputIdx] = gPrime(oGPU[:, outputIdx],
                                        y[:, outIdx],
                                        Nout) \
                                 * fPrime(iGPU[:, outputIdx])

        vectorMagnitude = mag(P * C[:, outputIdx].reshape((-1, 1)),
                              P[outputIdx].reshape((1, -1))
                              * C[:, outputIdx].reshape((-1, 1)))
        # print(vectorMagnitude)
        vectorMagnitude[vectorMagnitude == 0] = 1.

        lossWRTHiddenOutputGPU += ((W[:, outputIdx]
                                    * Ch[:, outputIdx]).reshape((1, -1))
                                   * gradBGPU[:, outputIdx].reshape((-1, 1))
                                   * R[:, 0].reshape((1, -1))
                                   / vectorMagnitude)

        gradWGPU[:, :, outputIdx] = oGPU \
                                    * gradBGPU[:, outputIdx].reshape((-1, 1)) \
                                    * C[:, outputIdx].reshape((1, -1)) \
                                    * R[:, 0].reshape((1, -1)) \
                                    / vectorMagnitude

        # print(f'Loss WRT Hidden Output:\n{lossWRTHiddenOutput[0]}\n')
        # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
        x0 = P[outputIdx, 0]
        y0 = P[outputIdx, 1]
        z0 = P[outputIdx, 2]

        x1 = P[:, 0]
        y1 = P[:, 1]
        z1 = P[:, 2]

        gradPGPU[:, outputIdx, 0] += cp.sum(
            oGPU * gradBGPU[:, outputIdx].reshape((-1, 1))
            * W[:, outputIdx].reshape((1, -1))
            * (R[:, 0].reshape((1, -1))
               * -(x0 - x1)
               / (vectorMagnitude ** 3)),
            axis=1
        )

        gradPGPU[:, outputIdx, 1] += cp.sum(
            oGPU * gradBGPU[:, outputIdx].reshape((-1, 1))
            * W[:, outputIdx].reshape((1, -1))
            * (R[:, 0].reshape((1, -1))
               * -(y0 - y1)
               / (vectorMagnitude ** 3)),
            axis=1
        )

        gradPGPU[:, outputIdx, 2] += cp.sum(
            oGPU
            * gradBGPU[:, outputIdx].reshape((-1, 1))
            * W[:, outputIdx].reshape((1, -1))
            * (R[:, 0].reshape((1, -1))
               * -(z0 - z1)
               / (vectorMagnitude ** 3)),
            axis=1
        )

        gradPGPU[:, :, 0] += oGPU \
                             * gradBGPU[:, outputIdx].reshape((-1, 1)) \
                             * W[:, outputIdx].reshape((1, -1)) \
                             * (R[:, 0].reshape((1, -1))
                                * (x0 - x1)
                                / (vectorMagnitude ** 3))

        gradPGPU[:, :, 1] += oGPU \
                             * gradBGPU[:, outputIdx].reshape((-1, 1)) \
                             * W[:, outputIdx].reshape((1, -1)) \
                             * (R[:, 0].reshape((1, -1))
                                * (y0 - y1)
                                / (vectorMagnitude ** 3))

        gradPGPU[:, :, 2] += oGPU \
                             * gradBGPU[:, outputIdx].reshape((-1, 1)) \
                             * W[:, outputIdx].reshape((1, -1)) \
                             * (R[:, 0].reshape((1, -1))
                                * (z0 - z1)
                                / (vectorMagnitude ** 3))

    # print(f'Deviation between gradW of CPU and GPU:\n{cp.array(gradUCPU) - gradWGPU[0]}\n')
    # print(f'Deviation between gradB of CPU and GPU:\n{cp.array(gradBCPU) - gradBGPU[0]}\n')

    for hiddenIdx in hiddenIdc[::-1]:
        vectorMagnitude = mag(P * C[:, hiddenIdx].reshape((-1, 1)),
                              P[hiddenIdx].reshape((1, -1)) * C[:, hiddenIdx].reshape((-1, 1)))
        vectorMagnitude[vectorMagnitude == 0] = 1.

        if hiddenIdx + 1 in hiddenIdc and C[hiddenIdx, hiddenIdx + 1] != 0:
            lossWRTHiddenOutputGPU[:, hiddenIdx] += lossWRTHiddenOutputGPU[:, hiddenIdx + 1]

        gradBGPU[:, hiddenIdx] = lossWRTHiddenOutputGPU[:, hiddenIdx] \
                                 * fPrime(iGPU[:, hiddenIdx])

        gradWGPU[:, :, hiddenIdx] = oGPU \
                                    * C[:, hiddenIdx].reshape((1, -1)) \
                                    * gradBGPU[:, hiddenIdx].reshape((-1, 1)) \
                                    * R[:, 0].reshape((1, -1)) \
                                    / vectorMagnitude

        gradRGPU[:, :, 0] = oGPU \
                            * gradBGPU[:, hiddenIdx].reshape((-1, 1)) \
                            * W[:, hiddenIdx].reshape((1, -1)) \
                            / vectorMagnitude

        # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
        x0 = P[hiddenIdx, 0]
        y0 = P[hiddenIdx, 1]
        z0 = P[hiddenIdx, 2]

        x1 = P[:, 0]
        y1 = P[:, 1]
        z1 = P[:, 2]

        gradPGPU[:, hiddenIdx, 0] += cp.sum(
            oGPU
            * gradBGPU[:, hiddenIdx].reshape((-1, 1))
            * W[:, hiddenIdx].reshape((1, -1))
            * (R[:, 0].reshape((1, -1))
               * -(x0 - x1)
               / (vectorMagnitude ** 3)),
            axis=1
        )

        gradPGPU[:, hiddenIdx, 1] += cp.sum(
            oGPU
            * gradBGPU[:, hiddenIdx].reshape((-1, 1))
            * W[:, hiddenIdx].reshape((1, -1))
            * (R[:, 0].reshape((1, -1))
               * -(y0 - y1)
               / (vectorMagnitude ** 3)),
            axis=1
        )

        gradPGPU[:, hiddenIdx, 2] += cp.sum(
            oGPU
            * gradBGPU[:, hiddenIdx].reshape((-1, 1))
            * W[:, hiddenIdx].reshape((1, -1))
            * (R[:, 0].reshape((1, -1))
               * -(z0 - z1)
               / (vectorMagnitude ** 3)),
            axis=1
        )

        gradPGPU[:, :, 0] += oGPU \
                             * gradBGPU[:, hiddenIdx].reshape((-1, 1)) \
                             * W[:, hiddenIdx].reshape((1, -1)) \
                             * (R[:, 0].reshape((1, -1))
                                * (x0 - x1)
                                / (vectorMagnitude ** 3))

        gradPGPU[:, :, 1] += oGPU \
                             * gradBGPU[:, hiddenIdx].reshape((-1, 1)) \
                             * W[:, hiddenIdx].reshape((1, -1)) \
                             * (R[:, 0].reshape((1, -1))
                                * (y0 - y1)
                                / (vectorMagnitude ** 3))

        gradPGPU[:, :, 2] += oGPU \
                             * gradBGPU[:, hiddenIdx].reshape((-1, 1)) \
                             * W[:, hiddenIdx].reshape((1, -1)) \
                             * (R[:, 0].reshape((1, -1))
                                * (z0 - z1)
                                / (vectorMagnitude ** 3))

    # print(f'Loss WRT Hidden Output:\n{lossWRTHiddenOutputGPU[0]}\n')
    # print(f'Grad W:\n{gradWGPU[0]}\n')
    # print(f'Grad B:\n{gradBGPU[0]}\n')
