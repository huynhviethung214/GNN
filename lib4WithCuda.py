import math
import numpy as np

from numba import njit, prange, cuda


@njit()
def initialize(inputPerNode, outputPerNode, N, nInputs, nHiddens, nOutputs):  # NOQA
    # Let `i` is a row index and `j` is a column index
    # Node(s) is said to be connecting to other Node(s) has to have the following condition: C[i, j] = 1
    # Node(s) is said to be connected by other Node(s) has to have the following condition: C[j, i] = 1
    # * Diagonal will always be 0
    # * Input node column's elements are always equal to 0
    # * Output node row's elements are always equal to 0
    # * We will use C[i, j] for to find the next node(s) in forwarding operation (C[i, j] == 1)
    #           and C[j, i] for to find the next node(s) in backpropagation operation (C[j, i] == 1)
    #
    # * Example:
    #            0  1  2  3 <- (Column Index)
    #      0     0  0  1  0
    #      1     1  0  0  1
    #      2     0  0  0  0
    #      3     1  0  1  0
    #      ^
    #      |
    # (Row Index)

    # Initialize Nodes
    # inputIdc = [i for i in range(nInputs)]
    # outputIdc = [N - i for i in range(1, nOutputs + 1)]
    # hiddenIdc = []
    #
    # copiedInputIdc = inputIdc.copy()
    # copiedInputIdc.extend(outputIdc)
    #
    # for i in range(N):
    #     if i not in copiedInputIdc:
    #         hiddenIdc.append(i)

    inputIdc = [i for i in range(nInputs)]  # NOQA
    hiddenIdc = [i + nInputs for i in range(nHiddens)]  # NOQA
    outputIdc = [i + nInputs + nHiddens for i in range(nOutputs)]  # NOQA

    # inputIdc = [i for i in range(nInputs)]  # NOQA
    # outputIdc = [N - i for i in range(1, nOutputs + 1)]  # NOQA
    # outputIdc.sort()
    # # print(outputIdc)
    # hiddenIdc = []
    #
    # for i in range(N):  # NOQA
    #     if i not in outputIdc and i not in inputIdc:
    #         hiddenIdc.append(i)

    W = np.random.uniform(-6 / np.sqrt(nInputs + nOutputs),
                          6 / np.sqrt(nInputs + nOutputs), (N, N))  # Xavier's Weight Initialization
    C = np.ones((N, N))  # Connections Matrix

    # Initialize the connection matrix
    while True:
        copiedC = np.copy(C)

        # turn off connection(s) that is violating the constraint (see above)
        np.fill_diagonal(copiedC, 0)

        for i in outputIdc:
            copiedC[i] = 0

        for i in inputIdc:
            copiedC[:, i] = 0

        # Fix two-way connections
        for i in range(N):
            for j in range(N):
                if (i != j) and (copiedC[i, j] == copiedC[j, i] == 1):
                    rand = np.random.randint(0, 1)
                    if rand == 0:
                        copiedC[i, j] = 0
                    else:
                        copiedC[j, i] = 0

        # Only allow a certain number of connections (number of input(s) or number of output(s) / node)
        for inputIdx in inputIdc:
            nodeIdc = list(np.where(copiedC[inputIdx, :] == 1)[0])

            if len(nodeIdc) > outputPerNode:
                for r in range(len(nodeIdc) - outputPerNode):
                    randint = np.random.randint(0, len(nodeIdc))
                    copiedC[inputIdx, nodeIdc[randint]] = 0
                    nodeIdc.pop(randint)

        for hiddenIdx in hiddenIdc:
            nodeIdc = list(np.where(copiedC[hiddenIdx, :] == 1)[0])

            if len(nodeIdc) > outputPerNode:
                for r in range(len(nodeIdc) - outputPerNode):
                    randint = np.random.randint(0, len(nodeIdc))
                    copiedC[hiddenIdx, nodeIdc[randint]] = 0
                    nodeIdc.pop(randint)

        for hiddenIdx in hiddenIdc:
            nodeIdc = list(np.where(copiedC[:, hiddenIdx] == 1)[0])

            if len(nodeIdc) > inputPerNode:
                for r in range(len(nodeIdc) - inputPerNode):
                    randint = np.random.randint(0, len(nodeIdc))
                    copiedC[nodeIdc[randint], hiddenIdx] = 0
                    nodeIdc.pop(randint)

        for outputIdx in outputIdc:
            connections = list(np.where(copiedC[:, outputIdx] == 1)[0])

            if len(connections) > inputPerNode:
                for r in range(len(connections) - inputPerNode):
                    randint = np.random.randint(0, len(connections))
                    copiedC[connections[randint], outputIdx] = 0
                    connections.pop(randint)

        # If path exist then break the loop
        if pathExist(copiedC, inputIdc):
            # Remove 2-ways connection(s)
            for n0 in np.where(copiedC == 1)[0]:
                for n1 in np.where(C[n0] == 1)[0]:
                    if copiedC[n0, n1] == 1 and copiedC[n1, n0] == 1:
                        copiedC[n1, n0] = 0

            np.fill_diagonal(copiedC, 0)

            for i in outputIdc:
                copiedC[i] = 0

            for i in inputIdc:
                copiedC[:, i] = 0

            C = copiedC
            W = W * C
            return W, C


# Check if there is any path that lead from input node(s) to output node(s) using BFS
@njit()
def pathExist(C, inputIdc):
    queue = inputIdc.copy()

    while len(queue) > 0:
        e = queue[0]
        queue.pop(0)

        toIdc = np.where(C[e, :] == 1)[0]

        # toIdc represent the connections from i-th node to j-th node
        if toIdc.size > 0:
            for toIdx in toIdc:
                queue.append(toIdx)
        else:
            return True
    return False


@njit()
def magnitude(v0, v1):
    return np.sqrt((v0[0] - v1[0]) ** 2
                   + (v0[1] - v1[1]) ** 2
                   + (v0[2] - v1[2]) ** 2)


# Activation Function
@njit()
def tanh(x):
    return np.tanh(x)


@njit()
def tanhPrime(x):
    return 1 - (tanh(x) ** 2)


def makeF(sigmoid):
    @cuda.jit(device=True)
    def f(x):
        return sigmoid(x)

    return f


# Derivative of activation function
def makeFPrime(sigmoidPrime):
    @cuda.jit(device=True)
    def fPrime(x):
        return sigmoidPrime(x)

    return fPrime


@njit()
def relu(x):
    return np.maximum(0., x)


@njit()
def reluPrime(x):
    if x <= 0.:
        return 0.
    return 1.


@njit()
def sin(x):
    return math.sin(x * math.pi / 180.)


@njit()
def cos(x):
    return math.cos(x * math.pi / 180.) * (math.pi / 180.)


@njit()
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


@njit()
def softmaxPrime(x):
    x = softmax.reshape(-1, 1)
    return np.diagflat(x) - np.dot(x, x.T)


@cuda.jit(device=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))  # Sigmoid


@cuda.jit(device=True)
def sigmoidPrime(x):
    return sigmoid(x) * (1. - sigmoid(x))  # Derivative of Sigmoid


# Loss Function
def g(p, t):
    return np.sum((p - t) ** 2) / p.shape[0]


def gPrime(p, t):
    return 2 * (p - t) / p.shape[0]


@cuda.jit()
def forwardInputNodes(I, O, us):
    x = cuda.grid(1)

    if x < I.shape[0]:
        I[x] = us[x]
        O[x] = I[x]


@cuda.jit()
def forwardHiddenAndOutputNodes(I, O, P, R, W,
                                nodeIdx, bias,
                                minHiddenIdx, maxHiddenIdx):
    x = cuda.grid(1)

    if x < I.shape[0]:
        vectorMagnitude = math.sqrt(
            (P[nodeIdx, 0] - P[x, 0]) ** 2 +
            (P[nodeIdx, 1] - P[x, 1]) ** 2 +
            (P[nodeIdx, 2] - P[x, 2]) ** 2
        )

        if vectorMagnitude == 0.:
            vectorMagnitude = 1.

        I[nodeIdx] += O[x] \
                      * W[x, nodeIdx] \
                      * R[x, 0] \
                      / vectorMagnitude

        if minHiddenIdx <= nodeIdx <= maxHiddenIdx:
            O[nodeIdx] = 1. / (1. + math.exp(-(I[nodeIdx] + bias[0])))
        else:
            O[nodeIdx] = 1. / (1. + math.exp(-(I[nodeIdx])))


def forward(W, I, O, P, R, outputIdc,
            hiddenIdc, u, bias):
    threadsPerBlock = (32, 32)

    blocksPerGridX = math.ceil(I.shape[0] / threadsPerBlock[0])
    blocksPerGrid = (blocksPerGridX, 1)

    forwardInputNodes[blocksPerGrid, threadsPerBlock](I, O, u)

    assert not np.isnan(I.copy_to_host()).any()
    assert not np.isnan(O.copy_to_host()).any()

    for nodeIdx in np.concatenate((hiddenIdc, outputIdc), dtype=np.int64).flatten():
        forwardHiddenAndOutputNodes[blocksPerGrid, threadsPerBlock] \
            (I, O, P, R, W,
             nodeIdx, bias,
             min(hiddenIdc[:, 0]), max(hiddenIdc[:, 0]))

    assert not np.isnan(I.copy_to_host()).any()
    assert not np.isnan(O.copy_to_host()).any()

    return I.copy_to_host(), O.copy_to_host()


@njit()
def getMagnitude(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@cuda.jit('float64[:], float64[:], '
          'float64[:,:], float64[:,:], float64[:,:], '
          'float64[:], float64[:], float64[:], '
          'float64[:,:], float64[:,:], float64[:,:],'
          'int64, int64, int64, int64',
          cache=True)
def calculateOutputGradients(predict, target,
                             gradU, gradP, gradR,
                             lossWRTHiddenOutput,
                             O, I, W, P, R,
                             outputIdx, outIdx,
                             minHiddenIdx, maxHiddenIdx):
    x = cuda.grid(1)  # x = batch, y = N

    if x < I.shape[0]:
        vectorMagnitude = math.sqrt(
            (P[outputIdx, 0] - P[x, 0]) ** 2 +
            (P[outputIdx, 1] - P[x, 1]) ** 2 +
            (P[outputIdx, 2] - P[x, 2]) ** 2
        )

        if vectorMagnitude == 0.:
            vectorMagnitude = 1.

        outputSigmoid = 1. / (1. + math.exp(-I[outputIdx]))
        outputSigmoidPrime = outputSigmoid * (1. - outputSigmoid)

        if minHiddenIdx <= x <= maxHiddenIdx:
            lossWRTHiddenOutput[x] += W[x, outputIdx] \
                                      * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                                      * outputSigmoidPrime \
                                      * R[x, 0] \
                                      / vectorMagnitude

            if minHiddenIdx <= x + 1 <= maxHiddenIdx:
                lossWRTHiddenOutput[x] += lossWRTHiddenOutput[x + 1]

        gradU[x, outputIdx] += O[x] \
                               * (2 * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) / predict.shape[0]) \
                               * outputSigmoidPrime \
                               * R[x, 0] \
                               / vectorMagnitude

        gradR[x, 0] += O[x] \
                       * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                       * outputSigmoidPrime \
                       * W[x, outputIdx] \
                       / vectorMagnitude

        # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
        x0 = P[outputIdx, 0]
        y0 = P[outputIdx, 1]
        z0 = P[outputIdx, 2]

        x1 = P[x, 0]
        y1 = P[x, 1]
        z1 = P[x, 2]

        gradP[outputIdx, 0] += O[x] \
                               * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                               * outputSigmoidPrime \
                               * W[x, outputIdx] \
                               * (R[x, 0] * -(x0 - x1) / (vectorMagnitude ** 3))

        gradP[outputIdx, 1] += O[x] \
                               * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                               * outputSigmoidPrime \
                               * W[x, outputIdx] \
                               * (R[x, 0] * -(y0 - y1) / (vectorMagnitude ** 3))

        gradP[outputIdx, 2] += O[x] \
                               * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                               * outputSigmoidPrime \
                               * W[x, outputIdx] \
                               * (R[x, 0] * -(z0 - z1) / (vectorMagnitude ** 3))

        gradP[x, 0] += O[x] \
                       * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                       * outputSigmoidPrime \
                       * W[x, outputIdx] \
                       * (R[x, 0] * (x0 - x1) / (vectorMagnitude ** 3))

        gradP[x, 1] += O[x] \
                       * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                       * outputSigmoidPrime \
                       * W[x, outputIdx] \
                       * (R[x, 0] * (y0 - y1) / (vectorMagnitude ** 3))

        gradP[x, 2] += O[x] \
                       * (2 * (predict[outIdx] - target[outIdx]) / predict.shape[0]) \
                       * outputSigmoidPrime \
                       * W[x, outputIdx] \
                       * (R[x, 0] * (z0 - z1) / (vectorMagnitude ** 3))


@cuda.jit(cache=True)
def calculateHiddenGradients(gradU, gradP, gradR,
                             lossWRTHiddenOutput,
                             O, I, W, P, R, hiddenIdx):
    x = cuda.grid(1)  # x = batch, y = N

    if x < I.shape[0]:
        vectorMagnitude = math.sqrt(
            (P[hiddenIdx[0], 0] - P[x, 0]) ** 2 +
            (P[hiddenIdx[0], 1] - P[x, 1]) ** 2 +
            (P[hiddenIdx[0], 2] - P[x, 2]) ** 2
        )

        if vectorMagnitude == 0.:
            vectorMagnitude = 1.

        hiddenSigmoid = 1. / (1. + math.exp(-I[hiddenIdx[0]]))
        hiddenSigmoidPrime = hiddenSigmoid * (1. - hiddenSigmoid)

        gradU[x, hiddenIdx[0]] += O[x] \
                                  * lossWRTHiddenOutput[hiddenIdx[0]] \
                                  * hiddenSigmoidPrime \
                                  * R[x, 0] \
                                  / vectorMagnitude

        gradR[x, 0] += O[x] \
                       * lossWRTHiddenOutput[hiddenIdx[0]] \
                       * hiddenSigmoidPrime \
                       * W[x, hiddenIdx[0]] \
                       / vectorMagnitude

        # Loss w.r.t position of outputNode (0 <=> x0 coordinate, 1 <=> y0 coordinate)
        x0 = P[hiddenIdx[0], 0]
        y0 = P[hiddenIdx[0], 1]
        z0 = P[hiddenIdx[0], 2]

        x1 = P[x, 0]
        y1 = P[x, 1]
        z1 = P[x, 2]

        gradP[hiddenIdx[0], 0] += O[x] \
                                  * lossWRTHiddenOutput[hiddenIdx[0]] \
                                  * hiddenSigmoidPrime \
                                  * W[x, hiddenIdx[0]] \
                                  * (R[x, 0] * -(x0 - x1) / (vectorMagnitude ** 3))

        gradP[hiddenIdx[0], 1] += O[x] \
                                  * lossWRTHiddenOutput[hiddenIdx[0]] \
                                  * hiddenSigmoidPrime \
                                  * W[x, hiddenIdx[0]] \
                                  * (R[x, 0] * -(y0 - y1) / (vectorMagnitude ** 3))

        gradP[hiddenIdx[0], 2] += O[x] \
                                  * lossWRTHiddenOutput[hiddenIdx[0]] \
                                  * hiddenSigmoidPrime \
                                  * W[x, hiddenIdx[0]] \
                                  * (R[x, 0] * -(z0 - z1) / (vectorMagnitude ** 3))

        gradP[x, 0] += O[x] \
                       * lossWRTHiddenOutput[hiddenIdx[0]] \
                       * hiddenSigmoidPrime \
                       * W[x, hiddenIdx[0]] \
                       * (R[x, 0] * (x0 - x1) / (vectorMagnitude ** 3))

        gradP[x, 1] += O[x] \
                       * lossWRTHiddenOutput[hiddenIdx[0]] \
                       * hiddenSigmoidPrime \
                       * W[x, hiddenIdx[0]] \
                       * (R[x, 0] * (y0 - y1) / (vectorMagnitude ** 3))

        gradP[x, 2] += O[x] \
                       * lossWRTHiddenOutput[hiddenIdx[0]] \
                       * hiddenSigmoidPrime \
                       * W[x, hiddenIdx[0]] \
                       * (R[x, 0] * (z0 - z1) / (vectorMagnitude ** 3))


def calculateGradients(predicts, targets,
                       W, I, O, P, R,
                       gradU, gradR, gradP,
                       nInputs, nHiddens, nOutputs,  # NOQA
                       hiddenIdc, outputIdc,
                       lossWRTHiddenOutput):
    threadsPerBlock = (32, 32)

    blocksPerGridX = math.ceil(gradU.shape[0] / threadsPerBlock[0])
    blocksPerGrid = (blocksPerGridX, 1)

    for outputIdx in outputIdc:
        outIdx = outputIdx[0] - (nInputs + nHiddens)  # NOQA  (Index of output array with shape (10,))
        calculateOutputGradients[blocksPerGrid, threadsPerBlock] \
            (predicts, targets,
             gradU, gradP, gradR,
             lossWRTHiddenOutput,
             O, I, W, P, R,
             outputIdx[0], outIdx,
             min(hiddenIdc[:, 0]),
             max(hiddenIdc[:, 0]))

    assert not np.isnan(gradU).any()
    assert not np.isnan(gradR).any()
    assert not np.isnan(gradP).any()

    for hiddenIdx in hiddenIdc[::-1]:
        calculateHiddenGradients[blocksPerGrid, threadsPerBlock] \
            (gradU, gradP, gradR,
             lossWRTHiddenOutput,
             O, I, W, P, R, hiddenIdx)

    assert not np.isnan(gradU).any()
    assert not np.isnan(gradP).any()
    assert not np.isnan(gradR).any()

    return gradU.copy_to_host(), gradR.copy_to_host(), gradP.copy_to_host()
