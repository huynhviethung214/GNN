import math

import numpy as np

from numba import njit, prange


@njit()
def initialize(inputPerNode, outputPerNode, N,
               nInputs, nOutputs,
               inputIdc, hiddenIdc, outputIdc):  # NOQA
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

    W = np.random.uniform(-np.sqrt(6 / (nInputs + nOutputs)),
                          np.sqrt(6 / (nInputs + nOutputs)), (N, N))  # Xavier's Weight Initialization
    
    # Initialize the connection matrix
    while True:
        C = np.ones((N, N))  # Connections Matrix

        # turn off connection(s) that is violating the constraint (see above)
        np.fill_diagonal(C, 0)

        for i in outputIdc:
            C[i] = 0

        for i in inputIdc:
            C[:, i] = 0

        # Fix two-way connections
        for i in range(N):
            for j in range(N):
                if (i != j) and (C[i, j] == C[j, i] == 1):
                    rand = np.random.randint(0, 1)
                    if rand == 0:
                        C[i, j] = 0
                    else:
                        C[j, i] = 0

        # Only allow a certain number of connections
        for inputIdx in inputIdc:
            nodeIdc = list(np.where(C[inputIdx, :] == 1)[0])

            if len(nodeIdc) > outputPerNode:
                for r in range(len(nodeIdc) - outputPerNode):
                    randint = np.random.randint(0, len(nodeIdc))
                    C[inputIdx, nodeIdc[randint]] = 0
                    nodeIdc.pop(randint)

        for hiddenIdx in hiddenIdc:
            nodeIdc = list(np.where(C[hiddenIdx, :] == 1)[0])

            if len(nodeIdc) > outputPerNode:
                for r in range(len(nodeIdc) - outputPerNode):
                    randint = np.random.randint(0, len(nodeIdc))
                    C[hiddenIdx, nodeIdc[randint]] = 0
                    nodeIdc.pop(randint)

        for hiddenIdx in hiddenIdc:
            nodeIdc = list(np.where(C[:, hiddenIdx] == 1)[0])

            if len(nodeIdc) > inputPerNode:
                for r in range(len(nodeIdc) - inputPerNode):
                    randint = np.random.randint(0, len(nodeIdc))
                    C[nodeIdc[randint], hiddenIdx] = 0
                    nodeIdc.pop(randint)

        for outputIdx in outputIdc:
            connections = list(np.where(C[:, outputIdx] == 1)[0])

            if len(connections) > inputPerNode:
                for r in range(len(connections) - inputPerNode):
                    randint = np.random.randint(0, len(connections))
                    C[connections[randint], outputIdx] = 0
                    connections.pop(randint)

        # If path exist then break the loop
        if pathExist(C, list(inputIdc)):
            # Remove 2-ways connection(s)
            for hiddenIdx in hiddenIdc:
                for n1 in np.where(C[hiddenIdx] == 1)[0]:
                    if C[hiddenIdx, n1] == 1 and C[n1, hiddenIdx] == 1:
                        C[n1, hiddenIdx] = 0

            return W * C, C


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


@njit(fastmath=True)
def magnitude(v0, v1):
    return np.sqrt((v0[0] - v1[0]) ** 2 + (v0[1] - v1[1]) ** 2 + (v0[2] - v1[2]) ** 2)


# Activation Function
@njit()
def sigmoid(x):
    return 1. / (1. + np.exp(-x))  # Sigmoid


# Derivative of activation function
@njit()
def sigmoidPrime(x):
    return sigmoid(x) * (1. - sigmoid(x))  # Derivative of Sigmoid


@njit()
def relu(x):
    return np.maximum(0., x)


@njit()
def reluPrime(x):
    if x <= 0.:
        return 0.
    return 1.


# Loss Function
@njit()
def g(p, t, nOutputs):
    return (np.sum((p - t) ** 2)) / nOutputs


@njit(fastmath=True)
def gPrime(p, t, nOutputs):
    return (2 / nOutputs) * (p - t)


@njit()
def fixInvalid(W, P, minRadius, maxRadius, nodeIdc):
    # Remove invalid connection(s)
    for nodeIdx in nodeIdc:
        for toIdx in np.where(W[nodeIdx] != 0)[0]:
            dst = magnitude(P[nodeIdx], P[toIdx])

            if dst > maxRadius or dst < minRadius:
                W[nodeIdx, toIdx] = 1e-12

    return W


@njit(parallel=True, fastmath=True)
def forward(W, I, O, P, R, nodeIdc,
            inputIdc, hiddenIdc, outputIdc, u, bias, f):
    for i in prange(len(inputIdc)):
        inputIdx = inputIdc[i]
        I[inputIdx, inputIdx] = u[inputIdx]
        O[inputIdx, inputIdx] = I[inputIdx, inputIdx]

    for i in prange(len(nodeIdc)):
        nodeIdx = nodeIdc[i]
        byIdc = np.where(W[:, nodeIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[nodeIdx], P[byIdx])

            I[nodeIdx, nodeIdx] += (O[byIdx, byIdx]
                                    * W[byIdx, nodeIdx]
                                    * R[byIdx, 0]
                                    / vectorMagnitude)

        if nodeIdx in hiddenIdc:
            O[nodeIdx, nodeIdx] = f(I[nodeIdx, nodeIdx] + bias)
        elif nodeIdx in outputIdc:
            O[nodeIdx, nodeIdx] = f(I[nodeIdx, nodeIdx])

    return I, O


@njit()
def getMagnitude(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@njit(parallel=True, fastmath=True)
def calculateGradients(predict, target, W, I, O, P, R,
                       gradU, gradR, gradP,
                       nInputs, nHiddens, nOutputs,  # NOQA
                       hiddenIdc, outputIdc, fPrime, lossWRTHiddenOutput):
    for i in prange(nOutputs):
        outputIdx = outputIdc[i]
        outIdx = outputIdx - (nInputs + nHiddens)  # NOQA  (Index of output array with shape (10,))
        byIdc = np.where(W[:, outputIdx] != 0)[0]

        for byIdx in byIdc:
            vectorMagnitude = magnitude(P[byIdx], P[outputIdx])

            if len(np.where(hiddenIdc == byIdx)[0]):
                lossWRTHiddenOutput[byIdx] += W[byIdx, outputIdx] \
                                              * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                                              * fPrime(I[outputIdx, outputIdx]) \
                                              * R[byIdx, 0] \
                                              / vectorMagnitude

                if len(np.where(hiddenIdc == byIdx + 1)[0]):
                    lossWRTHiddenOutput[byIdx] += lossWRTHiddenOutput[byIdx + 1]

            gradU[byIdx, outputIdx] += O[byIdx, byIdx] \
                                       * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                                       * fPrime(I[outputIdx, outputIdx]) \
                                       * R[byIdx, 0] \
                                       / vectorMagnitude

            gradR[byIdx, 0] += O[byIdx, byIdx] \
                               * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                               * fPrime(I[outputIdx, outputIdx]) \
                               * W[byIdx, outputIdx] \
                               / vectorMagnitude

            # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
            x0 = P[outputIdx, 0]
            y0 = P[outputIdx, 1]
            z0 = P[outputIdx, 2]

            x1 = P[byIdx, 0]
            y1 = P[byIdx, 1]
            z1 = P[byIdx, 2]

            gradP[outputIdx, 0] += O[byIdx, byIdx] \
                                   * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                                   * fPrime(I[outputIdx, outputIdx]) \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0] * -(x0 - x1) / (vectorMagnitude ** 3))

            gradP[outputIdx, 1] += O[byIdx, byIdx] \
                                   * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                                   * fPrime(I[outputIdx, outputIdx]) \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0] * -(y0 - y1) / (vectorMagnitude ** 3))

            gradP[outputIdx, 2] += O[byIdx, byIdx] \
                                   * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                                   * fPrime(I[outputIdx, outputIdx]) \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0] * -(z0 - z1) / (vectorMagnitude ** 3))

            gradP[byIdx, 0] += O[byIdx, byIdx] \
                               * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                               * fPrime(I[outputIdx, outputIdx]) \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0] * (x0 - x1) / (vectorMagnitude ** 3))

            gradP[byIdx, 1] += O[byIdx, byIdx] \
                               * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                               * fPrime(I[outputIdx, outputIdx]) \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0] * (y0 - y1) / (vectorMagnitude ** 3))

            gradP[byIdx, 2] += O[byIdx, byIdx] \
                               * gPrime(predict[outIdx], target[outIdx], nOutputs) \
                               * fPrime(I[outputIdx, outputIdx]) \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0] * (z0 - z1) / (vectorMagnitude ** 3))

    for i in prange(nHiddens):
        hiddenIdx = hiddenIdc[::-1][i]
        byIdc = np.where(W[:, hiddenIdx] != 0)[0]

        for byIdx in byIdc:
            vectorMagnitude = magnitude(P[byIdx],
                                        P[hiddenIdx])  # Magnitude of vector hiddenNode -> outputNode

            gradU[byIdx, hiddenIdx] += O[byIdx, byIdx] \
                                       * lossWRTHiddenOutput[hiddenIdx] \
                                       * fPrime(I[hiddenIdx, hiddenIdx]) \
                                       * R[byIdx, 0] \
                                       / vectorMagnitude

            gradR[byIdx, 0] += O[byIdx, byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx, hiddenIdx]) \
                               * W[byIdx, hiddenIdx] \
                               / vectorMagnitude

            # Loss w.r.t position of outputNode (0 <=> x0 coordinate, 1 <=> y0 coordinate)
            x0 = P[hiddenIdx, 0]
            y0 = P[hiddenIdx, 1]
            z0 = P[hiddenIdx, 2]

            x1 = P[byIdx, 0]
            y1 = P[byIdx, 1]
            z1 = P[byIdx, 2]

            gradP[hiddenIdx, 0] += O[byIdx, byIdx] \
                                   * lossWRTHiddenOutput[hiddenIdx] \
                                   * fPrime(I[hiddenIdx, hiddenIdx]) \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0] * -(x0 - x1) / (vectorMagnitude ** 3))

            gradP[hiddenIdx, 1] += O[byIdx, byIdx] \
                                   * lossWRTHiddenOutput[hiddenIdx] \
                                   * fPrime(I[hiddenIdx, hiddenIdx]) \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0] * -(y0 - y1) / (vectorMagnitude ** 3))

            gradP[hiddenIdx, 2] += O[byIdx, byIdx] \
                                   * lossWRTHiddenOutput[hiddenIdx] \
                                   * fPrime(I[hiddenIdx, hiddenIdx]) \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0] * -(z0 - z1) / (vectorMagnitude ** 3))

            gradP[byIdx, 0] += O[byIdx, byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx, hiddenIdx]) \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0] * (x0 - x1) / (vectorMagnitude ** 3))

            gradP[byIdx, 1] += O[byIdx, byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx, hiddenIdx]) \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0] * (y0 - y1) / (vectorMagnitude ** 3))

            gradP[byIdx, 2] += O[byIdx, byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx, hiddenIdx]) \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0] * (z0 - z1) / (vectorMagnitude ** 3))

    return gradU, gradR, gradP
