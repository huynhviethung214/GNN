import numpy as np

from numba import njit, prange

PI = np.round(np.pi, 4)


@njit(fastmath=True, parallel=True)
def removeRandomConnections(inputIdc, hiddenIdc, outputIdc,
                            inputPerNode, outputPerNode, C):
    for i in prange(inputIdc.shape[0]):
        inputIdx = inputIdc[i]
        nodeIdc = list(np.where(C[inputIdx, :] == 1)[0])

        if len(nodeIdc) > outputPerNode:
            for r in range(len(nodeIdc) - outputPerNode):
                randint = np.random.randint(0, len(nodeIdc))
                C[inputIdx, nodeIdc[randint]] = 0
                nodeIdc.pop(randint)

    for i in prange(hiddenIdc.shape[0]):
        hiddenIdx = hiddenIdc[i]
        nodeIdc = list(np.where(C[hiddenIdx, :] == 1)[0])

        if len(nodeIdc) > outputPerNode:
            for r in range(len(nodeIdc) - outputPerNode):
                randint = np.random.randint(0, len(nodeIdc))
                C[hiddenIdx, nodeIdc[randint]] = 0
                nodeIdc.pop(randint)

    for i in prange(hiddenIdc.shape[0]):
        hiddenIdx = hiddenIdc[i]
        nodeIdc = list(np.where(C[:, hiddenIdx] == 1)[0])

        if len(nodeIdc) > inputPerNode:
            for r in range(len(nodeIdc) - inputPerNode):
                randint = np.random.randint(0, len(nodeIdc))
                C[nodeIdc[randint], hiddenIdx] = 0
                nodeIdc.pop(randint)

    for i in prange(outputIdc.shape[0]):
        outputIdx = outputIdc[i]
        connections = list(np.where(C[:, outputIdx] == 1)[0])

        if len(connections) > inputPerNode:
            for r in range(len(connections) - inputPerNode):
                randint = np.random.randint(0, len(connections))
                C[connections[randint], outputIdx] = 0
                connections.pop(randint)

    return C


@njit()
def enableConnections(C, idc0, idc1):
    for idx0 in idc0:
        for idx1 in idc1:
            C[idx0, idx1] = 1.


@njit(fastmath=True, parallel=True)
def initialize(inputPerNode, outputPerNode, N,
               nInputs, nOutputs,
               inputIdc, hiddenIdc, outputIdc):  # NOQA
    # Let `i` denoted as a row index and `j` denoted as a column index
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

        # for i in outputIdc:
        #     C[i, :min(outputIdc)] = 0

        for i in outputIdc:
            C[i, :] = 0

        for i in inputIdc:
            C[:, i] = 0

        # Fix two-way connections
        for i in range(N):
            for j in range(N):
                # if (i not in hiddenIdc or j not in hiddenIdc) and (i != j):
                if (i != j) and (C[i, j] == C[j, i] == 1):
                    rand = np.random.randint(0, 1)
                    if rand == 0:
                        C[i, j] = 0
                    else:
                        C[j, i] = 0

        C = removeRandomConnections(inputIdc, hiddenIdc, outputIdc,
                                    inputPerNode, outputPerNode, C)

        # Only allow a certain number of connections
        # for inputIdx in inputIdc:
        #     nodeIdc = list(np.where(C[inputIdx, :] == 1)[0])
        #
        #     if len(nodeIdc) > outputPerNode:
        #         for r in range(len(nodeIdc) - outputPerNode):
        #             randint = np.random.randint(0, len(nodeIdc))
        #             C[inputIdx, nodeIdc[randint]] = 0
        #             nodeIdc.pop(randint)
        #
        # for hiddenIdx in hiddenIdc:
        #     nodeIdc = list(np.where(C[hiddenIdx, :] == 1)[0])
        #
        #     if len(nodeIdc) > outputPerNode:
        #         for r in range(len(nodeIdc) - outputPerNode):
        #             randint = np.random.randint(0, len(nodeIdc))
        #             C[hiddenIdx, nodeIdc[randint]] = 0
        #             nodeIdc.pop(randint)
        #
        # for hiddenIdx in hiddenIdc:
        #     nodeIdc = list(np.where(C[:, hiddenIdx] == 1)[0])
        #
        #     if len(nodeIdc) > inputPerNode:
        #         for r in range(len(nodeIdc) - inputPerNode):
        #             randint = np.random.randint(0, len(nodeIdc))
        #             C[nodeIdc[randint], hiddenIdx] = 0
        #             nodeIdc.pop(randint)
        #
        # for outputIdx in outputIdc:
        #     connections = list(np.where(C[:, outputIdx] == 1)[0])
        #
        #     if len(connections) > inputPerNode:
        #         for r in range(len(connections) - inputPerNode):
        #             randint = np.random.randint(0, len(connections))
        #             C[connections[randint], outputIdx] = 0
        #             connections.pop(randint)

        # If path exist then break the loop
        if pathExist(C, list(inputIdc)):
            # Remove 2-ways connection(s)
            # for hiddenIdx in hiddenIdc:
            #     for n1 in np.where(C[hiddenIdx] == 1)[0]:
            #         if C[hiddenIdx, n1] == 1 and C[n1, hiddenIdx] == 1:
            #             C[n1, hiddenIdx] = 0

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


@njit(parallel=True, fastmath=True)
def removeSynapses(W, P, R, nodeIdc, nInputs, nOutputs):
    for n0 in nodeIdc:
        for n1 in nodeIdc:
            if n0 != n1:
                signalPreservationLevel = R[n0] / magnitude(P[n0], P[n1])
                if signalPreservationLevel <= 1e-5 and W[n0, n1] != 0:
                    W[n0, n1] = 0.
                elif 1e-2 <= signalPreservationLevel <= 1 and W[n0, n1] == 0:
                    W[n0, n1] = np.random.uniform(-np.sqrt(6 / (nInputs + nOutputs)),
                                                  np.sqrt(6 / (nInputs + nOutputs)), 1)[0]

    return W


@njit(fastmath=True)
def magnitude(v0, v1):
    return np.sqrt((v0[0] - v1[0]) ** 2 +
                   (v0[1] - v1[1]) ** 2 +
                   (v0[2] - v1[2]) ** 2)


# Activation Function
@njit(fastmath=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))  # Sigmoid


# Derivative of activation function
@njit(fastmath=True)
def sigmoidPrime(x):
    return sigmoid(x) * (1. - sigmoid(x))  # Derivative of Sigmoid


@njit(fastmath=True)
def sin(x):
    return np.sin(x * PI / 180.)


# Derivative of activation function
@njit(fastmath=True)
def cos(x):
    return np.cos(x * PI / 180.) * PI / 180.


@njit(fastmath=True)
def relu(x):
    return np.maximum(0., x)


@njit(fastmath=True)
def reluPrime(x):
    if x < 0.:
        return 0.
    return 1.


@njit(fastmath=True)
def tanh(x):
    return np.tanh(x)


@njit(fastmath=True)
def tanhPrime(x):
    return 1 - tanh(x) ** 2


# Loss Function
@njit(fastmath=True)
def mse(p, t, nOutputs):
    return np.sum((t - p) ** 2, axis=0) / nOutputs


@njit(fastmath=True)
def msePrime(p, t, nOutputs):
    return (2 / nOutputs) * (p - t)  # (p - t) 'cuz we take the partial derivative of (p - t)^2


@njit(fastmath=True)
def crossEntropy(p, t, *args):
    return -t * np.log(p)


@njit(fastmath=True)
def crossEntropyPrime(p, t, *args):
    return p - t


@njit(fastmath=True)
def absFunc(p, t, *args):
    return np.abs(p - t)


@njit(fastmath=True)
def absPrime(p, t, *args):
    return (p - t) / np.abs(p - t)


@njit(parallel=True, fastmath=True)
def forward(W, I, O, P, R, B,
            inputIdc, hiddenIdc, outputIdc, u, f):
    for i in prange(len(inputIdc)):
        inputIdx = inputIdc[i]
        I[inputIdx] = u[inputIdx]
        O[inputIdx] = u[inputIdx]

    for i in prange(len(hiddenIdc)):
        hiddenIdx = hiddenIdc[i]
        byIdc = np.where(W[:, hiddenIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[hiddenIdx], P[byIdx])

            if vectorMagnitude == 0.:
                vectorMagnitude = 1.

            # print('I', byIdx, hiddenIdx, O[byIdx], W[byIdx, hiddenIdx], R[byIdx, 0],
            #       vectorMagnitude)

            I[hiddenIdx] += (O[byIdx]
                             * W[byIdx, hiddenIdx]
                             * R[byIdx, 0]
                             / vectorMagnitude)

            # print('I', byIdx, hiddenIdx, I[hiddenIdx])

            # if I[hiddenIdx] + B[hiddenIdx] >= 0.:
            #     O[hiddenIdx] += f(I[hiddenIdx] + B[hiddenIdx])
            #     I[hiddenIdx] = 0.

        # print('O', hiddenIdx, I[hiddenIdx], B[hiddenIdx])
        O[hiddenIdx] = f(I[hiddenIdx] + B[hiddenIdx])

    for i in prange(len(outputIdc)):
        outputIdx = outputIdc[i]
        byIdc = np.where(W[:, outputIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[outputIdx], P[byIdx])

            if vectorMagnitude == 0.:
                vectorMagnitude = 1.

            # print('I', byIdx, outputIdx, O[byIdx], W[byIdx, outputIdx], R[byIdx, 0],
            #       vectorMagnitude)

            I[outputIdx] += (O[byIdx]
                             * W[byIdx, outputIdx]
                             * R[byIdx, 0]
                             / vectorMagnitude)

            # print(I[outputIdx])

            # if I[outputIdx] + B[outputIdx] >= 0.:
            #     O[outputIdx] += f(I[outputIdx] + B[outputIdx])
            #     I[outputIdx] = 0.

        O[outputIdx] = f(I[outputIdx] + B[outputIdx])

    return I, O


# @njit(parallel=True, fastmath=True)
def calculateGradients(target, W, I, O, P, R,
                       gradU, gradR, gradP, gradB,
                       nInputs, nHiddens, nOutputs,  # NOQA
                       hiddenIdc, outputIdc, fPrime, gPrime,
                       lossWRTHiddenOutput):
    for i in prange(nOutputs):
        outputIdx = outputIdc[i]
        outIdx = outputIdx - (nInputs + nHiddens)
        byIdc = np.where(W[:, outputIdx] != 0)[0]

        gradB[outputIdx] = gPrime(O[outputIdx],
                                  target[outIdx],
                                  nOutputs) \
                           * fPrime(I[outputIdx])

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[outputIdx])
            # print(vectorMagnitude)

            if len(np.where(hiddenIdc == byIdx)[0]):
                lossWRTHiddenOutput[byIdx] += W[byIdx, outputIdx] \
                                              * gradB[outputIdx] \
                                              * R[byIdx, 0] \
                                              / vectorMagnitude

                if len(np.where(hiddenIdc == byIdx + 1)[0]) and W[byIdx, byIdx + 1] != 0.:
                    lossWRTHiddenOutput[byIdx] += lossWRTHiddenOutput[byIdx + 1]

            gradU[byIdx, outputIdx] += O[byIdx] \
                                       * gradB[outputIdx] \
                                       * R[byIdx, 0] \
                                       / vectorMagnitude

            gradR[byIdx, 0] += O[byIdx] \
                               * gradB[outputIdx] \
                               * W[byIdx, outputIdx] \
                               / vectorMagnitude

            # print(f'Grad W:\n{gradU}\n')

            # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
            x0 = P[outputIdx, 0]
            y0 = P[outputIdx, 1]
            z0 = P[outputIdx, 2]

            x1 = P[byIdx, 0]
            y1 = P[byIdx, 1]
            z1 = P[byIdx, 2]

            gradP[outputIdx, 0] += O[byIdx] \
                                   * gradB[outputIdx] \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0]
                                      * -(x0 - x1)
                                      / (vectorMagnitude ** 3))

            gradP[outputIdx, 1] += O[byIdx] \
                                   * gradB[outputIdx] \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0]
                                      * -(y0 - y1)
                                      / (vectorMagnitude ** 3))

            gradP[outputIdx, 2] += O[byIdx] \
                                   * gradB[outputIdx] \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0]
                                      * -(z0 - z1)
                                      / (vectorMagnitude ** 3))

            gradP[byIdx, 0] += O[byIdx] \
                               * gradB[outputIdx] \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0]
                                  * (x0 - x1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 1] += O[byIdx] \
                               * gradB[outputIdx] \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0]
                                  * (y0 - y1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 2] += O[byIdx] \
                               * gradB[outputIdx] \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0]
                                  * (z0 - z1)
                                  / (vectorMagnitude ** 3))

    for i in prange(nHiddens):
        hiddenIdx = hiddenIdc[::-1][i]
        byIdc = np.where(W[:, hiddenIdx] != 0)[0]

        gradB[hiddenIdx] = lossWRTHiddenOutput[hiddenIdx] \
                           * fPrime(I[hiddenIdx])

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[hiddenIdx])

            gradU[byIdx, hiddenIdx] += O[byIdx] \
                                       * gradB[hiddenIdx] \
                                       * R[byIdx, 0] \
                                       / vectorMagnitude

            gradR[byIdx, 0] += O[byIdx] \
                               * gradB[hiddenIdx] \
                               * W[byIdx, hiddenIdx] \
                               / vectorMagnitude

            # Loss w.r.t position of outputNode (0 <=> x0 coordinate, 1 <=> y0 coordinate)
            x0 = P[hiddenIdx, 0]
            y0 = P[hiddenIdx, 1]
            z0 = P[hiddenIdx, 2]

            x1 = P[byIdx, 0]
            y1 = P[byIdx, 1]
            z1 = P[byIdx, 2]

            gradP[hiddenIdx, 0] += O[byIdx] \
                                   * gradB[hiddenIdx] \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0]
                                      * -(x0 - x1)
                                      / (vectorMagnitude ** 3))

            gradP[hiddenIdx, 1] += O[byIdx] \
                                   * gradB[hiddenIdx] \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0]
                                      * -(y0 - y1)
                                      / (vectorMagnitude ** 3))

            gradP[hiddenIdx, 2] += O[byIdx] \
                                   * gradB[hiddenIdx] \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0]
                                      * -(z0 - z1)
                                      / (vectorMagnitude ** 3))

            gradP[byIdx, 0] += O[byIdx] \
                               * gradB[hiddenIdx] \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0]
                                  * (x0 - x1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 1] += O[byIdx] \
                               * gradB[hiddenIdx] \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0]
                                  * (y0 - y1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 2] += O[byIdx] \
                               * gradB[hiddenIdx] \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0]
                                  * (z0 - z1)
                                  / (vectorMagnitude ** 3))

    return gradU, gradB, gradR, gradP
