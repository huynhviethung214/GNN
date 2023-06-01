import numpy as np

from numba import njit, prange


PI = np.round(np.pi, 4)

@njit()
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


@njit()
def initialize(inputPerNode, outputPerNode, N,
               nInputs, nOutputs,
               inputIdc, hiddenIdc, outputIdc,
               numSubNetworks, isSequentialNetwork):  # NOQA
    subNumOfInputNodes = 0
    subNumOfHiddenNodes = 0
    subNumOfOutputNodes = 0

    if isSequentialNetwork:
        assert (inputIdc.shape[0] % numSubNetworks == 0) \
               & (hiddenIdc.shape[0] % numSubNetworks == 0) \
               & (outputIdc.shape[0] % numSubNetworks == 0), 'Number Of Input, ' \
                                                             'Hidden and Output Nodes ' \
                                                             'has to be divisible by ' \
                                                             f'numSubNetworks: {numSubNetworks}'

        subNumOfInputNodes = inputIdc.shape[0] / numSubNetworks
        subNumOfHiddenNodes = hiddenIdc.shape[0] / numSubNetworks
        subNumOfOutputNodes = outputIdc.shape[0] / numSubNetworks

    W = np.random.uniform(-np.sqrt(6 / (nInputs + nOutputs)),
                          np.sqrt(6 / (nInputs + nOutputs)), (N, N))  # Xavier's Weight Initialization

    # Initialize the connection matrix
    while True:
        C = np.ones((N, N))  # Connections Matrix

        # turn off connection(s) that is violating the constraint (see above)
        np.fill_diagonal(C, 0)

        for i in outputIdc:
            C[i, :min(outputIdc)] = 0

        for i in inputIdc:
            C[:, i] = 0

        # Fix two-way connections
        for i in prange(N):
            for j in prange(N):
                if (i != j) and (C[i, j] == C[j, i] == 1):
                    rand = np.random.randint(0, 1)
                    if rand == 0:
                        C[i, j] = 0
                    else:
                        C[j, i] = 0

        # Only allow a certain number of connections
        if isSequentialNetwork:
            for i in prange(numSubNetworks):
                subInputIdc = inputIdc[subNumOfInputNodes * i:
                                       subNumOfInputNodes * (i + 1)]

                subHiddenIdc = hiddenIdc[subNumOfHiddenNodes * i:
                                         subNumOfHiddenNodes * (i + 1)]

                subOutputIdc = outputIdc[subNumOfOutputNodes * i:
                                         subNumOfOutputNodes * (i + 1)]

                C[min(subInputIdc):max(subInputIdc),
                  min(subHiddenIdc):min(subHiddenIdc)] = 1.

                C[min(subInputIdc):max(subInputIdc),
                  min(subOutputIdc):max(subOutputIdc)] = 1.

                # Inter-connection with-in subnetwork
                # and connections that lead to the next subnetwork
                C[min(subHiddenIdc):max(subHiddenIdc),
                  min(subHiddenIdc):min(subHiddenIdc)] = 1.

                if i + 1 < numSubNetworks:
                    nextSubHiddenIdc = hiddenIdc[subNumOfInputNodes * (i + 1):
                                                 subNumOfOutputNodes * (i + 2)]

                    C[min(subHiddenIdc):max(subHiddenIdc),
                      min(nextSubHiddenIdc):min(nextSubHiddenIdc)] = 1.

                C[min(subHiddenIdc):max(subHiddenIdc),
                  min(subOutputIdc):max(subOutputIdc)] = 1.

                C = removeRandomConnections(inputIdc, hiddenIdc, outputIdc,
                                            inputPerNode // numSubNetworks,
                                            outputPerNode // numSubNetworks,
                                            C)
        else:
            C = removeRandomConnections(inputIdc, hiddenIdc, outputIdc,
                                        inputPerNode, outputPerNode, C)

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


@njit()
def fixSynapses(W, P, maxRadius, nodeIdc):
    for n0 in nodeIdc:
        for n1 in nodeIdc:
            if W[n0, n1] != 0 and magnitude(P[n0], P[n1]) > maxRadius:
                W[n0, n1] = 0

    return W


@njit(fastmath=True)
def magnitude(v0, v1):
    return np.sqrt((v0[0] - v1[0]) ** 2
                   + (v0[1] - v1[1]) ** 2
                   + (v0[2] - v1[2]) ** 2)


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
    return np.cos(x * PI/ 180.) * PI / 180.


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
def g(p, t, nOutputs):
    return (np.sum((p - t) ** 2, axis=0)) / nOutputs


@njit(fastmath=True)
def gPrime(p, t, nOutputs):
    return (2 / nOutputs) * (p - t)


@njit(parallel=True, fastmath=True)
def forward(W, I, O, P, R, nodeIdc,
            inputIdc, hiddenIdc, outputIdc,
            u, bias, f):
    for i in prange(len(inputIdc)):
        inputIdx = inputIdc[i]
        I[inputIdx] = u[inputIdx]
        O[inputIdx] = u[inputIdx]

    for i in prange(len(nodeIdc)):
        nodeIdx = nodeIdc[i]
        byIdc = np.where(W[:, nodeIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[nodeIdx], P[byIdx])

            I[nodeIdx] += (O[byIdx]
                           * W[byIdx, nodeIdx]
                           * R[byIdx, 0]
                           / vectorMagnitude)

        # if nodeIdx in hiddenIdc:
        #     O[nodeIdx] = f(I[nodeIdx] + bias)
        # elif nodeIdx in outputIdc:
        #     O[nodeIdx] = f(I[nodeIdx])
        O[nodeIdx] = f(I[nodeIdx] + bias)

    return I, O


@njit()
def getMagnitude(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@njit(parallel=True, fastmath=True)
def calculateGradients(target, W, I, O, P, R,
                       gradU, gradR, gradP,
                       nInputs, nHiddens, nOutputs,  # NOQA
                       hiddenIdc, outputIdc, fPrime,
                       lossWRTHiddenOutput):
    for i in prange(nOutputs):
        outputIdx = outputIdc[i]
        outIdx = outputIdx - (nInputs + nHiddens)  # NOQA  (Index of output array with shape (10,))
        byIdc = np.where(W[:, outputIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[outputIdx])

            if len(np.where(hiddenIdc == byIdx)[0]):
                lossWRTHiddenOutput[byIdx] += W[byIdx, outputIdx] \
                                              * gPrime(O[outputIdx],
                                                       target[outIdx],
                                                       nOutputs) \
                                              * fPrime(I[outputIdx]) \
                                              * R[byIdx, 0] \
                                              / vectorMagnitude

                if len(np.where(hiddenIdc == byIdx + 1)[0]):
                    lossWRTHiddenOutput[byIdx] += lossWRTHiddenOutput[byIdx + 1]

            gradU[byIdx, outputIdx] += O[byIdx] \
                                       * gPrime(O[outputIdx],
                                                target[outIdx],
                                                nOutputs) \
                                       * fPrime(I[outputIdx]) \
                                       * R[byIdx, 0] \
                                       / vectorMagnitude

            gradR[byIdx, 0] += O[byIdx] \
                               * gPrime(O[outputIdx],
                                        target[outIdx],
                                        nOutputs) \
                               * fPrime(I[outputIdx]) \
                               * W[byIdx, outputIdx] \
                               / vectorMagnitude

            # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
            x0 = P[outputIdx, 0]
            y0 = P[outputIdx, 1]
            z0 = P[outputIdx, 2]

            x1 = P[byIdx, 0]
            y1 = P[byIdx, 1]
            z1 = P[byIdx, 2]

            gradP[outputIdx, 0] += O[byIdx] \
                                   * gPrime(O[outputIdx],
                                            target[outIdx],
                                            nOutputs) \
                                   * fPrime(I[outputIdx]) \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0]
                                      * -(x0 - x1)
                                      / (vectorMagnitude ** 3))

            gradP[outputIdx, 1] += O[byIdx] \
                                   * gPrime(O[outputIdx],
                                            target[outIdx],
                                            nOutputs) \
                                   * fPrime(I[outputIdx]) \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0]
                                      * -(y0 - y1)
                                      / (vectorMagnitude ** 3))

            gradP[outputIdx, 2] += O[byIdx] \
                                   * gPrime(O[outputIdx],
                                            target[outIdx],
                                            nOutputs) \
                                   * fPrime(I[outputIdx]) \
                                   * W[byIdx, outputIdx] \
                                   * (R[byIdx, 0]
                                      * -(z0 - z1)
                                      / (vectorMagnitude ** 3))

            gradP[byIdx, 0] += O[byIdx] \
                               * gPrime(O[outputIdx],
                                        target[outIdx],
                                        nOutputs) \
                               * fPrime(I[outputIdx]) \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0]
                                  * (x0 - x1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 1] += O[byIdx] \
                               * gPrime(O[outputIdx],
                                        target[outIdx],
                                        nOutputs) \
                               * fPrime(I[outputIdx]) \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0]
                                  * (y0 - y1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 2] += O[byIdx] \
                               * gPrime(O[outputIdx],
                                        target[outIdx],
                                        nOutputs) \
                               * fPrime(I[outputIdx]) \
                               * W[byIdx, outputIdx] \
                               * (R[byIdx, 0]
                                  * (z0 - z1)
                                  / (vectorMagnitude ** 3))

    for i in prange(nHiddens):
        hiddenIdx = hiddenIdc[::-1][i]
        byIdc = np.where(W[:, hiddenIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[hiddenIdx])

            gradU[byIdx, hiddenIdx] += O[byIdx] \
                                       * lossWRTHiddenOutput[hiddenIdx] \
                                       * fPrime(I[hiddenIdx]) \
                                       * R[byIdx, 0] \
                                       / vectorMagnitude

            gradR[byIdx, 0] += O[byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx]) \
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
                                   * lossWRTHiddenOutput[hiddenIdx] \
                                   * fPrime(I[hiddenIdx]) \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0]
                                      * -(x0 - x1)
                                      / (vectorMagnitude ** 3))

            gradP[hiddenIdx, 1] += O[byIdx] \
                                   * lossWRTHiddenOutput[hiddenIdx] \
                                   * fPrime(I[hiddenIdx]) \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0]
                                      * -(y0 - y1)
                                      / (vectorMagnitude ** 3))

            gradP[hiddenIdx, 2] += O[byIdx] \
                                   * lossWRTHiddenOutput[hiddenIdx] \
                                   * fPrime(I[hiddenIdx]) \
                                   * W[byIdx, hiddenIdx] \
                                   * (R[byIdx, 0]
                                      * -(z0 - z1)
                                      / (vectorMagnitude ** 3))

            gradP[byIdx, 0] += O[byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx]) \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0]
                                  * (x0 - x1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 1] += O[byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx]) \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0]
                                  * (y0 - y1)
                                  / (vectorMagnitude ** 3))

            gradP[byIdx, 2] += O[byIdx] \
                               * lossWRTHiddenOutput[hiddenIdx] \
                               * fPrime(I[hiddenIdx]) \
                               * W[byIdx, hiddenIdx] \
                               * (R[byIdx, 0]
                                  * (z0 - z1)
                                  / (vectorMagnitude ** 3))

    return gradU, gradR, gradP
