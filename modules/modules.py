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


@njit(fastmath=True, parallel=True)
def initialize(inputPerNode, outputPerNode,
               N, Nin, Nout,
               inputIdc, hiddenIdc, outputIdc):  # NOQA
    # Let `i` denoted as a row index and `j` denoted as a column index
    # Node(s) is said to be connecting to other Node(s) has to have the following condition: C[i, j] = 1
    # Node(s) is said to be connected by other Node(s) has to have the following condition: C[j, i] = 1
    # * Diagonal will always be 0
    # * Input node column's elements are always equal to 0
    # * Output node row's elements are always equal to 0
    # * We will use C[i, j] to find the next node(s) in forwarding operation (C[i, j] == 1)
    #           and C[j, i] to find the next node(s) in backpropagation operation (C[j, i] == 1)
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
    # W = np.random.uniform(0, 1, (N, N))
    W = np.random.uniform(-np.sqrt(6 / (Nin + Nout)),
                          +np.sqrt(6 / (Nin + Nout)), (N, N))  # Xavier's Weight Initialization
    # W = np.random.uniform(0, np.sqrt(6 / (Nin + Nout)), (N, N))

    Nh = N - Nin - Nout

    # Initialize the connection matrix
    while True:
        C = np.ones((N, N))  # Connections Matrix

        # turn off connection(s) that is violating the constraint (see above)
        # np.fill_diagonal(C[:Nin, :Nin], 0)
        # np.fill_diagonal(C[Nin + Nh:, Nin + Nh:], 0)
        np.fill_diagonal(C, 0.)

        for i in outputIdc:
            C[i, :] = 0

        for i in inputIdc:
            C[:, i] = 0

        # Fix two-way connections
        # for i in range(N):
        #     for j in range(N):
        #         # if (i not in hiddenIdc or j not in hiddenIdc) and (i != j):
        #         if (i != j) and (C[i, j] == C[j, i] == 1):
        #             rand = np.random.uniform(0, 1, 1)[0]
        #             if rand <= 0:
        #                 C[i, j] = 0
        #             else:
        #                 C[j, i] = 0

        C = removeRandomConnections(inputIdc, hiddenIdc, outputIdc,
                                    inputPerNode, outputPerNode, C)

        # If path exist then break the loop
        if pathExist(C, list(inputIdc)):
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
def synapticPruning(W, P, R, nodeIdc, beta: float = 1e-6):
    for x in nodeIdc:
        toIdc = np.where(W[x] != 0.)[0]
        for toIdx in toIdc:
            if x != toIdx:
                if gaussianKernel(magnitude(P[x], P[toIdx]), R[x, 0]) <= beta:
                    W[x, toIdx] = 0.


@njit(fastmath=True)
def magnitude(u, v):
    return np.sqrt((u[0] - v[0]) ** 2 +
                   (u[1] - v[1]) ** 2 +
                   (u[2] - v[2]) ** 2)


# Activation Function
@njit(fastmath=True)
def sigmoid(x):
    # return 1. / (1. * np.exp(-np.power((2 * np.sin(x)), 3)))
    return 1. / (1. + np.exp(-2 * np.sin(x)))
    # return 1. / (1. + np.exp(-1 * x))  # Sigmoid


# Derivative of activation function
@njit(fastmath=True)
def sigmoidPrime(x):
    # return (12 * np.sin(2 * x) * np.sin(x)) * sigmoid(x) * (1 - sigmoid(x))
    return (2 * np.cos(x)) * sigmoid(x) * (1 - sigmoid(x))
    # return 1 * sigmoid(x) * (1. - sigmoid(x))  # Derivative of Sigmoid


@njit(fastmath=True)
def butterworth(x):
    return 1. / (1. + np.power(np.sin(x) + 1, 4))


@njit(fastmath=True)
def butterworthPrime(x):
    return (-4. * np.power(np.sin(x) + 1, 3) * np.cos(x)) * np.power(butterworth(x), 2)


@njit(fastmath=True)
def sin(x):
    return np.sin(x)


# Derivative of activation function
@njit(fastmath=True)
def cos(x):
    return np.cos(x)


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
def mse(p, t, Nout):
    return np.sum((t - p) ** 2, axis=0) / Nout


@njit(fastmath=True)
def msePrime(p, t, Nout):
    return (2 / Nout) * (p - t)  # (p - t) 'cuz we take the partial derivative of (p - t)^2


@njit(fastmath=True)
def crossEntropy(p, t, *args):
    summation = 0
    for i in range(p.shape[0]):
        if t == 0.:
            summation += np.log(1 - p)
        else:
            summation += np.log(p)

    return (-1 / p.shape[0]) * summation


@njit(fastmath=True)
def crossEntropyPrime(p, t, *args):
    return -((t / p) - ((1 - t) / (1 - p)))


@njit(fastmath=True)
def mae(p, t, *args):
    return np.sum(np.abs(p - t)) / p.shape[0]


@njit(fastmath=True)
def maePrime(p, t, *args):
    if p > t:
        return 1
    return -1


@njit(fastmath=True)
def gaussianKernel(dst, radius):
    return np.exp(-dst / (2 * (radius ** 2)))


@njit(fastmath=True)
def dervOfGaussianKernelWRTPosition(dst, signalStrength):
    return (1 / (2 * dst)) * signalStrength


@njit(fastmath=True)
def saveMemory(O, Oh, Nin, Nh, numberOfStates, stateFlags):
    for stateIdx in range(numberOfStates - 1, 0, -1):
        if not stateFlags[stateIdx]:
            Oh[stateIdx] = O[Nin: Nin + Nh]
            stateFlags[stateIdx] = 1.
            break


# Forward With ToIdc
# @njit(fastmath=True, parallel=True)
# def forwardInputToHidden(u, Nin, hiddenIdc, outputIdc, W, C, D,
#                          B, P, R, I, O, f, enableInputActivation):
#     for inputIdx in prange(Nin):
#         I[inputIdx] += u[inputIdx]
#         if enableInputActivation:
#             O[inputIdx] += f((I[inputIdx] / D[inputIdx]) + B[inputIdx])
#         else:
#             O[inputIdx] += u[inputIdx]
#
#     sortedInputIdc = np.argsort(O[:Nin])
#     for i in prange(Nin):
#         inputIdx = sortedInputIdc[i]
#         for j in prange(len(hiddenIdc)):
#             hiddenIdx = hiddenIdc[j]
#             if W[inputIdx, hiddenIdx] != 0.:
#                 vectorMagnitude = magnitude(P[inputIdx], P[hiddenIdx])
#
#                 if vectorMagnitude == 0.:
#                     W[inputIdx, hiddenIdx] *= 0.
#                     C[inputIdx, hiddenIdx] *= 0.
#                     continue
#
#                 throughput = R[inputIdx, 0] / vectorMagnitude
#
#                 if throughput > 1.:
#                     throughput = 1.
#
#                 I[hiddenIdx] += O[inputIdx] * W[inputIdx, hiddenIdx] * throughput
#
#         for j in prange(len(outputIdc)):
#             outputIdx = outputIdc[j]
#             if W[inputIdx, outputIdx] != 0.:
#                 vectorMagnitude = magnitude(P[inputIdx], P[outputIdx])
#
#                 if vectorMagnitude == 0.:
#                     W[inputIdx, outputIdx] *= 0.
#                     C[inputIdx, outputIdx] *= 0.
#                     continue
#
#                 throughput = R[inputIdx, 0] / vectorMagnitude
#
#                 if throughput > 1.:
#                     throughput = 1.
#
#                 I[outputIdx] += O[inputIdx] * W[inputIdx, outputIdx] * throughput
#
#
# @njit(fastmath=True, parallel=True)
# def forwardModelHidden(hiddenIdc, outputIdc, Nin, Nh, W, C, D, B, P, R, I, O, f):
#     sortedHiddenIdc = np.argsort(O[Nin: Nin + Nh])
#     for i in prange(Nh):
#         hiddenIdx0 = sortedHiddenIdc[i]
#         O[hiddenIdx0] += f((I[hiddenIdx0] / D[hiddenIdx0]) + B[hiddenIdx0])
#
#         for j in prange(Nh):
#             if i != j:
#                 hiddenIdx1 = sortedHiddenIdc[j]
#                 vectorMagnitude = magnitude(P[hiddenIdx0], P[hiddenIdx1])
#
#                 if vectorMagnitude == 0.:
#                     W[hiddenIdx1, hiddenIdx0] *= 0.
#                     C[hiddenIdx1, hiddenIdx0] *= 0.
#                     W[hiddenIdx0, hiddenIdx1] *= 0.
#                     C[hiddenIdx0, hiddenIdx1] *= 0.
#                     continue
#
#                 throughput = R[hiddenIdx0, 0] / vectorMagnitude
#
#                 if throughput > 1.:
#                     throughput = 1.
#
#                 I[hiddenIdx1] += O[hiddenIdx0] * W[hiddenIdx0, hiddenIdx1] * throughput
#
#         for j in prange(len(outputIdc)):
#             outputIdx = outputIdc[j]
#             vectorMagnitude = magnitude(P[hiddenIdx0], P[outputIdx])
#
#             if vectorMagnitude == 0.:
#                 W[hiddenIdx0, outputIdx] *= 0.
#                 C[hiddenIdx0, outputIdx] *= 0.
#                 W[outputIdx, hiddenIdx0] *= 0.
#                 C[outputIdx, hiddenIdx0] *= 0.
#                 continue
#
#             throughput = R[hiddenIdx0, 0] / vectorMagnitude
#
#             if throughput > 1.:
#                 throughput = 1.
#
#             I[outputIdx] += O[hiddenIdx0] * W[hiddenIdx0, outputIdx] * throughput
#
#
# @njit(fastmath=True, parallel=True)
# def forwardModelOutput(outputIdc, D, B, I, O, f):
#     for i in prange(len(outputIdc)):
#         outputIdx = outputIdc[i]
#         O[outputIdx] += f((I[outputIdx] / D[outputIdx]) + B[outputIdx])


# Forward with ByIdc
@njit(parallel=True, fastmath=True)
def forwardInput(D, I, O, B,
                 inputIdc, u, f,
                 enableInputActivation):
    if enableInputActivation:
        for i in prange(len(inputIdc)):
            inputIdx = inputIdc[i]
            if u[inputIdx] != 0.:
                I[inputIdx] += u[inputIdx]
                O[inputIdx] += f((I[inputIdx] / D[inputIdx]) + B[inputIdx])
    else:
        for i in prange(len(inputIdc)):
            inputIdx = inputIdc[i]
            if u[inputIdx] != 0.:
                I[inputIdx] += u[inputIdx]
                O[inputIdx] += u[inputIdx]


@njit(fastmath=True, parallel=True)
def forwardHidden(Nh, W, C, D, B, P, R, I, O, hiddenIdc, f):
    for i in prange(Nh):
        hiddenIdx = hiddenIdc[i]
        byIdc = np.where(W[:, hiddenIdx] != 0.)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            if O[byIdx] != 0.:
                vectorMagnitude = magnitude(P[hiddenIdx], P[byIdx])

                if vectorMagnitude == 0.:
                    W[byIdx, hiddenIdx] *= 0.
                    W[hiddenIdx, byIdx] *= 0.
                    C[byIdx, hiddenIdx] *= 0.
                    C[hiddenIdx, byIdx] *= 0.
                    continue

                throughput = gaussianKernel(vectorMagnitude, R[byIdx, 0])
                I[hiddenIdx] += O[byIdx] * W[byIdx, hiddenIdx] * throughput

        O[hiddenIdx] += f((I[hiddenIdx] / D[hiddenIdx]) + B[hiddenIdx])


@njit(fastmath=True, parallel=True)
def forwardOutput(Nout, outputIdc, W, C, D, B, P, R, I, O, f):
    for i in prange(Nout):
        outputIdx = outputIdc[i]
        byIdc = np.where(W[:, outputIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            if O[byIdx] != 0.:
                vectorMagnitude = magnitude(P[outputIdx], P[byIdx])

                if vectorMagnitude == 0.:
                    W[byIdx, outputIdx] = 0.
                    W[outputIdx, byIdx] = 0.
                    C[byIdx, outputIdx] = 0.
                    C[outputIdx, byIdx] = 0.
                    continue

                throughput = gaussianKernel(vectorMagnitude, R[byIdx, 0])

                I[outputIdx] += O[byIdx] * W[byIdx, outputIdx] * throughput

        O[outputIdx] += f((I[outputIdx] / D[outputIdx]) + B[outputIdx])


@njit(fastmath=True, parallel=True)
def recallMemory(stateIdx, byStateIdx, Ih, Oh, Wh, R, Nin, Nh, Nout,
                 hiddenIdc, outputIdc, stateFlags, numberOfStates):
    for i in prange(Nh):
        hiddenIdx = i + Nin
        byIdc = np.where(Wh[:, hiddenIdx] != 0.)[0]
        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            throughput = gaussianKernel(1, R[byIdx, 0])

            Ih[stateIdx, hiddenIdx] \
                += Oh[byStateIdx, byIdx] * Wh[byIdx, hiddenIdx] * throughput

    for i in prange(Nout):
        outputIdx = i + Nin + Nh
        byIdc = np.where(Wh[:, outputIdx] != 0.)[0]
        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            throughput = gaussianKernel(1, R[byIdx, 0])

            Ih[stateIdx, outputIdx] \
                += Oh[byStateIdx, byIdx] * Wh[byIdx, outputIdx] * throughput


@njit(fastmath=True, parallel=True)
def calculateGradientBetweenStates(gradWh, gradR, gradB, W, D, B, Wh, Ih, Oh, R,
                                   nodeIdc, N, Nin, Nh, stateIdx, byStateIdx,
                                   lossWRTHiddenOutput, fPrime):
    for i in prange(N):
        nodeIdx = nodeIdc[i]
        byIdc = np.where(Wh[:, nodeIdx] != 0.)[0]
        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            throughput = gaussianKernel(1, R[byIdx, 0])

            lossWRTHiddenOutput[byStateIdx, byIdx] \
                += gradB[nodeIdx] * Wh[byIdx, nodeIdx] * throughput

            gradBOh = gradB[nodeIdx] * Oh[byStateIdx, byIdx]
            gradWh[byIdx, nodeIdx] += gradBOh * throughput
            gradR[byIdx, 0] += gradBOh * Wh[byIdx, nodeIdx]


@njit(parallel=True, fastmath=True)
def calculateGradients(target, W, D, C, B, P, R, I, Ih, O, Oh,
                       gradW, gradD, gradR, gradP, gradB,
                       Nin, Nh, Nout, inputIdc, hiddenIdc,
                       outputIdc, f, fPrime, gPrime, lossWRTInputOutput,
                       lossWRTHiddenOutput, lossWRTOutputOutput, alpha, locks,
                       enableInputActivation, enableLoss, stateIdx):
    for i in prange(Nout):
        outputIdx = outputIdc[i]
        outIdx = outputIdx - (Nin + Nh)

        if np.abs(Oh[stateIdx, outputIdx] - target[outIdx]) > alpha:
            byIdc = np.where(W[:, outputIdx] != 0.)[0]

            # if enableLoss:
            gradB[outputIdx] = (gPrime(Oh[stateIdx, outputIdx], target[outIdx], Nout)
                                * fPrime((Ih[stateIdx, outputIdx]
                                          / D[outputIdx])
                                         + B[outputIdx]))
            # else:
            #     gradB[outputIdx] = lossWRTOutputOutput[outputIdx - Nin - Nh] \
            #                        * fPrime((I[outputIdx] / D[outputIdx]) + B[outputIdx])

            gradD[outputIdx] = gradB[outputIdx] \
                               * (-Ih[stateIdx, outputIdx]
                                  / np.power(D[outputIdx], 2))

            for j in prange(len(byIdc)):
                byIdx = byIdc[j]
                gradBW = W[byIdx, outputIdx] * gradB[outputIdx]
                vectorMagnitude = magnitude(P[outputIdx], P[byIdx])

                if vectorMagnitude == 0.:
                    W[byIdx, outputIdx] = 0.
                    W[outputIdx, byIdx] = 0.
                    C[byIdx, outputIdx] = 0.
                    C[outputIdx, byIdx] = 0.
                    continue

                throughput = gaussianKernel(vectorMagnitude, R[byIdx, 0])

                if len(np.where(hiddenIdc == byIdx)[0]):
                    lossWRTHiddenOutput[byIdx - Nin] \
                        += gradBW * throughput

                    if len(np.where(hiddenIdc == byIdx + 1)[0]) and W[byIdx, byIdx + 1] != 0.:
                        lossWRTHiddenOutput[byIdx - Nin] += \
                            lossWRTHiddenOutput[byIdx - Nin + 1]

                if enableInputActivation:
                    if len(np.where(inputIdc == byIdx)[0]):
                        lossWRTInputOutput[byIdx] += gradBW * throughput

                if Oh[stateIdx, byIdx] != 0.:
                    gradBO = Oh[stateIdx, byIdx] * gradB[outputIdx]
                    A = (Oh[stateIdx, byIdx] * gradBW
                         * dervOfGaussianKernelWRTPosition(vectorMagnitude,
                                                           throughput))

                    gradW[byIdx, outputIdx] += gradBO * throughput
                    gradR[byIdx, 0] += Oh[stateIdx, byIdx] * gradBW / vectorMagnitude

                    # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
                    x0 = P[outputIdx, 0]
                    y0 = P[outputIdx, 1]
                    z0 = P[outputIdx, 2]

                    x1 = P[byIdx, 0]
                    y1 = P[byIdx, 1]
                    z1 = P[byIdx, 2]

                    p0 = x0 - x1
                    p1 = y0 - y1
                    p2 = z0 - z1

                    if not locks[byIdx]:
                        gradP[byIdx, 0] += A * p0
                        gradP[byIdx, 1] += A * p1
                        gradP[byIdx, 2] += A * p2

                    gradP[outputIdx, 0] += A * -p0
                    gradP[outputIdx, 1] += A * -p1
                    gradP[outputIdx, 2] += A * -p2

    for i in prange(Nh):
        hiddenIdx = hiddenIdc[i]
        byIdc = np.where(W[:, hiddenIdx] != 0.)[0]

        gradB[hiddenIdx] = (lossWRTHiddenOutput[hiddenIdx - Nin]
                            * fPrime((Ih[stateIdx, hiddenIdx]
                                      / D[hiddenIdx])
                                     + B[hiddenIdx]))

        gradD[hiddenIdx] = gradB[hiddenIdx] \
                           * (-Ih[stateIdx, hiddenIdx] / np.power(D[hiddenIdx], 2))

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            gradBW = gradB[hiddenIdx] * W[byIdx, hiddenIdx]
            vectorMagnitude = magnitude(P[hiddenIdx], P[byIdx])

            if vectorMagnitude == 0.:
                W[byIdx, hiddenIdx] = 0.
                W[hiddenIdx, byIdx] = 0.
                C[byIdx, hiddenIdx] = 0.
                C[hiddenIdx, byIdx] = 0.
                continue

            throughput = gaussianKernel(vectorMagnitude, R[byIdx, 0])

            if enableInputActivation:
                if len(np.where(inputIdc == byIdx)[0]):
                    lossWRTInputOutput[byIdx] += gradBW * throughput

            if Oh[stateIdx, byIdx] != 0.:
                gradBO = gradB[hiddenIdx] * Oh[stateIdx, byIdx]

                A = (Oh[stateIdx, byIdx] * gradBW
                     * dervOfGaussianKernelWRTPosition(vectorMagnitude,
                                                       throughput))

                gradW[byIdx, hiddenIdx] += gradBO * throughput
                gradR[byIdx, 0] += Oh[stateIdx, byIdx] * gradBW / vectorMagnitude

                # Loss w.r.t position of outputNode (0 <=> x0 coordinate, 1 <=> y0 coordinate)
                x0 = P[hiddenIdx, 0]
                y0 = P[hiddenIdx, 1]
                z0 = P[hiddenIdx, 2]

                x1 = P[byIdx, 0]
                y1 = P[byIdx, 1]
                z1 = P[byIdx, 2]

                p0 = x0 - x1
                p1 = y0 - y1
                p2 = z0 - z1

                if not locks[byIdx]:
                    gradP[byIdx, 0] += A * p0
                    gradP[byIdx, 1] += A * p1
                    gradP[byIdx, 2] += A * p2

                if not locks[hiddenIdx]:
                    gradP[hiddenIdx, 0] += A * -p0
                    gradP[hiddenIdx, 1] += A * -p1
                    gradP[hiddenIdx, 2] += A * -p2

    if enableInputActivation:
        for inputIdx in prange(Nin):
            gradB[inputIdx] = (lossWRTInputOutput[inputIdx]
                               * fPrime((Ih[stateIdx, inputIdx]
                                         / D[inputIdx])
                                        + B[inputIdx]))

            gradD[inputIdx] = gradB[inputIdx] \
                              * (-Ih[stateIdx, inputIdx]
                                 / np.power(D[inputIdx], 2))
