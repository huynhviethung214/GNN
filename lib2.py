import numpy as np

from numba import njit


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

    W = np.random.uniform(-1 / np.sqrt(nInputs), 1 / np.sqrt(nInputs), (N, N))  # Xavier's Weight Initialization
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
def isEnoughInput(connectedNodesIdx, nodeState):
    for nodeIdx in connectedNodesIdx:
        if not nodeState[nodeIdx, nodeIdx]:
            return False
    return True


@njit()
def crossOver(crossOverRatio, N, W1, C1, W2, C2):
    # sliceIdx = int(crossOverRatio * N)
    #
    # _W1 = np.vstack((W1[sliceIdx:], W2[:sliceIdx]))
    # _W2 = np.vstack((W2[sliceIdx:], W1[:sliceIdx]))
    #
    # _C1 = np.vstack((C1[sliceIdx:], C2[:sliceIdx]))
    # _C2 = np.vstack((C2[sliceIdx:], C1[:sliceIdx]))

    numberOfCrossOver = int(crossOverRatio * N)
    _W1 = np.asarray(W1)
    _W2 = np.asarray(W2)

    _C1 = np.asarray(C1)
    _C2 = np.asarray(C2)

    for _ in range(numberOfCrossOver):
        randomIdx = np.random.randint(0, N - 1, 1)

        temp = _W1[randomIdx]
        _W1[randomIdx] = _W2[randomIdx]
        _W2[randomIdx] = temp

        temp = _C1[randomIdx]
        _C1[randomIdx] = _C2[randomIdx]
        _C2[randomIdx] = temp

    return _W1, _C1, _W2, _C2


# Inverting a random amount of paths (in term of row(s))
@njit()
def invertingPaths(W, C, nodeIdx, numInverting, nInputs, N):
    selectedCols = np.random.randint(nInputs, N - 1, numInverting)

    for selectedCol in selectedCols:
        C[nodeIdx, selectedCol] = int(not C[nodeIdx, selectedCol])
        W[nodeIdx, selectedCol] = int(not W[nodeIdx, selectedCol])

    return W, C


# @njit()
# def sortHiddenNodes(hiddenIdc, C):
#     q0 = np.zeros((1,), dtype=np.int64)
#     q1 = np.zeros((1,), dtype=np.int64)
#
#     for hiddenIdx in hiddenIdc:
#         connectedNodesIdx = np.where(C[:, hiddenIdx] == 1)[0]
#
#         for nodeIdx in connectedNodesIdx:
#             if nodeIdx in hiddenIdc:
#                 q0 = np.append(q0, hiddenIdx)
#             else:
#                 q1 = np.append(q1, hiddenIdx)
#
#     q0 = np.delete(q0, 0)
#     q1 = np.delete(q1, 0)
#
#     mergedQ = np.hstack((q1, q0))
#
#     return mergedQ


@njit()
def forward(W, C, O, I, inputIdc, hiddenIdc, outputIdc, u, bias):
    # Input Layer
    for i, inputIdx in enumerate(inputIdc):
        O[inputIdx, inputIdx] = u[i]
        I[inputIdx, inputIdx] = u[i]

    # Hidden Layer
    for hiddenIdx in hiddenIdc:
        rowIdc = np.where(C[:, hiddenIdx] == 1)[0]

        for rowIdx in rowIdc:
            I[hiddenIdx, hiddenIdx] = I[hiddenIdx, hiddenIdx] + (O[rowIdx, rowIdx] * W[rowIdx, hiddenIdx])
        O[hiddenIdx, hiddenIdx] = f(I[hiddenIdx, hiddenIdx] + bias)

    # Output Layer
    for outputIdx in outputIdc:
        rowIdc = np.where(C[:, outputIdx] == 1)[0]

        for rowIdx in rowIdc:
            I[outputIdx, outputIdx] = I[outputIdx, outputIdx] + (O[rowIdx, rowIdx] * W[rowIdx, outputIdx])
        O[outputIdx, outputIdx] = relu(I[outputIdx, outputIdx])

    return W, C, O, I


@njit()
def backward(W, C, O, I, U, nInputs, nHiddens, hiddenIdc, outputIdc, predict, v):  # NOQA
    for outputIdx in outputIdc:
        connectedNodesIdx = np.where(C[:, outputIdx] == 1)[0]

        for connectedNodeIdx in connectedNodesIdx:
            outIdx = outputIdx - (nInputs + nHiddens)  # NOQA  (Index of output array of shape (1, 10))
            U[connectedNodeIdx, outputIdx] = O[connectedNodeIdx, connectedNodeIdx] \
                                             * gPrime(predict[0, outIdx], v[0, outIdx]) \
                                             * drelu(I[outputIdx, outputIdx])

    for hiddenIdx in hiddenIdc:
        connectedNodesIdx = np.where(C[:, hiddenIdx] == 1)[0]

        for connectedNodeIdx in connectedNodesIdx:
            U[connectedNodeIdx, hiddenIdx] = O[connectedNodeIdx, hiddenIdx] * fPrime(
                I[connectedNodeIdx, connectedNodeIdx])

    return U


# Activation Function
@njit()
def f(x):
    return 1. / (1. + np.exp(-x))  # Sigmoid


# Derivative of activation function
@njit()
def fPrime(x):
    return f(x) * (1. - f(x))  # Derivative of Sigmoid


@njit()
def relu(x):
    return np.maximum(0., x)


@njit()
def drelu(x):
    if x <= 0.:
        return 0.
    return 1.


# Loss Function
# y_hat = output vector, y = target
@njit()
def g(p, y):
    # return np.square(y_hat - y) / 2
    celoss = 0

    for i in range(len(p[0, :])):
        if p[0, i] > 0:
            celoss += y[0, i] * np.log(p[0, i])
        else:
            celoss += y[0, i] * np.log(1e-15)

    return -celoss


@njit()
def gPrime(p, y):
    return p - y


@njit()
def isInPath(x, path):
    for e in path:
        if e == x:
            return True
    return False
