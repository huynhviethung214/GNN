import numpy as np
import torch

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


@njit()
def magnitudeNumpy(v0, v1):
    return np.sqrt((v0[0] - v1[0]) ** 2
                   + (v0[1] - v1[1]) ** 2
                   + (v0[2] - v1[2]) ** 2)


def magnitude(v0, v1):
    a = torch.sqrt(torch.sum((v0 - v1) ** 2, dim=1))
    a[a == 0.] = 1.
    return a.view(1, -1)


# Activation Function
def sigmoid(x):
    return 1. / (1. + torch.exp(-x))  # Sigmoid


# Derivative of activation function
def sigmoidPrime(x):
    return (sigmoid(x) * (1. - sigmoid(x))).view(1, -1)  # Derivative of Sigmoid


# Loss Function
def g(p, t, nOutputs):
    return (torch.sum((p - t) ** 2)) / nOutputs


def gPrime(p, t, nOutputs):
    return torch.sum((2 / nOutputs) * (p - t), dim=0).view(1, -1)


@njit()
def removeNode():
    return


@njit()
def removeConnection(W, i, j):
    W[i, j] = 0
    return W


def forward(W, I, O, P, R, nodeIdc,
            inputIdc, hiddenIdc, outputIdc, us, bias, f):
    for inputIdx in inputIdc:
        I[:, inputIdx] = us[:, inputIdx]
        O[:, inputIdx] = us[:, inputIdx]

    for nodeIdx in nodeIdc:
        vectorMagnitude = magnitude(P[nodeIdx], P)

        I[:, nodeIdx] += torch.sum(
            (O
             * W[:, nodeIdx].view(1, -1)
             * R.view(1, -1)
             / vectorMagnitude),
            dim=1
        )

        if nodeIdx in hiddenIdc:
            O[:, nodeIdx] = f(I[:, nodeIdx] + bias)
        elif nodeIdx in outputIdc:
            O[:, nodeIdx] = f(I[:, nodeIdx])

    # return I, O


@njit()
def getMagnitude(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def calculateGradients(predicts, vs, W, I, O, P, R,
                       gradU, gradR, gradP, batchSize,
                       nInputs, nHiddens, nOutputs,  # NOQA
                       hiddenIdc, outputIdc, fPrime, lossWRTHiddenOutput,
                       C, Ch):
    for outputIdx in outputIdc:
        outIdx = outputIdx - (nInputs + nHiddens)
        vectorMagnitude = magnitude(P[outputIdx], P)

        lossWRTHiddenOutput += (gPrime(predicts[:, outIdx],
                                       vs[:, outIdx],
                                       nOutputs)
                                * fPrime(I[:, outputIdx])).view((-1, 1)) @ \
                               ((W[:, outputIdx].view(1, -1)
                                 * Ch.view(1, -1)).view(1, -1)
                                * R.view(1, -1)
                                / vectorMagnitude)

        gradU[:, :, outputIdx] += torch.sum(
            (O
             * C[:, outputIdx].view(1, -1)
             * R.view(1, -1)
             / vectorMagnitude).view((batchSize, -1, 1)) \
            @ (gPrime(predicts[:, outIdx],
                      vs[:, outIdx],
                      nOutputs)
               * fPrime(I[:, outputIdx])),
            dim=2
        )

        gradR += torch.sum(
            (O
             * W[:, outputIdx].view(1, -1)
             / vectorMagnitude).view((batchSize, -1, 1)) \
            @ (gPrime(predicts[:, outIdx],
                      vs[:, outIdx],
                      nOutputs)
               * fPrime(I[:, outputIdx])),
            dim=2,
            keepdim=True
        )

        gradP[:, outputIdx, :] += torch.sum(
            (((O
               * W[:, outputIdx].view(1, -1)
               * R.view(1, -1)
               / (vectorMagnitude ** 3))).view((batchSize, -1, 1))
             @ (gPrime(predicts[:, outIdx],
                       vs[:, outIdx],
                       nOutputs)
                * fPrime(I[:, outputIdx]))),
            dim=2
        ) @ -(P[outputIdx] - P).view(P.shape[0], 3)

        # gradP += torch.sum(
        #     (((O
        #        * W[:, outputIdx].view(1, -1)
        #        * R.view(1, -1)
        #        / (vectorMagnitude ** 3))).view((batchSize, -1, 1))
        #      @ (gPrime(predicts[:, outIdx],
        #                vs[:, outIdx],
        #                nOutputs)
        #         * fPrime(I[:, outputIdx]))),
        #     dim=2,
        #     keepdim=True
        # ) * (C[:, outputIdx].view(-1, 1)
        #      * (P[outputIdx] - P).view(P.shape[0], 3)).view(1, -1, 3)

    for hiddenIdx in hiddenIdc[::-1]:
        if hiddenIdx + 1 in hiddenIdc:
            lossWRTHiddenOutput[:, hiddenIdx] += lossWRTHiddenOutput[:, hiddenIdx + 1]
        vectorMagnitude = magnitude(P[hiddenIdx], P)

        gradU[:, :, hiddenIdx] += torch.sum(
            (O
             * C[:, hiddenIdx].view(1, -1)
             * R.view(1, -1)
             / vectorMagnitude).view((batchSize, -1, 1)) \
            @ (lossWRTHiddenOutput[:, hiddenIdx]
               * fPrime(I[:, hiddenIdx])),
            dim=2
        )

        gradR += torch.sum(
            (O
             * W[:, hiddenIdx].view(1, -1)
             / vectorMagnitude).view((batchSize, -1, 1)) \
            @ (lossWRTHiddenOutput[:, hiddenIdx]
               * fPrime(I[:, hiddenIdx])),
            dim=2,
            keepdim=True
        )

        gradP[:, hiddenIdx, :] += torch.sum(
            (((O
               * W[:, hiddenIdx].view(1, -1)
               * R.view(1, -1)
               / (vectorMagnitude ** 3))).view((batchSize, -1, 1))
             @ (lossWRTHiddenOutput[:, hiddenIdx]
                * fPrime(I[:, hiddenIdx]))),
            dim=2
        ) @ -(P[hiddenIdx] - P).view(P.shape[0], 3)

        # gradP += torch.sum(
        #     (((O
        #        * W[:, hiddenIdx].view(1, -1)
        #        * R.view(1, -1)
        #        / (vectorMagnitude ** 3))).view((batchSize, -1, 1))
        #      @ (lossWRTHiddenOutput[:, hiddenIdx]
        #         * fPrime(I[:, hiddenIdx]))),
        #     dim=2,
        #     keepdim=True
        # ) * (C[:, hiddenIdx].view(-1, 1)
        #      * (P[hiddenIdx] - P).view(P.shape[0], 3)).view(1, -1, 3)
