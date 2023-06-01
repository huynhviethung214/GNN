import torch
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

    W = np.random.uniform(-N / np.sqrt(nInputs + nOutputs),
                          N / np.sqrt(nInputs + nOutputs), (N, N))  # Xavier's Weight Initialization
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


# Activation Function
def f(x):
    return 1. / (1. + torch.exp(-x))  # Sigmoid


# Derivative of activation function
def fPrime(x):
    return f(x) * (1. - f(x))  # Derivative of Sigmoid


def relu(x):
    return torch.where(x < 0., 0., x)
    # return f(x)


def drelu(x):
    return torch.where(x <= 0., 0., 1.)
    # return fPrime(x)


# Loss Function
def g(p, t, N):
    return (1 / N) * torch.sum((p - t) ** 2, dim=1)

    # N = p.size(0)
    # CELoss = torch.sum(torch.sum(t * torch.log(p + 1e-9), dim=1), dim=0) / N
    # return -CELoss


def gPrime(p, t):
    return (2 / 10) * (p - t)
    # return (p - t) / ((1 - p) * p)
    # dMSE = (2 / p.size(0)) * torch.sum((p - t) * p, dim=0)
    # return dMSE / p.size(0)


def fixInvalid(W, R, minRadius, maxRadius, nodeIdc):
    # Remove invalid connection(s)
    repairMatrix = np.asarray(R.cpu().numpy())
    repairMatrix[np.where(repairMatrix > maxRadius)[0], 0] = 0
    repairMatrix[np.where(repairMatrix < minRadius)[0], 0] = 0

    repairMatrix[np.where(repairMatrix >= minRadius)[0], 0] = 1
    repairMatrix[np.where(repairMatrix <= maxRadius)[0], 0] = 1

    return W * torch.from_numpy(repairMatrix).to(R.get_device())


def magnitude(uMv):
    # mags = torch.sqrt(torch.sum(torch.square((u - v) + 1e-15), dim=1))
    mags = torch.sqrt(torch.sum(torch.square(uMv), dim=1))
    mags[mags == 0.] = 1.
    return mags


def forward(W, I, O, P, R,
            hiddenIdc, outputIdc, bias):
    # for inputIdx in inputIdc:
    #     O[:, inputIdx] = us[:, inputIdx]

    # assert not torch.isnan(O).any(), 'Output at forward Input layer'

    for hiddenIdx in hiddenIdc:
        uMv = P[hiddenIdx] - P
        I[:, hiddenIdx] = ((O * (R / magnitude(uMv).view(-1, 1)).view(1, -1))
                           @ W[:, hiddenIdx])
        O[:, hiddenIdx] = f(I[:, hiddenIdx] + bias)

    # assert not torch.isnan(O).any(), 'Output at forward Hidden layer'

    for outputIdx in outputIdc:
        uMv = P[outputIdx] - P
        I[:, outputIdx] = ((O * (R / magnitude(uMv).view(-1, 1)).view(1, -1))
                           @ W[:, outputIdx])
        O[:, outputIdx] = f(I[:, outputIdx])

    # assert not torch.isnan(O).any(), 'Output at forward Output layer'

    return O, I


def gradients(predicts, targets, C,
              W, I, O, P, R,
              gradU, gradR, gradP,
              nInputs, nHiddens, nOutputs,  # NOQA
              hiddenIdc, outputIdc, lossWRTHiddenOutput):
    for outputIdx in outputIdc:
        outIdx = outputIdx - (nInputs + nHiddens)  # NOQA  (Index of output array with shape (10,))
        uMv = P[outputIdx] - P
        vectorMagnitude = magnitude(uMv).view(-1, 1)

        # inHiddenIndices = torch.where(torch.isin(hiddenIdc, byIdc) == True)[0]
        lossWRTHiddenOutput += (
                (R / vectorMagnitude).view(1, -1)
                * (gPrime(predicts[:, outIdx], targets[:, outIdx])
                   * fPrime(I[:, outputIdx])).view(-1, 1)
                * W[:, outputIdx]
        )

        gradU[:, :, outputIdx] += ((gPrime(predicts[:, outIdx], targets[:, outIdx])
                                   * fPrime(I[:, outputIdx])).view(-1, 1)
                                  * (O * (R / vectorMagnitude).view(1, -1))) \
                                 * C[:, outputIdx]

        gradR += (O / vectorMagnitude.view(1, -1)) \
                 * (gPrime(predicts[:, outIdx], targets[:, outIdx])
                    * fPrime(I[:, outputIdx])).view(-1, 1) \
                 * W[:, outputIdx]

        # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
        # uMv = P[outputIdx] - P

        gradP[:, outputIdx, :] += (O * (R / vectorMagnitude).view(1, -1)) \
                                 * (gPrime(predicts[:, outIdx], targets[:, outIdx])
                                    * fPrime(I[:, outputIdx])).view(-1, 1) \
                                 @ (W[:, outputIdx].view(-1, 1) * uMv)

        # gradP[:, outputIdx, 1] = (O * (R / vectorMagnitude).view(1, -1)) \
        #                          * (gPrime(predicts[:, outIdx], targets[:, outIdx])
        #                             * fPrime(I[:, outputIdx])).view(-1, 1) \
        #                          @ (W[:, outputIdx] * -uMv[:, 1])
        #
        # gradP[:, outputIdx, 2] = (O * (R / vectorMagnitude).view(1, -1)) \
        #                          * (gPrime(predicts[:, outIdx], targets[:, outIdx])
        #                             * fPrime(I[:, outputIdx])).view(-1, 1) \
        #                          @ (W[:, outputIdx] * -uMv[:, 2])

        gradP[:, :, 0] += (O * (R / vectorMagnitude).view(1, -1)) \
                          * (gPrime(predicts[:, outIdx], targets[:, outIdx])
                             * fPrime(I[:, outputIdx])).view(-1, 1) \
                          * (W[:, outputIdx] * -uMv[:, 0])

        gradP[:, :, 1] += (O * (R / vectorMagnitude).view(1, -1)) \
                          * (gPrime(predicts[:, outIdx], targets[:, outIdx])
                             * fPrime(I[:, outputIdx])).view(-1, 1) \
                          * (W[:, outputIdx] * -uMv[:, 1])

        gradP[:, :, 2] += (O * (R / vectorMagnitude).view(1, -1)) \
                          * (gPrime(predicts[:, outIdx], targets[:, outIdx])
                             * fPrime(I[:, outputIdx])).view(-1, 1) \
                          * (W[:, outputIdx] * -uMv[:, 2])

    for hiddenIdx in hiddenIdc:
        if hiddenIdx != len(hiddenIdc) - 1:
            lossWRTHiddenOutput[:, hiddenIdx] += lossWRTHiddenOutput[:, hiddenIdx + 1].clone()
        uMv = P[hiddenIdx] - P
        vectorMagnitude = magnitude(uMv).view(-1, 1)

        gradU[:, :, hiddenIdx] += (
                (lossWRTHiddenOutput[:, hiddenIdx]
                 * fPrime(I[:, hiddenIdx])).view(-1, 1)
                * (O * (R / vectorMagnitude).view(1, -1))
        )

        gradR += ((O / vectorMagnitude.view(1, -1))) \
                 * (lossWRTHiddenOutput[:, hiddenIdx]
                    * fPrime(I[:, hiddenIdx])).view(-1, 1) \
                 * W[:, hiddenIdx]

        # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
        # uMv = P[hiddenIdx] - P

        gradP[:, hiddenIdx, :] += (O * (R / vectorMagnitude).view(1, -1)) \
                                 * (lossWRTHiddenOutput[:, hiddenIdx]
                                    * fPrime(I[:, hiddenIdx])).view(-1, 1) \
                                 @ (W[:, hiddenIdx].view(-1, 1) * uMv)

        # gradP[:, hiddenIdx, 1] = (O * (R / vectorMagnitude).view(1, -1)) \
        #                          * (lossWRTHiddenOutput[:, hiddenIdx]
        #                             * fPrime(I[:, hiddenIdx])).view(-1, 1) \
        #                          @ (W[:, hiddenIdx] * uMv[:, 2])
        #
        # gradP[:, hiddenIdx, 2] = (O * (R / vectorMagnitude).view(1, -1)) \
        #                          * (lossWRTHiddenOutput[:, hiddenIdx]
        #                             * fPrime(I[:, hiddenIdx])).view(-1, 1) \
        #                          @ (W[:, hiddenIdx] * uMv[:, 2])

        gradP[:, :, 0] += (O * (R / vectorMagnitude).view(1, -1)) \
                          * (lossWRTHiddenOutput[:, hiddenIdx]
                             * fPrime(I[:, hiddenIdx])).view(-1, 1) \
                          * (W[:, hiddenIdx] * -uMv[:, 0])

        gradP[:, :, 1] += (O * (R / vectorMagnitude).view(1, -1)) \
                          * (lossWRTHiddenOutput[:, hiddenIdx]
                             * fPrime(I[:, hiddenIdx])).view(-1, 1) \
                          * (W[:, hiddenIdx] * -uMv[:, 1])

        gradP[:, :, 2] += (O * (R / vectorMagnitude).view(1, -1)) \
                          * (lossWRTHiddenOutput[:, hiddenIdx]
                             * fPrime(I[:, hiddenIdx])).view(-1, 1) \
                          * (W[:, hiddenIdx] * -uMv[:, 2])

    return gradU, gradP, gradR
