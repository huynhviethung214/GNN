import numpy as np

from numba import njit, prange

# TODO: BUY A NEW CPU BEFORE RUNNING THIS BECAUSE IT'S GONNA TAKE ABOUT
#  3 HOURS TO TRAIN ON AN I3-10100F (WITH AN EXTRA PARAMETER IS `bias`
#  WHICH IS A VECTOR THAT HOLD THE ACTIVATION LIMIT FOR EACH NEURON)


PI = np.round(np.pi, 4)


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
        C = np.ones((N, N), dtype=np.int64)  # Connections Matrix

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
    return np.sqrt((v0[0] - v1[0]) ** 2
                   + (v0[1] - v1[1]) ** 2
                   + (v0[2] - v1[2]) ** 2)


# Loss Function
@njit(fastmath=True)
def g(p, t, nOutputs):
    return (np.sum((p - t) ** 2, axis=0)) / nOutputs


@njit(fastmath=True)
def gPrime(p, t, nOutputs):
    return (2 / nOutputs) * (p - t)


# Based on mechanical wave equation y(x,t) = A * sin(2pi * x/lambda + 2pi * f * t + phi)
@njit(parallel=True, fastmath=True)
def forward(t, A, L, F, PHI, W, P, R,
            I, O, u, inputIdc, hiddenIdc, outputIdc):
    for i in prange(len(inputIdc)):
        inputIdx = inputIdc[i]
        I[inputIdx] = u[inputIdx]
        O[inputIdx] = u[inputIdx]

    for i in prange(len(hiddenIdc)):
        nodeIdx = hiddenIdc[i]
        byIdc = np.where(W[:, nodeIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[nodeIdx])

            I[nodeIdx] += O[byIdx] * W[byIdx, nodeIdx] \
                          * R[nodeIdx] / vectorMagnitude

        O[nodeIdx] = A[nodeIdx] \
                     * np.sin((2 * PI * I[nodeIdx]) / L[nodeIdx]
                              + (2 * PI * F[nodeIdx] * t)
                              + PHI[nodeIdx])

    for i in prange(len(outputIdc)):
        nodeIdx = outputIdc[i]
        byIdc = np.where(W[:, nodeIdx] != 0)[0]

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[nodeIdx])

            I[nodeIdx] += O[byIdx] * W[byIdx, nodeIdx] \
                          * R[nodeIdx] / vectorMagnitude

        O[nodeIdx] = A[nodeIdx] \
                     * np.sin((2 * PI * I[nodeIdx]) / L[nodeIdx]
                              + (2 * PI * F[nodeIdx] * t)
                              + PHI[nodeIdx])

    return I, O


@njit()
def getMagnitude(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@njit(parallel=True, fastmath=True)
def calculateGradients(v, t,
                       A, L, F, PHI,
                       W, R, P, I, O,
                       outputIdc, nOutputs,
                       hiddenIdc, nHiddens,
                       gradA, gradL, gradF, gradPHI,
                       gradW, gradP, gradR,
                       lossWRTHiddenOutput, nInputs):
    for i in prange(nOutputs):
        outputIdx = outputIdc[i]
        outIdx = outputIdx - (nInputs + nHiddens)
        byIdc = np.where(W[:, outputIdx] != 0)[0]

        gradA[outputIdx] = np.sin((2 * PI * I[outputIdx]) / L[outputIdx]
                                  + (2 * PI * F[outputIdx] * t)
                                  + PHI[outputIdx])

        gradL[outputIdx] = -(2 * PI * I[outputIdx]) \
                           * 1 \
                           * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                    + (2 * PI * F[outputIdx] * t)
                                    + PHI[outputIdx]) \
                           / (L[outputIdx] ** 2)

        gradF[outputIdx] = 2 * PI * t \
                           * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                    + (2 * PI * F[outputIdx] * t)
                                    + PHI[outputIdx])

        gradPHI[outputIdx] = 1 \
                             * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                      + (2 * PI * F[outputIdx] * t)
                                      + PHI[outputIdx])

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[outputIdx])

            # * gPrime(O[outputIdx], v[outIdx], nOutputs)

            if len(np.where(hiddenIdc == byIdx)[0]):
                lossWRTHiddenOutput[byIdx] += W[byIdx, outputIdx] \
                                              * gPrime(O[outputIdx], v[outIdx], nOutputs) \
                                              * 2 * PI \
                                              * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                                       + (2 * PI * F[outputIdx] * t)
                                                       + PHI[outputIdx]) \
                                              * R[byIdx] \
                                              / (vectorMagnitude * L[outputIdx])

                if len(np.where(hiddenIdc == byIdx + 1)[0]):
                    lossWRTHiddenOutput[byIdx] += lossWRTHiddenOutput[byIdx + 1]

            gradW[byIdx, outputIdx] += O[byIdx] \
                                       * gPrime(O[outputIdx], v[outIdx], nOutputs) \
                                       * 2 * PI \
                                       * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                                + (2 * PI * F[outputIdx] * t)
                                                + PHI[outputIdx]) \
                                       * R[byIdx] \
                                       / (vectorMagnitude * L[outputIdx])

            gradR[byIdx] += O[byIdx] \
                            * gPrime(O[outputIdx], v[outIdx], nOutputs) \
                            * 2 * PI \
                            * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                     + (2 * PI * F[outputIdx] * t)
                                     + PHI[outputIdx]) \
                            * W[byIdx, outputIdx] \
                            / (vectorMagnitude * L[outputIdx])

            # Loss w.r.t position of outputNode (0 <=> x's coordinate, 1 <=> y's coordinate, 2 <=> z's coordinate)
            # x0 = P[outputIdx, 0]
            # y0 = P[outputIdx, 1]
            # z0 = P[outputIdx, 2]
            #
            # x1 = P[byIdx, 0]
            # y1 = P[byIdx, 1]
            # z1 = P[byIdx, 2]
            directionalVector = (P[outputIdx] - P[byIdx]).flatten()

            for axisIdx in prange(3):
                gradP[outputIdx, 0] += O[byIdx] \
                                       * gPrime(O[outputIdx], v[outIdx], nOutputs) \
                                       * W[byIdx, outputIdx] \
                                       * R[byIdx] \
                                       * 2 * PI \
                                       * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                                + (2 * PI * F[outputIdx] * t)
                                                + PHI[outputIdx]) \
                                       * -directionalVector[axisIdx] \
                                       / ((vectorMagnitude ** 3) * L[outputIdx])

            for axisIdx in prange(3):
                gradP[byIdx, 0] += O[byIdx] \
                                   * gPrime(O[outputIdx], v[outIdx], nOutputs) \
                                   * W[byIdx, outputIdx] \
                                   * R[byIdx] \
                                   * 2 * PI \
                                   * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
                                            + (2 * PI * F[outputIdx] * t)
                                            + PHI[outputIdx]) \
                                   * directionalVector[axisIdx] \
                                   / ((vectorMagnitude ** 3) * L[outputIdx])

            # gradP[outputIdx, 0] += O[byIdx] \
            #                        * gPrime(O[outputIdx],
            #                                 v[outIdx],
            #                                 nOutputs) \
            #                        * W[byIdx, outputIdx] \
            #                        * R[byIdx] \
            #                        * 2 * PI \
            #                        * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
            #                                 + (2 * PI * F[outputIdx] * t)
            #                                 + PHI[outputIdx]) \
            #                        * -(x0 - x1) \
            #                        / ((vectorMagnitude ** 3) * L[outputIdx])

            # gradP[outputIdx, 1] += O[byIdx] \
            #                        * gPrime(O[outputIdx],
            #                                 v[outIdx],
            #                                 nOutputs) \
            #                        * W[byIdx, outputIdx] \
            #                        * R[byIdx] \
            #                        * 2 * PI \
            #                        * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
            #                                 + (2 * PI * F[outputIdx] * t)
            #                                 + PHI[outputIdx]) \
            #                        * -(y0 - y1) \
            #                        / ((vectorMagnitude ** 3) * L[outputIdx])
            #
            # gradP[outputIdx, 2] += O[byIdx] \
            #                        * gPrime(O[outputIdx],
            #                                 v[outIdx],
            #                                 nOutputs) \
            #                        * W[byIdx, outputIdx] \
            #                        * R[byIdx] \
            #                        * 2 * PI \
            #                        * np.cos((2 * PI * I[outputIdx]) / L[outputIdx]
            #                                 + (2 * PI * F[outputIdx] * t)
            #                                 + PHI[outputIdx]) \
            #                        * -(z0 - z1) \
            #                        / ((vectorMagnitude ** 3) * L[outputIdx])

            # gradP[byIdx, 0] += O[byIdx] \
            #                    * gPrime(O[outputIdx],
            #                             v[outIdx],
            #                             nOutputs) \
            #                    * W[byIdx, outputIdx] \
            #                    / L[outputIdx] \
            #                    * 2 * PI * 1 \
            #                    * np.cos(2 * PI
            #                             * (I[outputIdx] / L[outputIdx]
            #                                + F[outputIdx] * t)
            #                             + PHI[outputIdx]) \
            #                    * (R[byIdx]
            #                       * (x0 - x1)
            #                       / (vectorMagnitude ** 3))
            #
            # gradP[byIdx, 1] += O[byIdx] \
            #                    * gPrime(O[outputIdx],
            #                             v[outIdx],
            #                             nOutputs) \
            #                    * W[byIdx, outputIdx] \
            #                    / L[outputIdx] \
            #                    * 2 * PI * 1 \
            #                    * np.cos(2 * PI
            #                             * (I[outputIdx] / L[outputIdx]
            #                                + F[outputIdx] * t)
            #                             + PHI[outputIdx]) \
            #                    * (R[byIdx]
            #                       * (y0 - y1)
            #                       / (vectorMagnitude ** 3))
            #
            # gradP[byIdx, 2] += O[byIdx] \
            #                    * gPrime(O[outputIdx],
            #                             v[outIdx],
            #                             nOutputs) \
            #                    * W[byIdx, outputIdx] \
            #                    / L[outputIdx] \
            #                    * 2 * PI * 1 \
            #                    * np.cos(2 * PI
            #                             * (I[outputIdx] / L[outputIdx]
            #                                + F[outputIdx] * t)
            #                             + PHI[outputIdx]) \
            #                    * (R[byIdx]
            #                       * (z0 - z1)
            #                       / (vectorMagnitude ** 3))

    for i in prange(nHiddens):
        hiddenIdx = hiddenIdc[::-1][i]
        byIdc = np.where(W[:, hiddenIdx] != 0)[0]

        gradA[hiddenIdx] = np.sin((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                  + (2 * PI * F[hiddenIdx] * t)
                                  + PHI[hiddenIdx])

        gradL[hiddenIdx] = -(2 * PI * I[hiddenIdx]) \
                           * A[hiddenIdx] \
                           * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                    + (2 * PI * F[hiddenIdx] * t)
                                    + PHI[hiddenIdx]) \
                           / (L[hiddenIdx] ** 2)

        gradF[hiddenIdx] = A[hiddenIdx] * 2 * PI * t \
                           * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                    + (2 * PI * F[hiddenIdx] * t)
                                    + PHI[hiddenIdx])

        gradPHI[hiddenIdx] = A[hiddenIdx] \
                             * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                      + (2 * PI * F[hiddenIdx] * t)
                                      + PHI[hiddenIdx])

        for j in prange(len(byIdc)):
            byIdx = byIdc[j]
            vectorMagnitude = magnitude(P[byIdx], P[hiddenIdx])

            gradW[byIdx, hiddenIdx] += O[byIdx] \
                                       * lossWRTHiddenOutput[hiddenIdx] \
                                       * R[byIdx] \
                                       * 2 * PI * A[hiddenIdx] \
                                       * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                                + (2 * PI * F[hiddenIdx] * t)
                                                + PHI[hiddenIdx]) \
                                       / (vectorMagnitude * L[hiddenIdx])

            gradR[byIdx] += O[byIdx] \
                            * lossWRTHiddenOutput[hiddenIdx] \
                            * W[byIdx, hiddenIdx] \
                            * 2 * PI * A[hiddenIdx] \
                            * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                     + (2 * PI * F[hiddenIdx] * t)
                                     + PHI[hiddenIdx]) \
                            / (vectorMagnitude * L[hiddenIdx])

            # Loss w.r.t position of outputNode (0 <=> x0 coordinate, 1 <=> y0 coordinate)
            directionalVector = (P[hiddenIdx] - P[byIdx]).flatten()

            for axisIdx in prange(3):
                gradP[hiddenIdx, 0] += O[byIdx] \
                                       * lossWRTHiddenOutput[hiddenIdx] \
                                       * W[byIdx, hiddenIdx] \
                                       * R[byIdx] \
                                       * 2 * PI * A[hiddenIdx] \
                                       * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                                + (2 * PI * F[hiddenIdx] * t)
                                                + PHI[hiddenIdx]) \
                                       * -directionalVector[axisIdx] \
                                       / ((vectorMagnitude ** 3) * L[hiddenIdx])

            for axisIdx in prange(3):
                gradP[byIdx, 0] += O[byIdx] \
                                   * lossWRTHiddenOutput[hiddenIdx] \
                                   * W[byIdx, hiddenIdx] \
                                   * R[byIdx] \
                                   * 2 * PI * A[hiddenIdx] \
                                   * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
                                            + (2 * PI * F[hiddenIdx] * t)
                                            + PHI[hiddenIdx]) \
                                   * directionalVector[axisIdx] \
                                   / ((vectorMagnitude ** 3) * L[hiddenIdx])

            # x0 = P[hiddenIdx, 0]
            # y0 = P[hiddenIdx, 1]
            # z0 = P[hiddenIdx, 2]
            #
            # x1 = P[byIdx, 0]
            # y1 = P[byIdx, 1]
            # z1 = P[byIdx, 2]
            #
            # gradP[hiddenIdx, 0] += O[byIdx] \
            #                        * lossWRTHiddenOutput[hiddenIdx] \
            #                        * W[byIdx, hiddenIdx] \
            #                        * R[byIdx] \
            #                        * 2 * PI * A[hiddenIdx] \
            #                        * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
            #                                     + (2 * PI * F[hiddenIdx] * t)
            #                                     + PHI[hiddenIdx]) \
            #                        * -(x0 - x1) \
            #                        / ((vectorMagnitude ** 3) * L[hiddenIdx])
            #
            # gradP[hiddenIdx, 1] += O[byIdx] \
            #                        * lossWRTHiddenOutput[hiddenIdx] \
            #                        * W[byIdx, hiddenIdx] \
            #                        * R[byIdx] \
            #                        * 2 * PI * A[hiddenIdx] \
            #                        * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
            #                                 + (2 * PI * F[hiddenIdx] * t)
            #                                 + PHI[hiddenIdx]) \
            #                        * -(y0 - y1) \
            #                        / ((vectorMagnitude ** 3) * L[hiddenIdx])
            #
            # gradP[hiddenIdx, 2] += O[byIdx] \
            #                        * lossWRTHiddenOutput[hiddenIdx] \
            #                        * W[byIdx, hiddenIdx] \
            #                        * R[byIdx] \
            #                        * 2 * PI * A[hiddenIdx] \
            #                        * np.cos((2 * PI * I[hiddenIdx]) / L[hiddenIdx]
            #                                 + (2 * PI * F[hiddenIdx] * t)
            #                                 + PHI[hiddenIdx]) \
            #                        * -(z0 - z1) \
            #                        / ((vectorMagnitude ** 3) * L[hiddenIdx])

            # gradP[byIdx, 0] += O[byIdx] \
            #                    * lossWRTHiddenOutput[hiddenIdx] \
            #                    * W[byIdx, hiddenIdx] \
            #                    / L[hiddenIdx] \
            #                    * 2 * PI * A[hiddenIdx] \
            #                    * np.cos(2 * PI
            #                             * (I[hiddenIdx] / L[hiddenIdx]
            #                                + F[hiddenIdx] * t)
            #                             + PHI[hiddenIdx]) \
            #                    * (R[byIdx]
            #                       * (x0 - x1)
            #                       / (vectorMagnitude ** 3))
            #
            # gradP[byIdx, 1] += O[byIdx] \
            #                    * lossWRTHiddenOutput[hiddenIdx] \
            #                    * W[byIdx, hiddenIdx] \
            #                    / L[hiddenIdx] \
            #                    * 2 * PI * A[hiddenIdx] \
            #                    * np.cos(2 * PI
            #                             * (I[hiddenIdx] / L[hiddenIdx]
            #                                + F[hiddenIdx] * t)
            #                             + PHI[hiddenIdx]) \
            #                    * (R[byIdx]
            #                       * (y0 - y1)
            #                       / (vectorMagnitude ** 3))
            #
            # gradP[byIdx, 2] += O[byIdx] \
            #                    * lossWRTHiddenOutput[hiddenIdx] \
            #                    * W[byIdx, hiddenIdx] \
            #                    / L[hiddenIdx] \
            #                    * 2 * PI * A[hiddenIdx] \
            #                    * np.cos(2 * PI
            #                             * (I[hiddenIdx] / L[hiddenIdx]
            #                                + F[hiddenIdx] * t)
            #                             + PHI[hiddenIdx]) \
            #                    * (R[byIdx]
            #                       * (z0 - z1)
            #                       / (vectorMagnitude ** 3))

    return gradW, gradR, gradP, gradA, gradL, gradF, gradPHI
