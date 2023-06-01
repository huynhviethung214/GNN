import numpy as np
import torchvision
import os

from tqdm import tqdm
from lib2 import relu, drelu, g, gPrime, f, fPrime, isEnoughInput, initialize

DEBUG = False
simulationIdx = len(os.listdir('./records'))
filename = f'simulation_{simulationIdx}'
dirName = f'records/{filename}'
os.mkdir(f'./{dirName}')

n0 = 4  # Number of chromosomes in a params' population
crossOverRatio = 0.5
mutationProbability = 0.1
numberOfSelectedChromosome0 = 2

n1 = 6  # Number of chromosomes in a connection matrix's population
numInvertedPath = 80
pathCrossOverRatio = 0.5
pathMutationProbability = 0.5
pathInvertingProbability = 0.5
numberOfSelectedChromosome1 = 2

nOutputs = 10
nInputs = 28 * 28

maxFrequency = 3
minFrequency = 2
stepFrequency = 1

maxEta = 9e-3
minEta = 1e-3
stepEta = 1e-3

maxCap = 4000
minCap = 1000
stepCap = 50

maxDecayValue = 9e-4
minDecayValue = 1e-4
stepDecayValue = 1e-4

maxEpoch = 6
minEpoch = 3
stepEpoch = 1

maxNumberOfOutputPerNode = 80
minNumberOfOutputPerNode = 19

maxNumberOfInputPerNode = 40
minNumberOfInputPerNode = 2

initialPopulation = []
fixedVariableIndex = [8]
pathMutationTypes = ['+', '-', '%']

cap = 4000  # Dataset's Cap
generationPerPopulation = 8

bestParams = []
bestParamAccuracy = 0
records = {}

# Preprocessing Training Data
trainingMNIST = torchvision.datasets.MNIST('./data', train=True)
trainingDataset = []

for i, (u, v) in enumerate(tqdm(trainingMNIST, desc='Preprocessing Training Data: ', colour='green')):
    # Target output vector v
    t = np.zeros((1, nOutputs))
    t[0, v] = 1

    trainingDataset.append([(np.array(u) / 255.).flatten(), t])

for _ in range(5):
    np.random.shuffle(trainingDataset)

# Preprocessing Evaluating Data
evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)
evaluatingDataset = []

for u, v in tqdm(evaluatingMNIST, desc='Preprocessing Evaluating Data: ', colour='green'):
    # Target output vector v
    t = np.zeros((1, nOutputs))
    t[0, v] = 1
    evaluatingDataset.append([(np.array(u) / 255.).flatten(), t])


# eta, bias, epochs, decay factor, frequency, input / node, output / node, cap, fixed number of nodes
for i in range(n0):
    initialPopulation.append(
        [
            # np.random.choice(np.arange(minEta, maxEta, stepEta), size=1)[0],  # Random choices of eta (learning rate)
            np.random.choice(np.arange(minEta, maxEta, stepEta), size=1)[0],  # Random choices of eta
            np.random.choice([0, 1], size=1)[0],  # Random choices of bias
            np.random.choice(np.arange(minEpoch, maxEpoch, stepEpoch), size=1)[0],  # Random choices of epochs
            # np.random.choice(np.arange(minDecayValue, maxDecayValue, stepDecayValue), size=1)[0],  # Random choices of decay factor
            np.random.choice(np.arange(minDecayValue, maxDecayValue, stepDecayValue), size=1)[0],  # Random choices of decay factor
            np.random.choice(np.arange(minFrequency, maxFrequency, stepFrequency), size=1)[0],  # Random choices of frequency
            np.random.choice(np.arange(minNumberOfInputPerNode, maxNumberOfInputPerNode, 1), size=1)[0],  # Random number of input / node
            np.random.choice(np.arange(minNumberOfOutputPerNode, maxNumberOfOutputPerNode, 1), size=1)[0],  # Random Number of output / node
            # 8,  # Random number of input / node
            # 15,  # Random Number of output / node
            # cap,  # Dataset's Cap
            np.random.choice(np.arange(minCap, maxCap, stepCap), size=1)[0],  # Dataset's Cap
            nInputs + nOutputs + 6,  # fixed number of nodes ((Number of Input Nodes + Number of Output Nodes) + Number of Hidden Nodes)
        ]
    )


# print(initialPopulation)


# Cross-over & Mutation for Paths
def fixConnectionMatrix(W, C, inputIdc, outputIdc):
    # Remove 2-ways connection(s)
    for n0 in np.where(C == 1)[0]:
        for n1 in np.where(C[n0] == 1)[0]:
            if C[n0, n1] == 1 and C[n1, n0] == 1:
                C[n1, n0] = 0
                C[n0, n1] = 0
                W[n1, n0] = 0

    np.fill_diagonal(C, 0)
    np.fill_diagonal(W, 0)

    for i in outputIdc:
        C[i] = 0
        W[i] = 0

    for i in inputIdc:
        C[:, i] = 0
        W[:, i] = 0

    return [W, C]


def pathCrossOver(crossOverRatio: float, W1, C1, W2, C2):
    sliceIdx = int(crossOverRatio * N)

    _W1 = np.vstack((W1[sliceIdx:], W2[:sliceIdx]))
    _W2 = np.vstack((W2[sliceIdx:], W1[:sliceIdx]))

    _C1 = np.vstack((C1[sliceIdx:], C2[:sliceIdx]))
    _C2 = np.vstack((C2[sliceIdx:], C1[:sliceIdx]))

    return [_W1, _C1], [_W2, _C2]


def addPaths(W, C, nodeIdx, isUpdateCol):
    try:
        if not isUpdateCol:
            rowIdc = np.where(C[:728, nodeIdx] == 0)[0]

            if len(rowIdc) > 0:
                if maxNumberOfOutputPerNode - len(rowIdc) > 0:
                    randomAmountOfPaths = np.random.randint(0, maxNumberOfOutputPerNode - len(rowIdc))
                    rowIdc = np.random.choice(rowIdc, randomAmountOfPaths)[0]
                    C[rowIdc, nodeIdx] = 1
                    W[rowIdc, nodeIdx] = 1
                else:
                    C[np.random.choice(rowIdc, maxNumberOfOutputPerNode)[0]] = 0
        else:
            colIdc = np.where(C[nodeIdx] == 0)[0]

            if len(colIdc) > 0:
                if maxNumberOfInputPerNode - len(colIdc) > 0:
                    randomAmountOfPaths = np.random.randint(0, maxNumberOfInputPerNode - len(colIdc))
                    colIdc = np.random.choice(colIdc, randomAmountOfPaths)[0]
                    C[nodeIdx, colIdc] = 1
                    W[nodeIdx, colIdc] = 1
                else:
                    C[nodeIdx] = 0
                    C[nodeIdx, np.random.choice(colIdc, maxNumberOfInputPerNode)[0]] = 1

    except IndexError:
        pass

    return W, C


def subtractPaths(W, C, nodeIdx, isUpdateCol):
    try:
        if not isUpdateCol:
            rowIdc = np.where(C[:728, nodeIdx] == 1)[0]

            if len(rowIdc) > 0:
                if maxNumberOfOutputPerNode - len(rowIdc) > 0:
                    randomAmountOfPaths = np.random.randint(0, maxNumberOfOutputPerNode - len(rowIdc))
                    rowIdc = np.random.choice(rowIdc, randomAmountOfPaths)[0]
                    C[rowIdc, nodeIdx] = 0
                    W[rowIdc, nodeIdx] = 0
                else:
                    C[np.random.choice(rowIdc, maxNumberOfOutputPerNode)[0]] = 1
        else:
            colIdc = np.where(C[nodeIdx] == 1)[0]

            if len(colIdc) > 0:
                if maxNumberOfInputPerNode - len(colIdc) > 0:
                    randomAmountOfPaths = np.random.randint(0, maxNumberOfInputPerNode - len(colIdc))
                    colIdc = np.random.choice(colIdc, randomAmountOfPaths)[0]
                    C[nodeIdx, colIdc] = 0
                    W[nodeIdx, colIdc] = 0
                else:
                    C[nodeIdx] = 1
                    C[nodeIdx, np.random.choice(colIdc, maxNumberOfInputPerNode)[0]] = 0

    except IndexError:
        pass

    return W, C


# Inverting a random amount of paths (in term of row(s))
def invertingPaths(W, C, nodeIdx):
    selectedCols = np.random.randint(nInputs, N - 1, numInvertedPath)

    C[nodeIdx, selectedCols] = int(not C[nodeIdx, selectedCols].all())
    W[nodeIdx, selectedCols] = int(not W[nodeIdx, selectedCols].all())

    return W, C


def mutatePath(W, C, hiddenIdc):
    # Mutate Input Gate(s) of Hidden Node(s)
    for selectedIdx in hiddenIdc:
        if np.random.uniform(0, 1, 1)[0] <= pathInvertingProbability:
            mutateType = np.random.choice(pathMutationTypes, 1)[0]

            if mutateType == '+':
                W, C = addPaths(W, C, selectedIdx, False)

            elif mutateType == '-':
                W, C = subtractPaths(W, C, selectedIdx, False)

            elif mutateType == '%':
                W, C = invertingPaths(W, C, selectedIdx)

    # Mutate Output Gate(s) of Hidden Node(s)
    for selectedIdx in hiddenIdc:
        if np.random.uniform(0, 1, 1)[0] <= pathInvertingProbability:
            mutateType = np.random.choice(pathMutationTypes, 1)[0]

            if mutateType == '+':
                W, C = addPaths(W, C, selectedIdx, False)

            elif mutateType == '-':
                W, C = subtractPaths(W, C, selectedIdx, False)

            elif mutateType == '%':
                W, C = invertingPaths(W, C, selectedIdx)

    return [W, C]


def findPaths(C, inputIdc, outputIdc, hiddenIdc):
    savedPaths = []

    for outputIdx in outputIdc:
        for inputIdx in inputIdc:
            q = [[inputIdx]]

            while len(q) > 0:
                path = q.pop(0)
                currentNode = path[-1]

                if currentNode == outputIdx:
                    savedPaths.append(path)

                connectedNodes = np.where(C[currentNode, :] == 1)[0]
                for nodeIdx in connectedNodes:
                    if not (nodeIdx in path):
                        newPath = path.copy()
                        newPath.append(nodeIdx)
                        q.append(newPath)

    for outputIdx in outputIdc:
        for hiddenIdx in hiddenIdc:
            q = [[hiddenIdx]]

            while len(q) > 0:
                path = q.pop(0)
                currentNode = path[-1]

                if currentNode == outputIdx:
                    savedPaths.append(path)

                connectedNodes = np.where(C[currentNode, :] == 1)[0]
                for nodeIdx in connectedNodes:
                    if not (nodeIdx in path):
                        newPath = path.copy()
                        newPath.append(nodeIdx)
                        q.append(newPath)

    return savedPaths


def run_model(eta, bias, epochs,
              decay, frequency, W,
              C, N, cap, chromosomeIdx,
              inputIdc, hiddenIdc, outputIdc):
    U = np.zeros((N, N))
    I = np.zeros((N, N))
    O = np.zeros((N, N))
    nodeState = np.zeros((N, N))

    savedPaths = findPaths(C, inputIdc, outputIdc, hiddenIdc)
    sorted(savedPaths, key=lambda p: len(p))

    for epoch in range(1, epochs + 1):
        loss = 0

        if epoch % frequency == 0:
            eta *= decay

        for u, v in tqdm(trainingDataset[:cap],
                         desc=f'Training Chromosome {chromosomeIdx}, Number of paths: {len(savedPaths):<4}, Epoch {epoch:<2} / {epochs:<2}',
                         colour='green',
                         leave=False):
            # Reset Model Variables
            U.fill(0)
            I.fill(0)
            O.fill(0)
            nodeState.fill(0)

            # Forwarding
            queue = hiddenIdc.copy()

            # Input Layer
            for i, inputIdx in enumerate(inputIdc):
                O[inputIdx, inputIdx] = u[i]
                I[inputIdx, inputIdx] = u[i]
                nodeState[inputIdx, inputIdx] = 1

            # Hidden Layer
            counter = 0

            while len(queue) > 0:
                e = queue.pop(0)
                rowIdc = np.where(C[:, e] == 1)[0]

                if DEBUG:
                    counter += 1
                    if counter > 1000000:
                        for e in hiddenIdc:
                            print(f'Not Enough: {e}')
                            rowIdc = np.where(C[:, e] == 1)[0]
                            for rowIdx in rowIdc:
                                if nInputs < rowIdx < min(outputIdc):
                                    print(f'State of Node {rowIdx}: {nodeState[rowIdx, rowIdx]}')
                        print(f'Remained Hidden Nodes: {queue}')
                        raise Exception()

                if isEnoughInput(rowIdc, nodeState):
                    for rowIdx in rowIdc:
                        I[e, e] += W[rowIdx, e] * O[rowIdx, rowIdx]
                    O[e, e] = f(I[e, e] + bias)
                    nodeState[e, e] = 1

                    if DEBUG:
                        counter = 0
                else:
                    queue.append(e)

            # Output Layer
            out = []
            for outputIdx in outputIdc:
                rowIdc = np.where(C[:, outputIdx] == 1)[0]

                for rowIdx in rowIdc:
                    I[outputIdx, outputIdx] += W[rowIdx, outputIdx] * O[rowIdx, rowIdx]

                O[outputIdx, outputIdx] = relu(I[outputIdx, outputIdx])
                nodeState[outputIdx, outputIdx] = 1
                out.append(O[outputIdx, outputIdx])

            predict = np.array(out).reshape(1, -1)

            for path in savedPaths:
                # Backpropagation
                i = path[0]
                j = path[1]
                deltaU = 0

                if len(path) > 2:
                    deltaU = gPrime(predict[0, N - path[-1] - 1], v[0, N - path[-1] - 1]) * O[i, i] * drelu(
                        I[path[-2], path[-2]])

                    for k in range(0, len(path) - 1):
                        nodeIdx0 = path[k]
                        nodeIdx1 = path[k + 1]
                        deltaU *= W[nodeIdx0, nodeIdx1] * fPrime(I[nodeIdx0, nodeIdx0])

                elif len(path) == 2:
                    deltaU = O[i, i] * gPrime(predict[0, N - path[-1] - 1], v[0, N - path[-1] - 1]) * drelu(
                        I[path[-1], path[-1]])

                U[i, j] += deltaU

            W += -eta * U
            loss += g(predict, v)

    score = 0

    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating Chromosome {chromosomeIdx}, Number of paths: {len(savedPaths):<4}',
                     colour='green'):
        I.fill(0)
        O.fill(0)
        nodeState.fill(0)

        # Forwarding
        queue = hiddenIdc.copy()

        # Input Layer
        for i, inputIdx in enumerate(inputIdc):
            O[inputIdx, inputIdx] = u[i]
            I[inputIdx, inputIdx] = u[i]
            nodeState[inputIdx, inputIdx] = 1

        # Hidden Layer
        while len(queue) > 0:
            e = queue.pop(0)
            rowIdc = np.where(C[:, e] == 1)[0]

            if isEnoughInput(rowIdc, nodeState):
                for rowIdx in rowIdc:
                    I[e, e] += W[rowIdx, e] * O[rowIdx, rowIdx]
                O[e, e] = f(I[e, e] + bias)
                nodeState[e, e] = 1
            else:
                queue.append(e)

        # Output Layer
        out = []
        for outputIdx in outputIdc:
            rowIdc = np.where(C[:, outputIdx] == 1)[0]

            for rowIdx in rowIdc:
                I[outputIdx, outputIdx] += W[rowIdx, outputIdx] * O[rowIdx, rowIdx]

            O[outputIdx, outputIdx] = relu(I[outputIdx, outputIdx])
            nodeState[outputIdx, outputIdx] = 1
            out.append(O[outputIdx, outputIdx])

        predict = np.array(out).reshape(1, -1)

        if np.argmax(predict[0]) == np.argmax(v):
            score += 1

    accuracy = (score / len(evaluatingDataset)) * 100
    tqdm.write(f'Chromosome {chromosomeIdx}\'s Accuracy: {accuracy:.2f}')

    return W, C, accuracy


def crossOver(crossOverRatio: float, x1, x2):
    sliceIdx = int(crossOverRatio * len(x1))
    return [*x1[:sliceIdx], *x2[sliceIdx:]], \
           [*x2[:sliceIdx], *x1[sliceIdx:]]


def mutate(x):
    chosenIdc = np.random.choice(np.arange(0, 9, 1), size=3)

    for chosenIdx in chosenIdc:
        if chosenIdx not in fixedVariableIndex:
            if chosenIdx == 0:
                x[chosenIdx] = np.random.choice(np.arange(minEta, maxEta, stepEta), size=1)[0]

            if chosenIdx == 1:
                x[chosenIdx] = np.random.choice([0, 1], size=1)[0]

            if chosenIdx == 2:
                x[chosenIdx] = np.random.choice(np.arange(minEpoch, maxEpoch, stepEpoch), size=1)[0]

            if chosenIdx == 3:
                x[chosenIdx] = np.random.choice(np.arange(minDecayValue, maxDecayValue, stepDecayValue), size=1)[0]

            if chosenIdx == 4:
                x[chosenIdx] = np.random.choice(np.arange(minFrequency, maxFrequency, 1), size=1)[0]

            if chosenIdx == 5:
                x[chosenIdx] = np.random.choice(np.arange(minNumberOfInputPerNode, maxNumberOfInputPerNode, 1), size=1)[
                    0]

            if chosenIdx == 6:
                x[chosenIdx] = np.random.choice(np.arange(minNumberOfOutputPerNode, maxNumberOfOutputPerNode, 1), size=1)[0]

            if chosenIdx == 7:
                x[chosenIdx] = np.random.choice(np.arange(minCap, maxCap, stepCap), size=1)[0]

    return x


def crossOverAndMutation(c0, c1, inputIdc, hiddenIdc, outputIdc):
    chromosome0, chromosome1 = pathCrossOver(crossOverRatio, *c0, *c1)

    if np.random.uniform(0, 1, 1)[0] <= mutationProbability:
        chromosome0 = mutatePath(*chromosome0, hiddenIdc)

    if np.random.uniform(0, 1, 1)[0] <= mutationProbability:
        chromosome1 = mutatePath(*chromosome1, hiddenIdc)

    return fixConnectionMatrix(*chromosome0, inputIdc, outputIdc), fixConnectionMatrix(*chromosome1, inputIdc, outputIdc)


def simulate(eta, bias, epochs, decay, frequency, inputPerNode, outputPerNode, cap, N, paramRecordLabel):
    # print(f'\nPopulation {populationIdx}-th')
    # os.mkdir(f'./{dirName}/{paramRecordLabel}/p_{populationIdx}')
    # populationRecordLabel = f'Population {populationIdx}'
    #
    # if populationRecordLabel not in records.keys():
    #     records[paramRecordLabel].update({populationRecordLabel: {}})

    bestPathAccuracy = 0
    initialpopulation1 = []

    for _ in range(n1):
        W, C = initialize(np.random.randint(inputPerNode, maxNumberOfInputPerNode, 1)[0],
                          np.random.randint(outputPerNode, maxNumberOfOutputPerNode, 1)[0],
                          N)
        initialpopulation1.append([W, C])

    # Array of input, hidden & output nodes' index
    inputIdc = [i for i in range(nInputs)]
    outputIdc = [N - i for i in range(1, nOutputs + 1)]
    hiddenIdc = []

    copiedInputIdc = inputIdc.copy()
    copiedInputIdc.extend(outputIdc)

    for i in range(N):
        if i not in copiedInputIdc:
            hiddenIdc.append(i)

    population1 = initialpopulation1

    for gen in range(generationPerPopulation):
        print(f'\nConnection Matrix Generation {gen}-th')
        print(f'Current Best Connection Matrix Accuracy: {bestPathAccuracy:.2f}')
        os.mkdir(f'./{dirName}/{paramRecordLabel}/gen_{gen}')
        res = []

        for chromosomeIdx, (W, C) in enumerate(population1):
            model_res = run_model(eta, bias, epochs, decay, frequency,
                                  W, C, N,
                                  cap, chromosomeIdx,
                                  inputIdc, hiddenIdc, outputIdc)
            res.append(model_res)

            if model_res[2] > bestPathAccuracy:
                os.mkdir(f'./{dirName}/{paramRecordLabel}/gen_{gen}/ch_{chromosomeIdx}-acc_{model_res[2]:.2f}')
                # records[paramRecordLabel][populationRecordLabel].update({
                #     f'Generation {gen}': {
                #         'connection_matrix': model_res[0],
                #         'weight_matrix': model_res[1],
                #         'accuracy': model_res[2]
                #     }
                # })
                bestPathAccuracy = model_res[2]
                np.save(f'./{dirName}/{paramRecordLabel}/gen_{gen}/ch_{chromosomeIdx}-acc_{model_res[2]:.2f}/c.npy', model_res[0])
                np.save(f'./{dirName}/{paramRecordLabel}/gen_{gen}/ch_{chromosomeIdx}-acc_{model_res[2]:.2f}/w.npy', model_res[1])

        population1 = [[W, C] for (W, C, _) in sorted(res, key=lambda x: x[-1])]
        population1.reverse()

        remainChild = population1[numberOfSelectedChromosome1:]
        newPopulation = population1[:numberOfSelectedChromosome1]  # newPopulation contain `n` number of elites

        for i in range(0, len(remainChild), 2):
            chromosome0, chromosome1 = crossOverAndMutation(remainChild[i], remainChild[i + 1],
                                                            inputIdc, hiddenIdc, outputIdc)
            newPopulation.append(chromosome0)
            newPopulation.append(chromosome1)

        population1 = newPopulation

    return bestPathAccuracy


if __name__ == '__main__':
    population0 = initialPopulation
    gc = 0

    while bestParamAccuracy <= 90:
        # if bestParamAccuracy >= 70:
        #     try:
        #         eta, bias, epochs, decay, frequency, inputPerNode, outputPerNode, cap, N, populationIdx = bestParams
        #         strEta = str(eta).replace('-', '.')
        #         path = f'{bestParamAccuracy:.2f}-{N}-{inputPerNode}-{outputPerNode}-{cap}-{strEta}-{epochs}-{bias}-{decay}-{frequency}'
        #         os.mkdir(f'./{path}')
        #         np.save(f'./{path}/weights', record[f'Population {populationIdx}']['weight_matrix'])
        #         np.save(f'./{path}/connections', record[f'Population {populationIdx}']['connection_matrix'])
        #
        #     except Exception as e:
        #         pass

        print(f'\nParams\' Generation {gc}', end='')
        paramAccuracies = []
        for paramIdx, param in enumerate(population0):
            eta, bias, epochs, decay, frequency, inputPerNode, outputPerNode, cap, N = param
            strEta = str(eta).replace('-', '.')

            i = 0
            paramRecordLabel = f'{N}-{inputPerNode}-{outputPerNode}-{cap}-{strEta}-{epochs}-{bias}-{decay}-{frequency} ({i})'

            while True:
                try:
                    paramRecordLabel = f'{N}-{inputPerNode}-{outputPerNode}-{cap}-{strEta}-{epochs}-{bias}-{decay}-{frequency} ({i})'
                    os.mkdir(f'./{dirName}/{paramRecordLabel}')
                    break

                except FileExistsError:
                    i += 1
                    continue

            print(f'\nParameters\' {paramIdx}-th: {param}, (eta, bias, epochs, decay factor, frequency, input / node, output / node, cap, fixed number of nodes)')
            paramAccuracies.append(simulate(*param, paramRecordLabel=paramRecordLabel))

        population0 = sorted(population0, key=lambda x: paramAccuracies[population0.index(x)])
        population0.reverse()
        bestParams = population0[0]
        bestParamAccuracy = max(paramAccuracies)
        print(f'Best Parameters: {bestParams}, Accuracy: {bestParamAccuracy:.2f}\n')

        elites = population0[:numberOfSelectedChromosome0]
        remainPopulation = population0[numberOfSelectedChromosome0:]
        newPopulation = elites

        for i in range(0, len(remainPopulation), 2):
            child1, child2 = crossOver(crossOverRatio, remainPopulation[i], remainPopulation[i + 1])

            if np.random.uniform(0, 1, 1)[0] <= mutationProbability:
                child1 = mutate(child1)

            if np.random.uniform(0, 1, 1)[0] <= mutationProbability:
                child2 = mutate(child2)

            newPopulation.append(child1)
            newPopulation.append(child2)

        population0 = newPopulation
        gc = gc + 1
