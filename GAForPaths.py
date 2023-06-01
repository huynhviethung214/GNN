import numpy as np
import torchvision
import os
import json

from time import sleep
from tqdm import tqdm
from lib2 import initialize, invertingPaths, crossOver, forward, backward

DEBUG = True


class GeneticAlgorithm:
    def __init__(self, configs: dict, download_dataset: bool = False, config_path: str = ''):
        self.simulationIdx = len(os.listdir('./records')) + 1
        self.filename = f'connection_matrix_simulation_{self.simulationIdx}'
        self.dirName = f'records/{self.filename}'

        os.mkdir(f'./{self.dirName}')
        os.mkdir(f'./{self.dirName}/history')

        # Configurations
        self.mutateTypes = ['+', '-', '%', '+n']

        if not config_path:
            self.applyConfigs(configs)
            self.saveConfigs(configs)
        else:
            self.loadConfigs(config_path)

        self.inputIdc = [i for i in range(self.nInputs)]  # NOQA
        self.hiddenIdc = [i + self.nInputs for i in range(self.nHiddens)]  # NOQA
        self.outputIdc = [i + self.nInputs + self.nHiddens for i in range(self.nOutputs)]  # NOQA

        # Preprocessing Training Data
        trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=download_dataset)
        # trainingMNIST = torchvision.datasets.EMNIST('./data', train=True, download=download_dataset, split='balanced')
        self.trainingDataset = []

        for i, (u, v) in enumerate(tqdm(trainingMNIST, desc='Preprocessing Training Data: ', colour='green')):
            # Target output vector v
            t = np.zeros((1, self.nOutputs))  # NOQA
            t[0, v] = 1

            self.trainingDataset.append([(np.array(u) / 255.).flatten(), t])

        # Preprocessing Evaluating Data
        evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)
        # evaluatingMNIST = torchvision.datasets.EMNIST('./data', train=False, split='balanced')
        self.evaluatingDataset = []

        for u, v in tqdm(evaluatingMNIST, desc='Preprocessing Evaluating Data: ', colour='green'):
            # Target output vector v
            t = np.zeros((1, self.nOutputs))  # NOQA
            t[0, v] = 1
            self.evaluatingDataset.append([(np.array(u) / 255.).flatten(), t])

        self.initialPopulation = []
        for i in range(self.n):  # NOQA

            W, C = initialize(np.random.randint(self.minInputPerNode, self.maxInputPerNode, 1)[0],  # NOQA
                              np.random.randint(self.minOutputPerNode, self.maxOutputPerNode, 1)[0],  # NOQA
                              self.N, self.nInputs, self.nHiddens, self.nOutputs)  # NOQA
            self.initialPopulation.append(self.fixConnectionMatrix(W, C))

    def saveConfigs(self, configs):
        with open(f'./{self.dirName}/configs.json', 'w') as f:
            json.dump(configs, f, indent=4)

    def applyConfigs(self, configs):
        for (key, value) in configs.items():
            setattr(self, key, value)

    def loadConfigs(self, path):
        with open(path, 'r') as f:
            configs = json.load(f)
            self.applyConfigs(configs)

    def fixConnectionMatrix(self, W, C):
        # Remove 2-ways connection(s)
        for n0 in np.where(C == 1)[0]:
            for n1 in np.where(C[n0] == 1)[0]:
                if C[n0, n1] == 1 and C[n1, n0] == 1:
                    if np.random.uniform(0, 1, 1) <= 0.5:
                        C[n1, n0] = 0
                        W[n1, n0] = 0
                    else:
                        C[n0, n1] = 0
                        W[n0, n1] = 0

        # Fix connections that is violating the constraints
        np.fill_diagonal(C, 0)
        np.fill_diagonal(W, 0)

        C[self.outputIdc] = 0
        W[self.outputIdc] = 0

        C[:, self.inputIdc] = 0
        W[:, self.inputIdc] = 0

        # Remove loops in hidden layer
        loops = []
        for h0 in self.hiddenIdc:
            closed = []
            queue = [h0]

            while len(queue) > 0:
                n0 = queue.pop(0)
                connectedNodes = np.where(C[n0, :] == 1)[0]
                for node in connectedNodes:
                    if node != h0 and node not in closed:
                        queue.append(node)
                closed.append(n0)

        if len(loops) > 0:
            for loop in loops:
                if len(loop) > 1:
                    C[loop[-2], loop[-1]] = 0
                    W[loop[-2], loop[-1]] = 0

        return [W, C]

    def addConnections(self, W, C, nodeIdx, isUpdateCol):
        try:
            if not isUpdateCol:
                rowIdc = np.where(C[:728, nodeIdx] == 0)[0]

                if len(rowIdc) > 0:
                    if self.maxOutputPerNode - len(rowIdc) > 0:  # NOQA
                        randomAmountOfPaths = np.random.randint(0, self.maxOutputPerNode - len(rowIdc))  # NOQA
                        rowIdc = np.random.choice(rowIdc, randomAmountOfPaths)[0]
                        C[rowIdc, nodeIdx] = 1
                        W[rowIdc, nodeIdx] = 1
                    else:
                        C[np.random.choice(rowIdc, self.maxOutputPerNode)[0], nodeIdx] = 0  # NOQA
                        W[np.random.choice(rowIdc, self.maxOutputPerNode)[0], nodeIdx] = 0  # NOQA
            else:
                colIdc = np.where(C[nodeIdx] == 0)[0]

                if len(colIdc) > 0:
                    if self.maxInputPerNode - len(colIdc) > 0:  # NOQA
                        randomAmountOfPaths = np.random.randint(0, self.maxInputPerNode - len(colIdc))  # NOQA
                        colIdc = np.random.choice(colIdc, randomAmountOfPaths)[0]
                        C[nodeIdx, colIdc] = 1
                        W[nodeIdx, colIdc] = 1
                    else:
                        C[nodeIdx, np.random.choice(colIdc, self.maxInputPerNode)[0]] = 0  # NOQA
                        W[nodeIdx, np.random.choice(colIdc, self.maxInputPerNode)[0]] = 0  # NOQA

        except IndexError:
            pass

        return W, C

    def subtractConnections(self, W, C, nodeIdx, isUpdateCol):
        try:
            if not isUpdateCol:
                rowIdc = np.where(C[:728, nodeIdx] == 1)[0]

                if len(rowIdc) > 0:
                    if self.maxOutputPerNode - len(rowIdc) > 0:  # NOQA
                        randomAmountOfPaths = np.random.randint(0, self.maxOutputPerNode - len(rowIdc))  # NOQA
                        rowIdc = np.random.choice(rowIdc, randomAmountOfPaths)[0]
                        C[rowIdc, nodeIdx] = 0
                        W[rowIdc, nodeIdx] = 0
                    else:
                        C[np.random.choice(rowIdc, self.maxOutputPerNode)[0], nodeIdx] = 1  # NOQA
                        W[np.random.choice(rowIdc, self.maxOutputPerNode)[0], nodeIdx] = 1  # NOQA
            else:
                colIdc = np.where(C[nodeIdx] == 1)[0]

                if len(colIdc) > 0:
                    if self.maxInputPerNode - len(colIdc) > 0:  # NOQA
                        randomAmountOfPaths = np.random.randint(0, self.maxInputPerNode - len(colIdc))  # NOQA
                        colIdc = np.random.choice(colIdc, randomAmountOfPaths)[0]
                        C[nodeIdx, colIdc] = 0
                        W[nodeIdx, colIdc] = 0
                    else:
                        C[nodeIdx, np.random.choice(colIdc, self.maxInputPerNode)[0]] = 0  # NOQA
                        W[nodeIdx, np.random.choice(colIdc, self.maxInputPerNode)[0]] = 0  # NOQA

        except IndexError:
            pass

        return W, C

    def mutation(self, W, C):
        # Mutate Input Gate(s) of Hidden Node(s)
        for selectedIdx in self.hiddenIdc:
            if np.random.uniform(0, 1, 1)[0] <= self.signalProbability:  # NOQA
                for mutateType in np.random.choice(self.mutateTypes, 2):
                    if mutateType == '+':
                        W, C = self.addConnections(W, C, selectedIdx, True)  # NOQA

                    elif mutateType == '-':
                        W, C = self.subtractConnections(W, C, selectedIdx, True)  # NOQA

                    elif mutateType == '%':
                        W, C = invertingPaths(W, C, selectedIdx, self.numInverting, self.nInputs, self.N)  # NOQA

        # Mutate Output Gate(s) of Hidden Node(s)
        for selectedIdx in self.hiddenIdc:
            if np.random.uniform(0, 1, 1)[0] <= self.signalProbability:  # NOQA
                for mutateType in np.random.choice(self.mutateTypes, 2):
                    if mutateType == '+':
                        W, C = self.addConnections(W, C, selectedIdx, False)  # NOQA

                    elif mutateType == '-':
                        W, C = self.subtractConnections(W, C, selectedIdx, False)  # NOQA

                    elif mutateType == '%':
                        W, C = invertingPaths(W, C, selectedIdx, self.numInverting, self.nInputs, self.N)  # NOQA

        return [W, C]

    def run_model(self, chromosomeIdx, W, C, eta, genFolderPath):
        U = np.zeros((self.N, self.N))  # NOQA
        I = np.zeros((self.N, self.N))  # NOQA
        O = np.zeros((self.N, self.N))  # NOQA

        numberOfTrainableParams = len(np.where(W != 0)[0])
        numberOfUsableParams = np.square(self.N) - self.N - self.nInputs * self.N - self.nOutputs * (self.N - self.nInputs)  # NOQA

        for epoch in range(1, self.epochs + 1):  # NOQA
            if epoch % self.frequency == 0:  # NOQA
                eta *= self.decay  # NOQA

            for u, v in tqdm(self.trainingDataset[:self.cap],  # NOQA
                             desc=f'Training Chromosome {chromosomeIdx:<{len(str(self.n))}}, '  # NOQA
                                  f'Number of trainable parameters: {numberOfTrainableParams:<5}, '
                                  f'Number of usable parameters: {numberOfUsableParams - numberOfTrainableParams:<5}, '
                                  f'Epoch {epoch:<2} / {self.epochs:<2}',  # NOQA
                             colour='green',
                             leave=False):
                # Reset Model Variables
                I.fill(0)
                O.fill(0)
                U.fill(0)

                # Forwarding
                W, C, O, I = forward(W, C, O, I,
                                     np.array(self.inputIdc, dtype=np.int32),
                                     np.array(self.hiddenIdc, dtype=np.int32),
                                     np.array(self.outputIdc, dtype=np.int32),
                                     u, self.bias)  # NOQA

                predict = np.diag(O)[min(self.outputIdc):].reshape((1, -1))

                # Backward
                hiddenIdc = self.hiddenIdc.copy()
                hiddenIdc.reverse()
                hiddenIdc = np.array(hiddenIdc)

                U = backward(W, C, O, I, U,
                             self.nInputs, self.nHiddens, hiddenIdc,  # NOQA
                             np.array(self.outputIdc), predict, v)  # NOQA
                W += -eta * U

        score = 0
        for u, v in tqdm(self.evaluatingDataset,
                         desc=f'Evaluating Chromosome {chromosomeIdx:<{len(str(self.n))}}, '  # NOQA
                              f'Number of trainable parameters: {numberOfTrainableParams:<5}, '  # NOQA
                              f'Number of usable parameters: {numberOfUsableParams - numberOfTrainableParams:<5}',  # NOQA
                         colour='green'):
            I.fill(0)
            O.fill(0)

            # Forwarding
            W, C, O, I = forward(W, C, O, I,
                                 np.array(self.inputIdc, dtype=np.int32),
                                 np.array(self.hiddenIdc, dtype=np.int32),
                                 np.array(self.outputIdc, dtype=np.int32),
                                 u, self.bias)  # NOQA

            predict = np.diag(O)[min(self.outputIdc):].reshape((1, -1))

            if np.argmax(predict[0]) == np.argmax(v):
                score += 1

        accuracy = (score / len(self.evaluatingDataset)) * 100
        tqdm.write(f'Chromosome {chromosomeIdx:<{len(str(self.n))}}\'s Accuracy: {accuracy:.2f}\n')  # NOQA
        os.mkdir(f'{genFolderPath}/ch_{chromosomeIdx}-{accuracy:.2f}')
        np.save(f'{genFolderPath}/ch_{chromosomeIdx}-{accuracy:.2f}/w.npy', W)
        np.save(f'{genFolderPath}/ch_{chromosomeIdx}-{accuracy:.2f}/c.npy', C)
        sleep(0.5)

        return [W, C, accuracy]  # NOQA

    def crossOverAndMutation(self, c0, c1):
        W1, C1, W2, C2 = crossOver(self.crossOverRatio, self.N, *c0, *c1)  # NOQA
        chromosome0 = (W1, C1)
        chromosome1 = (W2, C2)

        if np.random.uniform(0, 1, 1)[0] <= self.mutationProbability:  # NOQA
            chromosome0 = self.mutation(*chromosome0)

        if np.random.uniform(0, 1, 1)[0] <= self.mutationProbability:  # NOQA
            chromosome1 = self.mutation(*chromosome1)

        return self.fixConnectionMatrix(*chromosome0), self.fixConnectionMatrix(*chromosome1)

    def run_algorithm(self):
        population = self.initialPopulation
        gc = 1
        currentBestAccuracy = 0

        while currentBestAccuracy <= self.goal:  # NOQA
            print(f'\nGeneration {gc}')
            genFolderPath = f'./{self.dirName}/history/gen_{gc}'
            os.mkdir(genFolderPath)

            res = []
            for chromosomeIdx, (W, C) in enumerate(population):
                tqdm.write(f'Assert len(W != 0) == len(C == 1): {len(np.where(W != 0)[0]) == len(np.where(C == 1)[0])}')
                sleep(0.5)
                res.append(self.run_model(chromosomeIdx, W, C, self.eta, genFolderPath))  # NOQA

            sortedRes = sorted(res, key=lambda x: x[-1])
            sortedRes.reverse()

            tqdm.write(f'Current Best Accuracy: {currentBestAccuracy:.2f}')

            newAccuracy = sortedRes[0][-1]
            if newAccuracy > currentBestAccuracy:
                currentBestAccuracy = newAccuracy

            tqdm.write(f'Generation\'s Best Accuracy: {newAccuracy:.2f}')

            population = [[W, C] for (W, C, _) in sortedRes]

            elites = population[:self.numberOfSelected]  # NOQA
            remainChild = population[self.numberOfSelected:]  # NOQA
            newGen = elites

            for i in range(0, len(remainChild), 2):
                chromosome0, chromosome1 = self.crossOverAndMutation(remainChild[i], remainChild[i + 1])
                newGen.append(chromosome0)
                newGen.append(chromosome1)

            population = newGen
            gc += 1


if __name__ == '__main__':
    try:
        os.mkdir('./records')
    except FileExistsError:
        pass

    nInputs = 28 * 28
    nHiddens = 10  # NOQA
    nOutputs = 47

    configs = {
        'n': 5,  # Number of chromosome in a population
        'goal': 95,
        'numberOfSelected': 1,
        'numInverting': nHiddens,
        'eta': 1e-3,  # Learning rate
        'bias': 1,
        'epochs': 10,
        'decay': 0.1,  # How much will eta (or the learning rate) decrease / increase
        'frequency': 5,  # How frequently will the learning rate decrease / increase
        'maxInputPerNode': (nInputs + nHiddens) // 3,
        'minInputPerNode': 1,
        'maxOutputPerNode': (nOutputs + nHiddens) // 3,
        'minOutputPerNode': 1,
        'cap': 4000,  # Dataset's Cap
        'nOutputs': nOutputs,  # Number of Output Node(s)
        'nInputs': nInputs,  # Number of Input Feature(s)
        'nHiddens': nHiddens,  # Number of Intermediate Node(s) (or Hidden Node(s))  # NOQA
        'N': nInputs + nOutputs + nHiddens,  # Number of nodes ((Number of Input Nodes + Number of Output Nodes) + Number of Hidden Nodes)
        'crossOverRatio': 0.6,
        'signalProbability': 0.6,  # Probability of 1 Node to mutate
        'mutationProbability': 0.7,  # Probability of 1 Chromosome to mutate
        'invertingProbability': 0.6  # Probability of Node's connections to be inverted
    }

    algorithm = GeneticAlgorithm(configs)
    algorithm.run_algorithm()
