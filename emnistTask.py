import os
import time
import cv2
import numpy as np
import torchvision
import random

from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername


def train(network, trainingDataset, simulationFolderPath):
    network.getActivationFunction()
    time.sleep(1)

    for epoch in range(network.epochs):
        etaW = network.etaW
        etaP = network.etaP
        etaR = network.etaR

        network.save_weight_image_per_epoch(epoch, simulationFolderPath)

        if (epoch + 1) % network.frequency == 0:
            etaW *= network.decay
            etaP *= network.decay
            etaR *= network.decay

        for u, v in tqdm(trainingDataset,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {network.epochs:<2}',
                         colour='green',
                         leave=False,
                         total=network.datasetCap):
            network.zeros_grad()
            network.zeros_io()

            network.predict(u)
            network.update_params(v, etaW, etaP, etaR)


def evaluate(network, evaluatingDataset, simulationFolderPath):
    confusionMatrix = np.zeros((network.nOutputs, network.nOutputs),
                               dtype=np.int64)

    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False):
        predict = np.argmax(network.predict(u))
        target = np.argmax(v)

        confusionMatrix[target, predict] += 1

    np.save(f'./{simulationFolderPath}/confusion_matrix.npy', confusionMatrix)
    return confusionMatrix


def getPrecisionForEachClass(network, confusionMatrix):
    precisionVector = np.zeros((network.nOutputs, 1))

    for outputIdx in range(network.outputIdc.shape[0]):
        precisionVector[outputIdx] = confusionMatrix[outputIdx, outputIdx] \
                                     / np.sum(confusionMatrix[:, outputIdx])

    return precisionVector * 100.


if __name__ == '__main__':
    nInputs = 28 * 28
    nHiddens = 80  # NOQA
    nOutputs = 26
    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    EPOCHS = 24
    datasetCap = nOutputs * 800
    record_no = 85
    nTest = 100

    labels = 'abcdefghijklmnopqrstuvwxyz'
    labels = [c for c in labels]

    trainingClasses = np.zeros((nOutputs,), dtype=np.int64)
    evaluatingClasses = np.zeros((nOutputs,), dtype=np.int64)

    evaluatingDataset = []
    trainingDataset = []
    validationSet = []

    if isTrain:
        trainingMNIST = torchvision.datasets.EMNIST('./data',
                                                    train=True,
                                                    download=True,
                                                    split='letters')

        evaluatingMNIST = torchvision.datasets.EMNIST('./data',
                                                      train=False,
                                                      split='letters')

        # Preprocessing Training Data
        for u, v in tqdm(trainingMNIST,
                         desc='Preprocessing Training Data: ',
                         colour='green'):
            v = v - 1
            if v < nOutputs:
                if trainingClasses[v] < datasetCap // nOutputs:
                    # Target output vector v
                    t = np.zeros((nOutputs,), dtype=np.float64)
                    t[v] = 1

                    u = np.array(u).T

                    if trainingClasses[v] + 1 == 100:
                        cv2.imwrite(f'./training_samples/{v}.jpg', u)

                    u = u / np.max(u)

                    trainingDataset.append([u.flatten(), t])
                    trainingClasses[v] += 1

        # Preprocessing Evaluating Data
        for u, v in tqdm(evaluatingMNIST,
                         desc='Preprocessing Evaluating Data: ',
                         colour='green'):
            # Target output vector v
            v = v - 1
            if v < nOutputs:
                t = np.zeros((nOutputs,), dtype=np.float64)
                t[v] = 1

                u = np.array(u).T

                if evaluatingClasses[v] + 1 == 100:
                    cv2.imwrite(f'./evaluating_samples/{v}.jpg', u)

                u = u / np.max(u)

                evaluatingDataset.append([u.flatten(), t])
                evaluatingClasses[v] += 1

        for _ in range(100):
            random.shuffle(trainingDataset)
            random.shuffle(evaluatingDataset)

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
        minRadius=500, maxRadius=800,
        etaW=1.2, etaP=1.2, etaR=1.2, decay=0.1,
        width=5000, height=5000, depth=5000, bias=-0.5,
        hiddenZoneOffset=1500, outputZoneOffset=500,
        maxInputPerNode=maxInputPerNode // 2, minInputPerNode=maxInputPerNode // 4,
        maxOutputPerNode=maxOutputPerNode // 2, minOutputPerNode=maxOutputPerNode // 4,
        activationFunc='sigmoid', datasetName='emnist', train=isTrain
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        os.mkdir(simulationFolderPath)
        print(f'Using Folder\'s Named: {folderName}')

        if len(np.where(trainingClasses == datasetCap // nOutputs)[0]) == nOutputs:
            startTime = time.perf_counter()  # Start timer

            train(network, trainingDataset, simulationFolderPath)

            confusionMatrix = evaluate(network,
                                       evaluatingDataset,
                                       simulationFolderPath)

            sumOfDiag = np.sum(np.diag(confusionMatrix))
            accuracy = sumOfDiag * 100. / len(evaluatingDataset)

            totalTime = time.perf_counter() - startTime  # End timer

            print(f'Accuracy: {accuracy:.2f}%')
            print(f'Total Established Time: {totalTime} (sec)')
            print(f'Precision:\n{getPrecisionForEachClass(network, confusionMatrix)}')

            network.save_result(simulationFolderPath)
            network.save_config(trainingTime=totalTime,
                                accuracy=accuracy,
                                simulationFolderPath=simulationFolderPath)
            network.plot()
    else:
        network.load_simulation(f'records/3d_nodes_simulation_{record_no}')

        fp = './testA.jpg'
        u = cv2.imread(fp)
        u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

        u = np.array(u) / np.max(u)
        u = u.flatten()

        print(f'Testing {fp}')
        res = np.zeros((network.nOutputs,), dtype=np.int64)

        for i in range(nTest):
            predict = network.predict(u)
            res[np.argmax(predict)] += 1

        print(f'Prediction: {labels[np.argmax(res)]} '
              f'with confidence: {res[np.argmax(res)] * 100. / nTest}%')
