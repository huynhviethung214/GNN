import os
import time
import cv2
import numpy as np

from sklearn.datasets import load_digits
from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername


def train(network, trainingDataset, simulationFolderPath):
    time.sleep(1)

    for epoch in range(network.epochs):
        etaW = network.etaW
        etaB = network.etaB
        etaP = network.etaP
        etaR = network.etaR

        network.save_weight_image_per_epoch(epoch, simulationFolderPath)

        if (epoch + 1) % network.frequency == 0:
            etaW *= network.decay
            etaB *= network.decay
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
            network.update_params(v, etaW, etaB, etaP, etaR)


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
    nInputs = 8 * 8
    nHiddens = 80  # NOQA
    nOutputs = 10

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    EPOCHS = 100
    record_no = 78
    nTest = 100

    dataset = []
    datasetForRecord = []
    classes = np.zeros((nOutputs,))

    folderName = get_foldername()
    simulationFolderPath = f'./records/{folderName}'
    os.mkdir(simulationFolderPath)
    print(f'Using Folder\'s Named: {folderName}')

    digits = load_digits()
    datasetCap = nOutputs * 80

    trainingDataset = []
    evaluatingDataset = []

    if isTrain:
        # Preprocessing Training Data
        for i in tqdm(range(len(digits.images)),
                      desc='Preprocessing Training Data: ',
                      colour='green'):
            u = np.array(digits.images[i], dtype=np.float64)
            # u = (u - np.min(u)) / (np.max(u) - np.min(u))

            t = np.zeros((nOutputs,), dtype=np.float64)
            t[digits.target[i]] = 1.

            if classes[digits.target[i]] == 0:
                datasetForRecord.append([u.flatten(), digits.target[i]])

            if classes[digits.target[i]] < datasetCap // nOutputs:
                trainingDataset.append([u.flatten(), t])
                classes[digits.target[i]] += 1
            else:
                evaluatingDataset.append([u.flatten(), t])

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=80,
        minRadius=10, maxRadius=50,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=0.1,
        width=150, height=150, depth=150, minBias=0.5, maxBias=0.6,
        hiddenZoneOffset=40, outputZoneOffset=30,
        maxInputPerNode=maxInputPerNode, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode, minOutputPerNode=maxOutputPerNode // 2,
        activationFunc='sigmoid', lossFunc='mse', datasetName='optical_mnist',
        train=isTrain, isSequential=False
    )

    if isTrain:
        startTime = time.perf_counter()  # Start timer

        train(network, trainingDataset, simulationFolderPath)
        confusionMatrix = evaluate(network,
                                   evaluatingDataset,
                                   simulationFolderPath)
        print(confusionMatrix)

        sumOfDiag = np.sum(np.diag(confusionMatrix))
        accuracy = sumOfDiag * 100. / len(evaluatingDataset)

        totalTime = time.perf_counter() - startTime  # End timer

        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Total Established Time: {totalTime} (sec)')
        print(f'Precision:\n{getPrecisionForEachClass(network, confusionMatrix)}')

        for u, v in datasetForRecord:
            network.zeros_io()
            network.predict(u)
            network.record_output_of_hidden_neurons(simulationFolderPath, v)

        network.save_result(simulationFolderPath)
        network.save_config(trainingTime=totalTime,
                            accuracy=accuracy,
                            simulationFolderPath=simulationFolderPath)
        network.plot()
    else:
        network.load_simulation(f'records/3d_nodes_simulation_{record_no}')

        fp = './test1.jpg'
        u = cv2.imread(fp)
        u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

        u = np.array(u) / np.max(u)
        u = u.flatten()

        print(f'Testing {fp}')
        res = np.zeros((network.nOutputs,), dtype=np.int64)

        for i in range(nTest):
            predict = network.predict(u)
            res[np.argmax(predict)] += 1

        print(f'Prediction: {np.argmax(res)} '
              f'with confidence: {res[np.argmax(res)] * 100. / nTest}%')
