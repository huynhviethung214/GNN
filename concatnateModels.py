import os
import time
import cv2
import numpy as np
import torchvision

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


def evaluate(network,
             evaluatingDataset,
             networkFolderPath,
             networkIdx):
    confusionMatrix = np.zeros((network.nOutputs, network.nOutputs),
                               dtype=np.int64)

    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False):
        if nOutputs * networkIdx <= v <= nOutputs * (networkIdx + 1) - 1:
            predict = np.argmax(network.predict(u)) + nOutputs * networkIdx
            target = np.argmax(v)

            confusionMatrix[target, predict] += 1

    np.save(f'./{networkFolderPath}/confusion_matrix.npy', confusionMatrix)
    return confusionMatrix


def getPrecisionForEachClass(network, confusionMatrix):
    precisionVector = np.zeros((network.nOutputs, 1))

    for outputIdx in range(network.outputIdc.shape[0]):
        precisionVector[outputIdx] = confusionMatrix[outputIdx, outputIdx] \
                                     / np.sum(confusionMatrix[:, outputIdx])

    return precisionVector * 100.


if __name__ == '__main__':
    nClasses = 26
    labels = 'abcdefghijklmnopqrstuvwxyz'
    labels = [c for c in labels]

    nInputs = 28 * 28
    nHiddens = 100  # NOQA

    isTrain = True
    EPOCHS = 12
    record_no = 78
    nTest = 100

    numberOfNetworks = 2
    networks = []

    nOutputs = nClasses // numberOfNetworks
    datasetCap = nOutputs * 800

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    evaluatingDataset = []
    trainingDataset = []
    validationSet = []

    folderName = get_foldername()
    simulationFolderPath = f'./records/{folderName}'
    os.mkdir(simulationFolderPath)
    print(f'Using Folder\'s Named: {folderName}')

    networkFolderPaths = []

    if isTrain:
        trainingMNIST = torchvision.datasets.EMNIST('./data',
                                                    train=True,
                                                    download=True,
                                                    split='letters')

        evaluatingMNIST = torchvision.datasets.EMNIST('./data',
                                                      train=False,
                                                      split='letters')

        evaluatingClasses = np.zeros((nClasses,), dtype=np.int64)
        # Preprocessing Evaluating Data
        for u, v in tqdm(evaluatingMNIST,
                         desc='Preprocessing Evaluating Data: ',
                         colour='green'):
            # Target output vector v
            v = v - 1
            t = np.zeros((nClasses,), dtype=np.float64)
            t[v] = 1

            u = np.array(u).T

            if evaluatingClasses[v] + 1 == 100:
                cv2.imwrite(f'./evaluating_samples/{v}.jpg', u)

            u = u / np.max(u)

            evaluatingDataset.append([u.flatten(), t])
            evaluatingClasses[v] += 1

        for networkIdx in range(numberOfNetworks):
            trainingClasses = np.zeros((nOutputs,), dtype=np.int64)
            subTrainingDataset = []

            # Preprocessing Training Data
            for u, v in tqdm(trainingMNIST,
                             desc='Preprocessing Training Data: ',
                             colour='green'):
                v = v - 1 - nOutputs * networkIdx
                if nOutputs * networkIdx <= v <= nOutputs * (networkIdx + 1) - 1:
                    if trainingClasses[v] < datasetCap // nOutputs:
                        # Target output vector v
                        t = np.zeros((nOutputs,), dtype=np.float64)
                        t[v] = 1

                        u = np.array(u)

                        if trainingClasses[v] + 1 == 100:
                            cv2.imwrite(f'./training_samples/{v}.jpg', u)

                        u = u / np.max(u)

                        subTrainingDataset.append([u.flatten(), t])
                        trainingClasses[v] += 1

            network = Network(
                nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
                epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
                minRadius=100, maxRadius=300,
                etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=0.1,
                width=3000, height=3000, depth=3000, bias=-0.5,
                hiddenZoneOffset=1000, outputZoneOffset=300,
                maxInputPerNode=maxInputPerNode, minInputPerNode=maxInputPerNode // 2,
                maxOutputPerNode=maxOutputPerNode, minOutputPerNode=maxOutputPerNode // 2,
                activationFunc='sigmoid', datasetName='emnist', train=isTrain
            )

            networkFolderPath = simulationFolderPath + f'/net_{networkIdx}'
            networkFolderPaths.append(networkFolderPath)
            os.mkdir(networkFolderPath)

            trainingDataset.append(subTrainingDataset)
            networks.append(network)

    if isTrain:
        startTime = time.perf_counter()  # Start timer

        for networkIdx in range(len(networks)):
            network = networks[networkIdx]
            networkFolderPath = networkFolderPaths[networkIdx]

            train(network,
                  trainingDataset[networkIdx],
                  networkFolderPath)

            confusionMatrix = evaluate(network,
                                       evaluatingDataset,
                                       networkFolderPath,
                                       networkIdx)

            sumOfDiag = np.sum(np.diag(confusionMatrix))
            accuracy = sumOfDiag * 100. / len(evaluatingDataset)

            totalTime = time.perf_counter() - startTime  # End timer

            print(f'Accuracy: {accuracy:.2f}%')
            print(f'Total Established Time: {totalTime} (sec)')
            print(f'Precision:\n{getPrecisionForEachClass(network, confusionMatrix)}')

            network.save_result(networkFolderPath)
            network.save_config(trainingTime=totalTime,
                                accuracy=accuracy,
                                simulationFolderPath=networkFolderPath)
            network.plot()
    # else:
    #     for networkIdx in range(numberOfNetworks):
    #         networkFolderPath = f'records/3d_nodes_simulation_{record_no}/net_{networkIdx}'
    #         network.load_simulation()
    #
    #     fp = './testA.jpg'
    #     u = cv2.imread(fp)
    #     u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)
    #
    #     u = np.array(u) / np.max(u)
    #     u = u.flatten()
    #
    #     print(f'Testing {fp}')
    #     res = np.zeros((network.nOutputs,), dtype=np.int64)
    #
    #     for i in range(nTest):
    #         predict = network.predict(u)
    #         # print(f'Prediction {i}-th: {np.argmax(predict)}')
    #         res[np.argmax(predict)] += 1
    #
    #     # print(f'Prediction: {np.argmax(res)} '
    #     #       f'with confidence: {res[np.argmax(res)] * 100. / nTest}%')
    #     print(f'Prediction: {labels[np.argmax(res)]} '
    #           f'with confidence: {res[np.argmax(res)] * 100. / nTest}%')
