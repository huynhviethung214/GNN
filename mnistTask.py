import os
import time
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from scipy.ndimage import zoom, rotate
from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername


def train(network, trainingDataset, simulationFolderPath):
    time.sleep(1)
    epochs = network.epochs

    etaW = network.etaW
    etaB = network.etaB
    etaP = network.etaP
    etaR = network.etaR

    for epoch in range(epochs):
        network.save_weight_image_per_epoch(epoch, simulationFolderPath)

        if (epoch + 1) % network.frequency == 0:
            etaW *= network.decay
            etaB *= network.decay
            etaP *= network.decay
            etaR *= network.decay

        for u, v in tqdm(trainingDataset,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {epochs:<2}',
                         colour='green',
                         leave=False,
                         total=network.datasetCap):
            network.zeros_grad()
            p = network.predict(u)
            network.losses.append(network.g(p, v, network.nOutputs))
            network.update_params(v, etaW, etaB, etaP, etaR)

            # network.W = removeSynapses(network.W, network.P, network.R,
            #                            network.nodeIdc, network.nInputs,
            #                            network.nOutputs)


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
        if confusionMatrix[outputIdx, outputIdx] != 0.:
            precisionVector[outputIdx] = confusionMatrix[outputIdx, outputIdx] \
                                         / np.sum(confusionMatrix[:, outputIdx])
        else:
            precisionVector[outputIdx] = 0.

    return precisionVector * 100.


if __name__ == '__main__':
    inputShape = [28, 28]
    nInputs = inputShape[0] * inputShape[1]
    nHiddens = 300  # NOQA
    nOutputs = 10

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    transformData = True
    DEBUG = False

    # fp = './test1.jpg'
    EPOCHS = 8
    datasetCap = nOutputs * 600
    record_no = 77
    nTest = 100

    # Preprocessing parameters
    minScalingFactor = 1.2
    maxScalingFactor = 1.4

    minRotateX = -20
    maxRotateX = 20

    P = 0.7
    numberOfPixelToJitter = 50
    minJitterBrightness = 0.3
    maxJitterBrightness = 0.6

    trainingClasses = np.zeros((nOutputs,), dtype=np.int64)
    evaluatingClasses = np.zeros((nOutputs,), dtype=np.int64)

    evaluatingDataset = []
    trainingDataset = []

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
        minRadius=1000, maxRadius=3000, etaW=1e-2, etaP=1e-2,
        etaB=1e-2, etaR=1e-2, decay=0.1,
        width=8000, height=8000, depth=8000, minBias=0.1, maxBias=0.8,
        hiddenZoneOffset=2000, outputZoneOffset=1000,
        maxInputPerNode=maxInputPerNode // 3, minInputPerNode=maxInputPerNode // 6,
        maxOutputPerNode=maxOutputPerNode // 3, minOutputPerNode=maxOutputPerNode // 6,
        activationFunc='sigmoid', lossFunc='mse', datasetName='mnist', train=isTrain
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        trainingSamplesFolderPath = f'{simulationFolderPath}/training_samples'
        evaluatingSamplesFolderPath = f'{simulationFolderPath}/evaluating_samples'

        os.mkdir(simulationFolderPath)
        os.mkdir(trainingSamplesFolderPath)
        os.mkdir(evaluatingSamplesFolderPath)

        print(f'Using Folder\'s Named: {folderName}')

        trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=True)
        evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)

        # Preprocessing Training Data
        for u, v in tqdm(trainingMNIST,
                         desc='Preprocessing Training Data:',
                         colour='green'):
            # Target output vector v
            t = np.zeros((nOutputs,), dtype=np.float64)
            t[v] = 1.

            u = np.array(u)
            if trainingClasses[v] + 1 == 10:
                cv2.imwrite(f'./training_samples/{v}.jpg', u)
            u = u / 255.

            if transformData:
                if np.random.uniform(0, 1, 1)[0] <= P:
                    u[np.random.randint(0, u.shape[0], numberOfPixelToJitter),
                    np.random.randint(0, u.shape[1], numberOfPixelToJitter)] \
                        += np.random.uniform(minJitterBrightness, maxJitterBrightness, numberOfPixelToJitter)
                    u[u > 0.6] = 1.

                if np.random.uniform(0, 1, 1)[0] <= P:
                    scalingFactor = np.random.uniform(minScalingFactor, maxScalingFactor, 1)[0]
                    scalingFactor = np.round(scalingFactor, 1)

                    u = zoom(u, scalingFactor)

                if np.random.uniform(0, 1, 1)[0] <= P:
                    u = rotate(u, np.random.randint(minRotateX, maxRotateX, 1)[0])

                if u.shape[0] != inputShape[0] and u.shape[1] != inputShape[1]:
                    if u.shape[0] % 2 != 0:
                        u = u[0: u.shape[0] - 1, 0: u.shape[1] - 1]

                    marginX = (u.shape[0] - inputShape[0]) // 2
                    marginY = (u.shape[1] - inputShape[1]) // 2

                    u = u[marginX: u.shape[0] - marginX, marginY: u.shape[1] - marginY]
                    u[u > 0.4] = 1.

                if DEBUG:
                    plt.imshow(u, cmap='gray')
                    plt.show()

            if trainingClasses[v] < datasetCap // nOutputs:
                if trainingClasses[v] == 10:
                    plt.imsave(f'{trainingSamplesFolderPath}/{v}.jpg', u, cmap='gray')

                trainingDataset.append([u.flatten(), t])
                trainingClasses[v] += 1
            # else:
            #     evaluatingDataset.append([u.flatten(), t])

        # Preprocessing Evaluating Data
        for u, v in tqdm(evaluatingMNIST,
                         desc='Preprocessing Evaluating Data: ',
                         colour='green'):
            if evaluatingClasses[v] == 10:
                plt.imsave(f'{evaluatingSamplesFolderPath}/{v}.jpg', u, cmap='gray')

            # Target output vector v
            t = np.zeros((nOutputs,), dtype=np.float64)
            t[v] = 1.

            u = np.array(u)

            if evaluatingClasses[v] + 1 == 10:
                cv2.imwrite(f'./evaluating_samples/{v}.jpg', u)

            u = u / 255.

            evaluatingDataset.append([u.flatten(), t])
            evaluatingClasses[v] += 1

        for _ in range(100):
            random.shuffle(trainingDataset)
            random.shuffle(evaluatingDataset)

        if len(np.where(trainingClasses == datasetCap // nOutputs)[0]) == nOutputs:
            startTime = time.perf_counter()  # Start timer

            train(network, trainingDataset, simulationFolderPath)
            confusionMatrix = evaluate(network,
                                       evaluatingDataset,
                                       simulationFolderPath)

            sumOfDiagonal = np.sum(np.diag(confusionMatrix))
            accuracy = sumOfDiagonal * 100. / len(evaluatingDataset)

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
        network.getActivationFunction()
        # network.plot()

        # u = cv2.imread(fp)
        # u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)
        #
        # u = np.array(u) / np.max(u)
        # u = u.flatten()
        #
        # print(f'Testing {fp}')
        # res = np.zeros((network.nOutputs,), dtype=np.int64)
        #
        # for i in range(nTest):
        #     predict = network.predict(u)
        #     res[np.argmax(predict)] += 1
        #
        # confidences = res * 100. / nTest
        # for i, res in enumerate(confidences):
        #     print(f'Prediction: {i} with confidence: {res}%')
        # print(f'Best Prediction: {np.argmax(confidences)}')

        # Evaluate with external data
        fp = './external data'
        for image in os.listdir(fp):
            imagePath = f'{fp}/{image}'
            label = image.split('.')[0]

            if ' ' in label:
                label = label.split(' ')[0]

            u = cv2.imread(imagePath)
            u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

            u = np.array(u) / np.max(u)
            u = u.flatten()

            res = np.zeros((network.nOutputs,), dtype=np.int64)

            for i in range(nTest):
                predict = network.predict(u)
                res[np.argmax(predict)] += 1

            # predict = network.predict(u)
            print(f'Best Prediction: {np.argmax(res)} | Target: {label:<10} |'
                  f' Image: {imagePath:<40}')
