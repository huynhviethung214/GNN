import os
import time
import cv2
import numpy as np
import torchvision
import random

from scipy.ndimage import zoom, rotate
from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername


def train(network0, network1, trainingDataset,
          network0FolderPath, network1FolderPath):
    time.sleep(1)

    hiddenNodesWeights = np.zeros((network0.N, network0.N),
                                  dtype=np.float64)

    # etaW2 = network2.etaW
    # etaP2 = network2.etaP
    # etaR2 = network2.etaR

    for epoch in range(network0.epochs):
        etaW0 = network0.etaW
        etaB0 = network0.etaB
        etaP0 = network0.etaP
        etaR0 = network0.etaR

        etaW1 = network1.etaW
        etaB1 = network1.etaB
        etaP1 = network1.etaP
        etaR1 = network1.etaR

        network0.save_weight_image_per_epoch(epoch, network0FolderPath)
        network1.save_weight_image_per_epoch(epoch, network1FolderPath)
        # network2.save_weight_image_per_epoch(epoch, network2FolderPath)

        if (epoch + 1) % network0.frequency == 0:
            etaW0 *= network0.decay
            etaB0 *= network0.decay
            etaP0 *= network0.decay
            etaR0 *= network0.decay

        if (epoch + 1) % network1.frequency == 0:
            etaW1 *= network1.decay
            etaB1 *= network1.decay
            etaP1 *= network1.decay
            etaR1 *= network1.decay

        for u, v in tqdm(trainingDataset,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {network0.epochs:<2}',
                         colour='green',
                         leave=False,
                         total=network0.datasetCap):
            network0.zeros_grad()
            network1.zeros_grad()

            # Before apply instructions
            predict = network0.predict(u)
            loss0 = network0.g(predict, v, network0.nOutputs)

            # Get instructions
            L2 = np.abs(predict - v)
            instructions = network1.predict(L2)
            instructions[instructions > 0] = 1.
            instructions[instructions < 0] = -1.

            # Disabling neuron if instruction return -1
            disabledHiddenNodes = np.array(np.where(instructions == -1)[0]) \
                                  + network0.nInputs

            hiddenNodesWeights[disabledHiddenNodes] = network0.W[disabledHiddenNodes]
            hiddenNodesWeights[:, disabledHiddenNodes] = network0.W[:, disabledHiddenNodes]

            network0.W[disabledHiddenNodes] = 0
            network0.W[:, disabledHiddenNodes] = 0

            # Enabling neuron if instruction return 1
            enabledHiddenNodes = np.array(np.where(instructions == 1)[0]) \
                                 + network0.nInputs
            network0.W[enabledHiddenNodes] = hiddenNodesWeights[enabledHiddenNodes]
            network0.W[:, enabledHiddenNodes] = hiddenNodesWeights[:, enabledHiddenNodes]

            # etas = network2.predict(np.concatenate((predict, instructions)))
            # etaW0, etaW1, etaP0, etaP1, etaR0, etaR1 = list(etas)

            # After apply instructions
            predict = network0.predict(u)
            loss1 = network0.g(predict, v, network0.nOutputs)

            if loss0 < loss1:
                network1.update_params(instructions, etaW1, etaB1, etaP1, etaR1)
                # network2.update_params(etas, etaW0, etaP0, etaR0)
            else:
                network1.update_params(np.zeros(instructions.shape, dtype=np.float64),
                                       etaW1, etaB1, etaP1, etaR1)
                # network2.update_params(np.zeros(etas.shape, dtype=np.float64),
                #                        etaW2, etaP2, etaR2)

            network0.update_params(v, etaW0, etaB0, etaP0, etaR0)


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
    nInputs0 = 28 * 28

    nHiddens0 = 80  # NOQA
    nHiddens1 = 160  # NOQA

    nOutputs0 = 10
    nOutputs1 = nHiddens0

    maxInputPerNode0 = nInputs0 + nHiddens0
    maxOutputPerNode0 = nOutputs0 + nHiddens0

    maxInputPerNode1 = nOutputs0 + nHiddens1
    maxOutputPerNode1 = nOutputs1 + nHiddens1

    isTrain = True
    transformData = True

    EPOCHS = 24
    datasetCap = nOutputs0 * 800
    record_no = 33
    nTest = 100

    # Preprocessing parameters
    minScalingFactor = 1.5
    maxScalingFactor = 1.8

    minRotateX = -20
    maxRotateX = 20

    P = 0.7
    numberOfPixelToJitter = 50
    maxJitterBrightness = 0.5

    trainingClasses = np.zeros((nOutputs0,), dtype=np.int64)
    evaluatingClasses = np.zeros((nOutputs0,), dtype=np.int64)

    evaluatingDataset = []
    trainingDataset = []
    validationSet = []

    if isTrain:
        trainingMNIST = torchvision.datasets.MNIST('./data', train=True, download=True)
        evaluatingMNIST = torchvision.datasets.MNIST('./data', train=False)

        # Preprocessing Training Data
        for u, v in tqdm(trainingMNIST,
                         desc='Preprocessing Training Data: ',
                         colour='green'):
            # Target output vector v
            t = np.zeros((nOutputs0,), dtype=np.float64)
            t[v] = 1

            u = np.array(u)

            if trainingClasses[v] + 1 == 100:
                cv2.imwrite(f'./training_samples/{v}.jpg', u)

            u = u / np.max(u)

            if transformData:
                if np.random.uniform(0, 1, 1)[0] <= P:
                    originalShape = u.shape

                    scalingFactor = np.random.uniform(minScalingFactor, maxScalingFactor, 1)[0]
                    scalingFactor = np.round(scalingFactor, 1)

                    u = zoom(u, scalingFactor, mode='mirror')

                    marginX = (u.shape[0] - originalShape[0]) // 2
                    marginY = (u.shape[1] - originalShape[1]) // 2

                    u = u[marginX: u.shape[0] - marginX, marginY: u.shape[1] - marginY]
                    u[u > 0.6] = 1.

                if np.random.uniform(0, 1, 1)[0] <= P:
                    u = rotate(u, np.random.randint(minRotateX, maxRotateX, 1)[0])

            if trainingClasses[v] < datasetCap // nOutputs0:
                trainingDataset.append([u.flatten(), t])
                trainingClasses[v] += 1
            else:
                evaluatingDataset.append([u.flatten(), t])

        # Preprocessing Evaluating Data
        for u, v in tqdm(evaluatingMNIST,
                         desc='Preprocessing Evaluating Data: ',
                         colour='green'):
            # Target output vector v
            t = np.zeros((nOutputs0,), dtype=np.float64)
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

    network0 = Network(
        nInputs=nInputs0, nHiddens=nHiddens0, nOutputs=nOutputs0,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=8,
        minRadius=4000, maxRadius=6000,
        etaW=1e-0, etaB=1e-0, etaP=1e-0, etaR=1e-0, decay=0.1,
        width=8000, height=8000, depth=8000, minBias=0.1, maxBias=0.8,
        hiddenZoneOffset=1000, outputZoneOffset=1000,
        maxInputPerNode=maxInputPerNode0 // 1, minInputPerNode=maxInputPerNode0 // 2,
        maxOutputPerNode=maxOutputPerNode0 // 1, minOutputPerNode=maxOutputPerNode0 // 2,
        activationFunc='sigmoid', lossFunc='mse', datasetName='mnist', train=isTrain
    )

    network1 = Network(
        nInputs=nOutputs0, nHiddens=nHiddens1, nOutputs=nOutputs1,
        epochs=0, datasetCap=0, frequency=8,
        minRadius=50, maxRadius=80,
        etaW=6e-1, etaB=6e-1, etaP=6e-1, etaR=6e-1, decay=0.1,
        width=500, height=500, depth=500, minBias=0.1, maxBias=0.8,
        hiddenZoneOffset=120, outputZoneOffset=50,
        maxInputPerNode=maxInputPerNode1, minInputPerNode=maxInputPerNode1 // 2,
        maxOutputPerNode=maxOutputPerNode1, minOutputPerNode=maxOutputPerNode1 // 2,
        activationFunc='tanh', lossFunc='mse', datasetName='mnist', train=isTrain
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        network0FolderPath = f'./records/{folderName}/network0'
        network1FolderPath = f'./records/{folderName}/network1'

        os.mkdir(simulationFolderPath)
        os.mkdir(network0FolderPath)
        os.mkdir(network1FolderPath)

        if len(np.where(trainingClasses == datasetCap // nOutputs0)[0]) == nOutputs0:
            startTime = time.perf_counter()  # Start timer

            train(network0, network1, trainingDataset,
                  network0FolderPath, network1FolderPath)

            confusionMatrix = evaluate(network0,
                                       evaluatingDataset,
                                       simulationFolderPath)
            sumOfDiag = np.sum(np.diag(confusionMatrix))
            accuracy = sumOfDiag * 100. / len(evaluatingDataset)

            totalTime = time.perf_counter() - startTime  # End timer

            print(f'Accuracy: {accuracy:.2f}%')
            print(f'Total Established Time: {totalTime} (sec)')
            print(f'Precision:\n{getPrecisionForEachClass(network0, confusionMatrix)}')

            network0.save_result(network0FolderPath)
            network0.save_config(trainingTime=totalTime,
                                 accuracy=accuracy,
                                 simulationFolderPath=network0FolderPath)
            network0.plot()

            network0.save_result(network1FolderPath)
            network1.save_config(trainingTime=totalTime,
                                 accuracy=0,
                                 simulationFolderPath=network1FolderPath)
            network1.plot()
    else:
        network0.load_simulation(f'records/3d_nodes_simulation_{record_no}/network0')
        network0.getActivationFunction()

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

            res = np.zeros((network0.nOutputs,), dtype=np.int64)

            for i in range(nTest):
                predict = network0.predict(u)
                res[np.argmax(predict)] += 1

            # predict = network.predict(u)
            print(f'Best Prediction: {np.argmax(res)} | Target: {label:<10} |'
                  f' Image: {imagePath:<40}')
