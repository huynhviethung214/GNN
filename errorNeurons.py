import os
import time
import numpy as np
from numba import njit, prange

from tqdm import tqdm
from models.GNN_4 import Network
from getSimulationFolderName import get_foldername
from processFraEng import prepareData
from sequentialPredictionForTranslationTask import binaryToStr


def train(network0, network1, trainingDataset,
          network0FolderPath, network1FolderPath,
          inputPerNetwork0, hiddenPerNetwork0, outputPerNetwork0,
          numSubNetworks0):
    time.sleep(1)

    etaW0 = network0.etaW
    etaP0 = network0.etaP
    etaR0 = network0.etaR

    etaW1 = network1.etaW
    etaP1 = network1.etaP
    etaR1 = network1.etaR

    for epoch in range(network0.epochs):
        network0.save_weight_image_per_epoch(epoch, network0FolderPath)
        network1.save_weight_image_per_epoch(epoch, network1FolderPath)

        if (epoch + 1) % network0.frequency == 0:
            etaW0 *= network0.decay
            etaP0 *= network0.decay
            etaR0 *= network0.decay

        if (epoch + 1) % network1.frequency == 0:
            etaW1 *= network1.decay
            etaP1 *= network1.decay
            etaR1 *= network1.decay

        previousLoss = np.Inf

        for u, v in tqdm(trainingDataset,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {network0.epochs:<2}',
                         colour='green',
                         leave=False,
                         total=network0.datasetCap):
            network0.zeros_grad()
            network1.zeros_grad()

            predict = network0.predict(u, inputPerNetwork0, hiddenPerNetwork0,
                                       outputPerNetwork0, numSubNetworks0)

            # Calculate loss using error-neurons
            concatenatedPredictV = np.concatenate((predict, v))
            loss = network1.predict(concatenatedPredictV, network1.nInputs,
                                    network1.nHiddens, network1.nOutputs, 1)

            AL = (1 / loss.shape[0]) * np.sum(loss)  # Average Loss
            if AL < previousLoss:
                previousLoss = AL
                network1.update_params(loss, etaW1, etaP1, etaR1)
            else:
                network1.update_params(np.zeros(loss.shape,
                                               dtype=np.float64),
                                       etaW1, etaP1, etaR1)
            network0.update_params(loss, etaW0, etaP0, etaR0)


@njit()
def getAccuracy(u, v):
    corrects = 0

    for i in prange(len(u)):
        if u[i] == v[i] and u[i] != 0:
            corrects += 1

    return corrects * 100 / u.shape[0]


def evaluate(network, evaluatingDataset,
             maxLenEng, engWordInBinaryLength,
             output_lang, inputPerNetwork0, hiddenPerNetwork0,
             outputPerNetwork0, numSubNetworks0):
    averageAccuracy = 0

    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False):
        predict = network.predict(u, inputPerNetwork0, hiddenPerNetwork0,
             outputPerNetwork0, numSubNetworks0).reshape((maxLenEng, engWordInBinaryLength))
        predict[predict > 0.9] = 1.
        predict[predict <= 0.9] = 0.

        v = v.reshape((maxLenEng, engWordInBinaryLength))

        _, predict = binaryToStr(output_lang, predict)
        _, v = binaryToStr(output_lang, v)

        # print(np.array(predict).flatten(), np.array(v).flatten())
        averageAccuracy += getAccuracy(np.array(predict).flatten(),
                                       np.array(v).flatten())

    return averageAccuracy / len(evaluatingDataset)


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    engWordInBinaryLength = len(np.binary_repr(max(list(output_lang.word2count.values()))))
    fraWordInBinaryLength = len(np.binary_repr(max(list(input_lang.word2count.values()))))

    maxLenFra = 0
    maxLenEng = 0
    dataset = []

    for i in tqdm(range(len(pairs)),
                  desc='Packing Pairs',
                  total=len(pairs)):
        pair = pairs[i]
        pairs[i][0] = pairs[i][0].replace(' .', '')
        pairs[i][1] = pairs[i][1].replace(' .', '')

        vectorizedFra = []
        vectorizedEng = []

        for word in pair[0].split(' '):
            vectorizedFra.append(input_lang.word2index[word])

        for word in pair[1].split(' '):
            vectorizedEng.append(output_lang.word2index[word])

        dataset.append([np.array(vectorizedFra, dtype=np.int64),
                        np.array(vectorizedEng, dtype=np.int64)])

        if dataset[i][0].shape[0] > maxLenFra:
            maxLenFra = dataset[i][0].shape[0]

        if dataset[i][1].shape[0] > maxLenEng:
            maxLenEng = dataset[i][1].shape[0]

    for i in tqdm(range(len(dataset)),
                  desc='Vectorized Data',
                  total=len(dataset)):
        vectorizedBinaryEng = np.zeros((maxLenEng, engWordInBinaryLength))
        vectorizedBinaryFra = np.zeros((maxLenFra, fraWordInBinaryLength))

        for j, ele in enumerate(dataset[i][0]):
            vectorizedBinaryFra[j] = np.array(
                [int(b) for b in np.binary_repr(ele, width=fraWordInBinaryLength)]
            )

        for j, ele in enumerate(dataset[i][1]):
            vectorizedBinaryEng[j] = np.array(
                [int(b) for b in np.binary_repr(ele, width=engWordInBinaryLength)]
            )

        dataset[i][0] = vectorizedBinaryFra.flatten()
        dataset[i][1] = vectorizedBinaryEng.flatten()

    nInputs0 = dataset[0][0].shape[0]
    nHiddens0 = maxLenFra * 50
    nOutputs0 = dataset[0][1].shape[0]

    nHiddens1 = 50
    nOutputs1 = nOutputs0

    maxInputPerNode0 = nInputs0 + nHiddens0
    maxOutputPerNode0 = nOutputs0 + nHiddens0

    maxInputPerNode1 = nOutputs0 + nHiddens1
    maxOutputPerNode1 = nOutputs1 + nHiddens1

    isTrain = True
    EPOCHS = 1
    datasetCap = len(dataset)
    record_no = 97
    nTest = 100

    trainingSet = dataset[:datasetCap]
    evaluatingSet = dataset[datasetCap:]

    network0 = Network(
        nInputs=nInputs0, nHiddens=nHiddens0, nOutputs=nOutputs0,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=1,
        minRadius=500, maxRadius=800,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=0.1,
        width=5000, height=5000, depth=5000, bias=-0.8,
        hiddenZoneOffset=1000, outputZoneOffset=1000,
        maxInputPerNode=maxInputPerNode0, minInputPerNode=maxInputPerNode0 // 2,
        maxOutputPerNode=maxOutputPerNode0, minOutputPerNode=maxOutputPerNode0 // 2,
        activationFunc='sigmoid', datasetName='fra-eng', train=isTrain
    )

    network1 = Network(
        nInputs=nOutputs0 * 2, nHiddens=50, nOutputs=nOutputs0,
        epochs=0, datasetCap=0, frequency=4,
        minRadius=50, maxRadius=80,
        etaW=6e-3, etaP=6e-3, etaR=6e-3, decay=0.1,
        width=500, height=500, depth=500, bias=-0.5,
        hiddenZoneOffset=120, outputZoneOffset=50,
        maxInputPerNode=maxInputPerNode1, minInputPerNode=maxInputPerNode1 // 2,
        maxOutputPerNode=maxOutputPerNode1, minOutputPerNode=maxOutputPerNode1 // 2,
        activationFunc='sin', datasetName='fra-eng', train=isTrain
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        network0FolderPath = f'./records/{folderName}/network0'
        network1FolderPath = f'./records/{folderName}/network1'

        os.mkdir(simulationFolderPath)
        os.mkdir(network0FolderPath)
        os.mkdir(network1FolderPath)

        startTime = time.perf_counter()

        network0.getActivationFunction()
        network1.getActivationFunction()

        train(network0, network1, trainingSet,
              network0FolderPath, network1FolderPath,
              fraWordInBinaryLength, nHiddens0 // maxLenFra,
              maxLenEng, maxLenFra)

        accuracy = evaluate(network0, trainingSet, maxLenEng,
                            engWordInBinaryLength, output_lang,
                            fraWordInBinaryLength, nHiddens0 // maxLenFra,
                            maxLenEng, maxLenFra)
        print(f'Accuracy: {accuracy:.2f}%')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')

        network0.save_result(network0FolderPath)
        network0.save_config(trainingTime=totalTime,
                             accuracy=accuracy,
                             simulationFolderPath=network0FolderPath)
        network0.plot()

        network1.save_result(network1FolderPath)
        network1.save_config(trainingTime=totalTime,
                             accuracy=0.0,
                             simulationFolderPath=network1FolderPath)
        network1.plot()
    else:
        network0.load_simulation(f'records/3d_nodes_simulation_{record_no}/network0')
        # i = 400
        for i in range(len(trainingSet)):
            if i == 400:
                break

            fraWords = trainingSet[i][0]
            engWords = trainingSet[i][1].reshape((maxLenEng, engWordInBinaryLength))

            p = network0.predict(fraWords, fraWordInBinaryLength,
                                 nHiddens0 // maxLenFra, maxLenEng, maxLenFra)\
                .reshape((maxLenEng, engWordInBinaryLength))
            print(p)
            p[p > 0.9] = 1.
            p[p <= 0.9] = 0.

            engPredict = binaryToStr(output_lang, p)[0]

            print(f'Target: {binaryToStr(output_lang, engWords)[0]:<60} '
                  f'| Predict: {engPredict:<60}')
