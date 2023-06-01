import json
import os
import time
import numpy as np

from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername
from preprocessingTranslationDataset import Lang


def train(network0, network1, trainingSet, network0FolderPath, network1FolderPath):
    time.sleep(1)

    for epoch in range(network0.epochs):
        previousLoss = np.inf

        etaW0 = network0.etaW
        etaB0 = network0.etaB
        etaP0 = network0.etaP
        etaR0 = network0.etaR

        etaW1 = network0.etaW
        etaB1 = network0.etaB
        etaP1 = network0.etaP
        etaR1 = network0.etaR

        network0.save_weight_image_per_epoch(epoch, network0FolderPath)
        network1.save_weight_image_per_epoch(epoch, network1FolderPath)

        if (epoch + 1) % network0.frequency == 0:
            etaW0 *= network0.decay
            etaB0 *= network0.decay
            etaP0 *= network0.decay
            etaR0 *= network0.decay

        if (epoch + 1) % network0.frequency == 0:
            etaW1 *= network0.decay
            etaB1 *= network0.decay
            etaP1 *= network0.decay
            etaR1 *= network0.decay

        for u, v in tqdm(trainingSet,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {network0.epochs:<2}',
                         colour='green',
                         leave=False,
                         total=network0.datasetCap):
            network0.zeros_grad()
            network0.zeros_io()

            network1.zeros_grad()
            network1.zeros_io()

            # Embedding Data
            embbed = network0.predict(u)
            p = network1.predict(embbed)

            loss = network1.g(p, v, network1.nOutputs)
            loss = np.sum(loss) / loss.shape[0]

            network1.update_params(v, etaW0, etaB0, etaP0, etaR0)

            if loss < previousLoss:
                previousLoss = loss
                network0.update_params(embbed + 0.01, etaW1, etaB1, etaP1, etaR1)
            else:
                network0.update_params(embbed * 1e+2, etaW1, etaB1, etaP1, etaR1)


def evaluate(network0, network1, evaluatingDataset):
    score = 0
    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False):
        embbed = network0.predict(u)
        p = network1.predict(embbed)
        print(p, v)
        p[p > 0.8] = 1.
        p[p <= 0.8] = 0.

        s0, engPredictInt = binaryToStr(engLang, p)
        s1, engTargetInt = binaryToStr(engLang, v)
        s2, inputInt = binaryToStr(engLang, u)

        if engPredictInt == engTargetInt:
            print(f'Input: {s2}, {inputInt} | Target: {s1}, {engTargetInt} '
                  f'| Predict: {s0}, {engPredictInt}')
            score += 1

    return score * 100 / len(evaluatingDataset)


def binaryToStr(lang, arr: np.ndarray):
    binaryToInt = 0b0

    for ele in ''.join(str(int(d)) for d in list(arr)):
        binaryToInt <<= 1
        binaryToInt += int(ele)

    try:
        s = lang.index2word[str(binaryToInt)]
    except KeyError:
        s = ''

    return s, binaryToInt


if __name__ == '__main__':
    maxLenFra = 0
    maxLenEng = 0
    dataset = []

    engLang = Lang()
    synonyms = np.load('./data/synonyms.npy')
    engLang.word2index = list(np.load('./data/engLangWord2Index.npy'))
    datasetCap = len(synonyms) // 256

    with open('./data/engLangIndex2Word.json', 'r') as f:
        engLang.index2word = json.load(f)

    maxLength = len(np.binary_repr(len(engLang.word2index)))
    for idx, (w0, w1) in tqdm(enumerate(synonyms),
                              desc='Vectorized Data',
                              colour='green',
                              total=datasetCap):
        if idx == datasetCap:
            break

        w0InInt = engLang.word2index.index(w0)
        w1InInt = engLang.word2index.index(w1)

        vectorizedInput = np.array(
            [int(b) for b in np.binary_repr(w0InInt, width=maxLength)]
        )

        vectorizedOutput = np.array(
            [int(b) for b in np.binary_repr(w1InInt, width=maxLength)]
        )
        dataset.append([vectorizedInput.flatten(), vectorizedOutput.flatten()])

    # Network's 0 Params
    nInputs0 = dataset[0][0].shape[0]
    nHiddens0 = 8
    nOutputs0 = 5

    maxInputPerNode0 = nInputs0 + nHiddens0
    maxOutputPerNode0 = nOutputs0 + nHiddens0

    # Network's 1 Params
    nInputs1 = 5
    nHiddens1 = 8
    nOutputs1 = dataset[0][1].shape[0]

    maxInputPerNode1 = nInputs1 + nHiddens1
    maxOutputPerNode1 = nOutputs1 + nHiddens1

    isTrain = True
    EPOCHS = 24
    record_no = 114
    nTest = 100

    trainingSet = dataset[:datasetCap]
    evaluatingSet = dataset[datasetCap:]

    network0 = Network(
        nInputs=nInputs0, nHiddens=nHiddens0, nOutputs=nOutputs0,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
        minRadius=500, maxRadius=800, minBias=0.5, maxBias=0.8,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, etaB=6e-2, decay=0.1,
        width=1000, height=1000, depth=1000,
        hiddenZoneOffset=200, outputZoneOffset=200,
        maxInputPerNode=maxInputPerNode0 // 1, minInputPerNode=maxInputPerNode0 // 2,
        maxOutputPerNode=maxOutputPerNode0 // 1, minOutputPerNode=maxOutputPerNode0 // 2,
        activationFunc='sigmoid', lossFunc='mse', datasetName='synonyms', train=isTrain,
        isSequential=False
    )

    network1 = Network(
        nInputs=nInputs1, nHiddens=nHiddens1, nOutputs=nOutputs1,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
        minRadius=500, maxRadius=800, minBias=0.5, maxBias=0.8,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, etaB=6e-2, decay=0.1,
        width=1000, height=1000, depth=1000,
        hiddenZoneOffset=200, outputZoneOffset=200,
        maxInputPerNode=maxInputPerNode1 // 1, minInputPerNode=maxInputPerNode1 // 2,
        maxOutputPerNode=maxOutputPerNode1 // 1, minOutputPerNode=maxOutputPerNode1 // 2,
        activationFunc='relu', lossFunc='ce', datasetName='synonyms', train=isTrain,
        isSequential=False
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        network0FolderPath = f'./records/{folderName}/network0'
        network1FolderPath = f'./records/{folderName}/network1'

        os.mkdir(simulationFolderPath)
        os.mkdir(network0FolderPath)
        os.mkdir(network1FolderPath)
        print(f'Using Folder\'s Named: {folderName}')

        startTime = time.perf_counter()
        train(network0, network1, trainingSet, network0FolderPath, network1FolderPath)
        accuracy = evaluate(network0, network1, trainingSet)
        print(f'Accuracy: {accuracy:.2f}%')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')

        network0.save_result(network0FolderPath)
        network0.save_config(trainingTime=totalTime,
                             accuracy=0.0,
                             simulationFolderPath=network0FolderPath)
        network0.plot()

        network1.save_result(network1FolderPath)
        network1.save_config(trainingTime=totalTime,
                             accuracy=accuracy,
                             simulationFolderPath=network1FolderPath)
        network1.plot()
