import json
import os
import random
import time
import numpy as np
import string

from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername
from preprocessingTranslationDataset import Lang


def train(network, trainingSet, simulationFolderPath, numberOfRehearsal):
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

        for u, v in tqdm(trainingSet,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {network.epochs:<2}',
                         colour='green',
                         leave=False,
                         total=len(trainingSet)):
            for rehearsalIdx in range(numberOfRehearsal):  # Rehearsal n times
                network.zeros_grad()
                network.predict(u)
                network.update_params(v, etaW, etaB, etaP, etaR)


def evaluate(network, evaluatingDataset, charInBinaryMaxLength, charsPerSentence, chars):
    score = 0
    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False,
                     total=len(evaluatingDataset)):
        p = network.predict(u).reshape(charsPerSentence,
                                       charInBinaryMaxLength)
        p[p > 0.5] = 1.
        p[p <= 0.5] = 0.

        s0 = binaryToStr(chars, p)
        s1 = binaryToStr(chars, v.reshape(charsPerSentence,
                                          charInBinaryMaxLength))
        s2 = binaryToStr(chars, u.reshape(charsPerSentence,
                                          charInBinaryMaxLength))

        # print(f'Input: {s2:<30}| Target: {s1:<30}| Predict: {s0:<30}')
        # print(f'Input: {s2:<{charsPerSentence}}| '
        #       f'Target: {s1:<{charsPerSentence}}| '
        #       f'Predict: {s0:<{charsPerSentence}}')

        if s0 == s1:
            print(f'Input: {s2:<{charsPerSentence}}| '
                  f'Target: {s1:<{charsPerSentence}}| '
                  f'Predict: {s0:<{charsPerSentence}}')
            score += 1

    return score


def binaryToStr(chars, arr: np.ndarray):
    s = ''

    for charIdx in range(arr.shape[0]):
        binaryToInt = 0b0

        for ele in ''.join(str(int(d)) for d in list(arr[charIdx])):
            binaryToInt <<= 1
            binaryToInt += int(ele)

        try:
            s += chars[int(binaryToInt)]
        except IndexError:
            s += ''

    return s


if __name__ == '__main__':
    maxLenFra = 0
    maxLenEng = 0

    dataset = []
    characters = [c for c in string.printable]
    characters.insert(0, '')
    charInBinaryMaxLength = len(np.binary_repr(len(characters)))
    uniqueData0 = []
    uniqueData1 = []

    engLang = Lang()
    synonyms = list(np.load('./data/synonyms.npy'))
    random.shuffle(synonyms)

    engLang.word2index = list(np.load('./data/engLangWord2Index.npy'))
    datasetCap = 30
    charsPerSentence = 40

    with open('./data/engLangIndex2Word.json', 'r') as f:
        engLang.index2word = json.load(f)

    maxLength = len(np.binary_repr(len(engLang.word2index)))

    for idx, (w0, w1) in tqdm(enumerate(synonyms),
                              desc='Vectorized Data',
                              colour='green',
                              total=datasetCap):
        if len(dataset) == datasetCap:
            break

        if len(list(w0)) <= charsPerSentence and len(list(w1)) <= charsPerSentence:
            vectorizedInput = np.zeros((charInBinaryMaxLength * charsPerSentence,))
            for i, c in enumerate(w0):
                vectorizedInput[charInBinaryMaxLength * i:
                                charInBinaryMaxLength * (i + 1)] \
                    = np.array([
                    int(b) for b in np.binary_repr(characters.index(c),
                                                   width=charInBinaryMaxLength)
                ])

            vectorizedOutput = np.zeros((charInBinaryMaxLength * charsPerSentence,))
            for i, c in enumerate(w1):
                vectorizedOutput[charInBinaryMaxLength * i:
                                 charInBinaryMaxLength * (i + 1)] \
                    = np.array([
                    int(b) for b in np.binary_repr(characters.index(c),
                                                   width=charInBinaryMaxLength)
                ])

            if w0 not in uniqueData0 and w1 not in uniqueData1:
                dataset.append([vectorizedInput, vectorizedOutput])
                uniqueData0.append(w0)
                uniqueData1.append(w1)

    nInputs = charInBinaryMaxLength * charsPerSentence
    nHiddens = 30
    nOutputs = charInBinaryMaxLength * charsPerSentence

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    EPOCHS = 60
    numberOfRehearsal = 200
    record_no = 30

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=10,
        minRadius=100, maxRadius=300, minBias=0.1, maxBias=0.8,
        etaW=1e-0, etaP=1e-0, etaR=1e-0, etaB=1e-0, decay=0.1,
        width=500, height=500, depth=500,
        hiddenZoneOffset=100, outputZoneOffset=100,
        maxInputPerNode=maxInputPerNode // 1, minInputPerNode=maxInputPerNode // 1,
        maxOutputPerNode=maxOutputPerNode // 1, minOutputPerNode=maxOutputPerNode // 1,
        activationFunc='sigmoid', lossFunc='mse', datasetName='synonyms', train=isTrain
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        os.mkdir(simulationFolderPath)
        print(f'Using Folder\'s Named: {folderName}')

        print(f'Number Of Words To Map: {len(dataset)}')
        startTime = time.perf_counter()
        train(network, dataset, simulationFolderPath, numberOfRehearsal)
        numberOfCorrectMapping = evaluate(network, dataset,
                                          charInBinaryMaxLength, charsPerSentence,
                                          characters)
        print(f'Number Of Correct Mapping: {numberOfCorrectMapping}')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')

        for u, v in dataset:
            network.zeros_io()
            network.predict(u)
            network.record_output_of_hidden_neurons(simulationFolderPath,
                                                    binaryToStr(characters,
                                                                v.reshape(charsPerSentence,
                                                                          charInBinaryMaxLength))[0])

        network.save_result(simulationFolderPath)
        network.save_config(trainingTime=totalTime,
                            accuracy=numberOfCorrectMapping,
                            simulationFolderPath=simulationFolderPath)
        network.plot()
