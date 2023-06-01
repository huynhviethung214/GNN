import json
import os
import time
import numpy as np

from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername
from preprocessingTranslationDataset import Lang


def train(network, trainingSet, simulationFolderPath):
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
            for rehearsalIdx in range(100):  # Rehearsal 100 times
                network.zeros_grad()
                network.zeros_io()

                network.predict(u)
                network.update_params(v, etaW, etaB, etaP, etaR)
                # network.W = fixSynapses(network.W, network.P, network.R,
                #                         network.nodeIdc, network.nInputs, network.nOutputs)


def evaluate(network, evaluatingDataset):
    score = 0
    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False,
                     total=len(evaluatingDataset)):
        # v = sine(v)
        # u = sine(u)

        u = u / 0.8
        p = network.predict(u)
        # p = sine(p)
        p[p > 0.5] = 1.
        p[p <= 0.5] = 0.
        # print(p, v)

        s0, engPredictInt = binaryToStr(engLang, p)
        s1, engTargetInt = binaryToStr(engLang, v)
        s2, inputInt = binaryToStr(engLang, u)

        # print(f'Input: {s2}, {inputInt} | Target: {s1}, {engTargetInt} '
        #       f'| Predict: {s0}, {engPredictInt}')

        if engPredictInt == engTargetInt:
            print(f'Input: {s2:<30}, {inputInt:<10} | Target: {s1:<30}, {engTargetInt:<10} '
                  f'| Predict: {s0:<30}, {engPredictInt:<10}')
            score += 1

    return score


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
    uniqueData0 = []
    uniqueData1 = []

    engLang = Lang()
    synonyms = list(np.load('./data/synonyms.npy'))
    # random.shuffle(synonyms)

    engLang.word2index = list(np.load('./data/engLangWord2Index.npy'))
    datasetCap = 40

    with open('./data/engLangIndex2Word.json', 'r') as f:
        engLang.index2word = json.load(f)

    # print(engLang.index2word)
    maxLength = len(np.binary_repr(len(engLang.word2index)))

    for idx, (w0, w1) in tqdm(enumerate(synonyms),
                              desc='Vectorized Data',
                              colour='green',
                              total=datasetCap):
        if len(dataset) == datasetCap:
            break

        w0InInt = engLang.word2index.index(w0)
        w1InInt = engLang.word2index.index(w1)

        vectorizedInput = np.array(
            [int(b) for b in np.binary_repr(w0InInt, width=maxLength)]
        ).flatten() * 0.8

        vectorizedOutput = np.array(
            [int(b) for b in np.binary_repr(w1InInt, width=maxLength)]
        ).flatten()

        # vectorizedInput = np.zeros((maxLength, maxLength))
        # binaryOfW0 = [int(b) for b in np.binary_repr(w0InInt, width=maxLength)]
        # for i, bit in enumerate(binaryOfW0):
        #     vectorizedInput[i, i] = bit
        # vectorizedInput = vectorizedInput.flatten()
        #
        # vectorizedOutput = np.zeros((maxLength, maxLength))
        # binaryOfW1 = [int(b) for b in np.binary_repr(w1InInt, width=maxLength)]
        # for i, bit in enumerate(binaryOfW1):
        #     vectorizedOutput[i, i] = bit
        # vectorizedOutput = vectorizedOutput.flatten()

        # dataset.append([vectorizedInput, vectorizedOutput])
        if w0 not in uniqueData0 and w1 not in uniqueData1:
            # dataset.append([np.array(w0InInt / maxLength).reshape(1,), vectorizedOutput])
            dataset.append([vectorizedInput, vectorizedOutput])
            uniqueData0.append(w0)
            uniqueData1.append(w1)

    nInputs = maxLength
    nHiddens = 8
    nOutputs = maxLength

    # maxInputPerNode = nInputs + nHiddens
    # maxOutputPerNode = nOutputs + nHiddens

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    EPOCHS = 800
    record_no = 114
    nTest = 100

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=EPOCHS,
        minRadius=400, maxRadius=800, minBias=0.1, maxBias=0.8,
        etaW=1e-0, etaP=1e-0, etaR=1e-0, etaB=1e-0, decay=0.1,
        width=500, height=500, depth=500,
        hiddenZoneOffset=100, outputZoneOffset=100,
        maxInputPerNode=maxInputPerNode // 1, minInputPerNode=maxInputPerNode // 1,
        maxOutputPerNode=maxOutputPerNode // 1, minOutputPerNode=maxOutputPerNode // 1,
        activationFunc='sigmoid', lossFunc='mse', datasetName='synonyms', train=isTrain,
        isSequential=False
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        os.mkdir(simulationFolderPath)
        print(f'Using Folder\'s Named: {folderName}')

        print(f'Number Of Words To Memorize: {len(dataset)}')
        startTime = time.perf_counter()
        train(network, dataset, simulationFolderPath)
        numberOfCorrectMapping = evaluate(network, dataset)
        print(f'Number Of Correct Mapping: {numberOfCorrectMapping}')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')

        for u, v in dataset:
            # v = np.sum(v.reshape((maxLength, maxLength)), axis=0)

            network.zeros_io()
            network.predict(u)
            network.record_output_of_hidden_neurons(simulationFolderPath,
                                                    binaryToStr(engLang, v)[0])

        network.save_result(simulationFolderPath)
        network.save_config(trainingTime=totalTime,
                            accuracy=numberOfCorrectMapping,
                            simulationFolderPath=simulationFolderPath)
        network.plot()
    else:
        network.load_simulation(f'records/3d_nodes_simulation_{record_no}')

        for i in tqdm(range(len(dataset)),
                      desc='Testing',
                      colour='green',
                      total=len(dataset)):
            v = dataset[i][1]
            # v[v >= BIAS] = 1.
            # v[v < BIAS] = 0.

            p = network.predict(dataset[i][0])

            engPredict, intPredict = binaryToStr(engLang, p)
            engTarget, intTarget = binaryToStr(engLang, v)

            print(f'Target: {engTarget:<60} | Predict: {engPredict:<60}')
