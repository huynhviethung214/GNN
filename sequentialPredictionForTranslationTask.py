import os
import time
import numpy as np

from tqdm import tqdm
from models.GNN import Network
from getSimulationFolderName import get_foldername
from processFraEng import prepareData


def train(network, trainingSet, simulationFolderPath):
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

        for u, v in tqdm(trainingSet,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {network.epochs:<2}',
                         colour='green',
                         leave=False,
                         total=network.datasetCap):
            network.zeros_grad()
            network.zeros_io()

            network.predict(u)
            network.update_params(v, etaW, etaP, etaR)


def evaluate(network, evaluatingDataset):
    score = 0

    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False):
        predict = network.predict(u)
        predict[predict > 0.9] = 1.
        predict[predict <= 0.9] = 0.

        if predict == v:
            score += 1

    return score * 100 / len(evaluatingDataset)


def binaryToStr(output_lang, arr: np.ndarray):
    s = ''
    ints = []

    for i in range(arr.shape[0]):
        binaryToInt = 0b0
        binaryToArrayOfInt = []

        for ele in arr[i]:
            binaryToInt <<= 1
            binaryToInt += int(ele)

        binaryToArrayOfInt.append(binaryToInt)

        if binaryToInt in list(output_lang.index2word.keys()):
            s += f'{output_lang.index2word[binaryToInt]} '
        else:
            s += ''

        ints.append(binaryToArrayOfInt)

    return s, ints


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    engWordInBinaryLength = len(np.binary_repr(max(list(output_lang.word2count.values()))))
    fraWordLengthInBinary = len(np.binary_repr(max(list(input_lang.word2count.values()))))

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
        vectorizedBinaryFra = np.zeros((maxLenFra, fraWordLengthInBinary))

        # print(dataset[i][0].shape[0])
        # dataset[i][0] = np.pad(dataset[i][0], (0, maxLenFra - dataset[i][0].shape[0]))
        # dataset[i][1] = np.pad(dataset[i][1], (0, maxLenEng - dataset[i][1].shape[0]))

        for j, ele in enumerate(dataset[i][0]):
            vectorizedBinaryFra[j] = np.array(
                [int(b) for b in np.binary_repr(ele, width=fraWordLengthInBinary)]
            )

        for j, ele in enumerate(dataset[i][1]):
            vectorizedBinaryEng[j] = np.array(
                [int(b) for b in np.binary_repr(ele, width=engWordInBinaryLength)]
            )

        # vectorizedBinaryFra[0] = np.zeros((1, fraWordLengthInBinary))
        # vectorizedBinaryFra[0, 0] = 1.
        # vectorizedBinaryFra[-1, -1] = 1.
        #
        # vectorizedBinaryEng[0] = np.zeros((1, engWordInBinaryLength))
        # vectorizedBinaryEng[0, 0] = 1.
        # vectorizedBinaryEng[-1, -1] = 1.

        dataset[i][0] = vectorizedBinaryFra.flatten()
        dataset[i][1] = vectorizedBinaryEng.flatten()

    nInputs = dataset[0][0].shape[0]
    nHiddens = 100
    nOutputs = dataset[0][1].shape[0]
    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    EPOCHS = 3
    datasetCap = len(dataset)
    record_no = 93
    nTest = 100

    trainingSet = dataset[:datasetCap]
    evaluatingSet = dataset[datasetCap:]

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=1,
        minRadius=400, maxRadius=800,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=0.1,
        width=5000, height=5000, depth=5000, bias=-0.9,
        hiddenZoneOffset=1000, outputZoneOffset=1000,
        maxInputPerNode=maxInputPerNode // 6, minInputPerNode=maxInputPerNode // 8,
        maxOutputPerNode=maxOutputPerNode // 6, minOutputPerNode=maxOutputPerNode // 8,
        activationFunc='sigmoid', datasetName='fra-eng', train=isTrain
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        os.mkdir(simulationFolderPath)
        print(f'Using Folder\'s Named: {folderName}')

        startTime = time.perf_counter()

        network.getActivationFunction()
        train(network, trainingSet, simulationFolderPath)

        accuracy = evaluate(network,
                            trainingSet)
        print(f'Accuracy: {accuracy:.2f}%')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')

        network.save_result(simulationFolderPath)
        network.save_config(trainingTime=totalTime,
                            accuracy=accuracy,
                            simulationFolderPath=simulationFolderPath)
        network.plot()
    else:
        network.load_simulation(f'records/3d_nodes_simulation_{record_no}')
        fraWords = trainingSet[5][0]
        engWords = trainingSet[5][1].reshape((maxLenEng, engWordInBinaryLength))

        p = network.predict(fraWords).reshape((maxLenEng, engWordInBinaryLength))
        p[p > 0.9] = 1.
        p[p <= 0.9] = 0.

        engPredict, _ = binaryToStr(p, output_lang)

        print(binaryToStr(engWords, output_lang)[0])
        print(engPredict)
