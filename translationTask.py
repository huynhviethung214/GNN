import os
import time
from io import open

import numpy as np
import unicodedata
import re

from tqdm import tqdm

from models.GNN import Network
from getSimulationFolderName import get_foldername


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: ' '}
        self.n_words = 1  # Count SOS and EOS

    def addSentence(self, sentence):
        if 'f .b .i' in sentence:
            sentence = sentence.replace('f .b .i', 'fbi')

        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    # s = s.replace("i m ", "i am ")
    # s = s.replace("he s ", "he is ")
    # s = s.replace("she s ", "she is ")
    # s = s.replace("you re ", "you are ")
    # s = s.replace("we re ", "we are ")
    # s = s.replace("they re", "they are")

    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/fra_eng/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    for i in range(len(pairs)):
        for j in range(len(pairs[i])):
            if 'cc by' in pairs[i][j]:
                pairs[i].remove(pairs[i][j])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 20

eng_prefixes = (
    "i am ",
    "he is",
    "she is",
    "you are",
    "we are",
    "they are",
)

# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s ",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def train(network, trainingSet, simulationFolderPath):
    network.getActivationFunction()
    time.sleep(1)

    for epoch in range(network.epochs):
        etaW = network.etaW
        etaP = network.etaP
        etaR = network.etaR

        network.save_weight_image_per_epoch(epoch, simulationFolderPath)

        if (epoch + 1) % network.frequency == 0:
            etaW *= 1 / (network.decay * (epoch + 1))
            etaP *= 1 / (network.decay * (epoch + 1))
            etaR *= 1 / (network.decay * (epoch + 1))

        for u, v in tqdm(trainingSet,
                         desc=f'Training | Epoch: {epoch + 1:<2} / {network.epochs:<2}',
                         colour='green',
                         leave=False,
                         total=network.datasetCap):
            network.zeros_grad()
            network.zeros_io()

            network.predict(u)
            network.update_params(v, etaW, etaP, etaR)
            # network.W = fixSynapses(network.W, network.P, network.minRadius,
            #                         network.maxRadius, network.nodeIdc)


def getAccuracy(u, v):
    score = 0
    c = 0

    for i in range(len(u)):
        if v[i] != [0]:
            if u[i] == v[i]:
                score += 1
            c += 1

    return score * 100 / c


def evaluate(network, evaluatingDataset, engWordInBinaryLength, maxLenEng):
    score = 0
    for u, v in tqdm(evaluatingDataset,
                     desc=f'Evaluating: ',
                     colour='green',
                     leave=False):
        predict = network.predict(u).reshape((maxLenEng,
                                              engWordInBinaryLength))
        predict[predict > 0.5] = 1.
        predict[predict <= 0.5] = 0.

        predict = binaryToStr(predict)[1]
        target = binaryToStr(v.reshape((maxLenEng, engWordInBinaryLength)))[1]

        score += getAccuracy(predict, target)

    return score / len(evaluatingDataset)


def binaryToStr(arr: np.ndarray):
    s = ''
    binaries = []

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

        binaries.append(binaryToArrayOfInt)

    return s, binaries


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

        dataset[i][0] = np.pad(dataset[i][0], (0, maxLenFra - dataset[i][0].shape[0]))
        dataset[i][1] = np.pad(dataset[i][1], (0, maxLenEng - dataset[i][1].shape[0]))

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

    nInputs = dataset[0][0].shape[0]
    nHiddens = 40
    nOutputs = dataset[0][1].shape[0]
    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    isTrain = True
    EPOCHS = 12
    datasetCap = len(dataset) // 8
    record_no = 97
    nTest = 100

    trainingSet = dataset[:datasetCap]
    evaluatingSet = dataset[datasetCap:]

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
        minRadius=1000, maxRadius=3000,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=3,
        width=10000, height=10000, depth=10000, bias=-0.5,
        hiddenZoneOffset=2500, outputZoneOffset=2200,
        maxInputPerNode=maxInputPerNode // 1, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode // 1, minOutputPerNode=maxOutputPerNode // 2,
        activationFunc='sigmoid', datasetName='fra-eng', train=isTrain,
        isSequential=False
    )

    if isTrain:
        folderName = get_foldername()
        simulationFolderPath = f'./records/{folderName}'
        os.mkdir(simulationFolderPath)
        print(f'Using Folder\'s Named: {folderName}')

        startTime = time.perf_counter()
        train(network, trainingSet, simulationFolderPath)
        accuracy = evaluate(network,
                            trainingSet,
                            engWordInBinaryLength,
                            maxLenEng)
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

        for i in range(400):
            v = trainingSet[i][1].reshape((maxLenEng, engWordInBinaryLength))
            p = network.predict(trainingSet[i][0]).reshape((maxLenEng, engWordInBinaryLength))

            p[p > 0.5] = 1.
            p[p <= 0.5] = 0.

            engPredict, _ = binaryToStr(p)
            engTarget, _ = binaryToStr(v)

            print(f'Target: {engTarget:<60} | Predict: {engPredict:<60}')
