import os
import numpy as np

from librosa import feature, load
from tqdm import tqdm


def get_audio_features(fp):
    y, sr = load(fp)
    features = np.mean(feature.mfcc(y=y, sr=sr, n_mfcc=128).T, axis=0)

    return features


if __name__ == '__main__':
    isTrainingType = 1
    datasetFP = './data/libriSpeech/test/test-clean'
    # words = ['']
    words = list(np.load(f'./data/PLS/train/vectorized_words.npy'))
    labelsVector = []
    X = []

    for readerID in tqdm(os.listdir(datasetFP)):
        for chapterID in os.listdir(f'{datasetFP}/{readerID}'):
            translationFileName = f'{readerID}-{chapterID}.trans.txt'
            with open(f'{datasetFP}/{readerID}/{chapterID}/{translationFileName}', 'r') as f:
                paragraphs = f.readlines()

                for paragraph in paragraphs:
                    if '\n' in paragraph:
                        paragraph = paragraph.replace('\n', '')

                    paragraph = paragraph.split(' ')[1:]

                    labelVector = []
                    for word in paragraph:
                        # if word not in words:
                        #     words.append(word)
                        labelVector.append(words.index(word))
                    labelsVector.append(np.array(labelVector, dtype=np.int64))

    for readerID in tqdm(os.listdir(datasetFP)):
        for chapterID in os.listdir(f'{datasetFP}/{readerID}'):
            translationFileName = f'{readerID}-{chapterID}.trans.txt'

            for audioFileName in os.listdir(f'{datasetFP}/{readerID}/{chapterID}'):
                if audioFileName != translationFileName:
                    features = get_audio_features(f'{datasetFP}/{readerID}/{chapterID}/{audioFileName}')
                    X.append(features)

    maxLen = max([s.shape[0] for s in labelsVector])
    for i in range(len(labelsVector)):
        labelsVector[i] = np.pad(labelsVector[i],
                                 (0, maxLen - labelsVector[i].shape[0]),
                                 constant_values=0)

    X = np.array(X, dtype=np.float64)
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (maxX - minX)

    labelsVector = np.array(labelsVector, dtype=np.int64)

    np.save('evaluatingX.npy', X)
    np.save('evaluatingY.npy', labelsVector)
    # np.save('vectorized_words.npy', words)
