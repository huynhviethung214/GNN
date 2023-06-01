import pickle
import re
import json

import numpy as np
import unicodedata
from tqdm import tqdm


class Lang:
    def __init__(self):
        self.word2index = []
        self.index2word = {}


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def fraToEngPreprocessing():
    with open('./data/fra.txt', 'r') as f:
        engLang = Lang()
        fraLang = Lang()

        fraToEng = []

        for line in f.readlines():
            line = line.replace('CC-BY 2 0 (France) Attribution: tatoeba', '')
            sentences = line.split('\t')

            engSentence = sentences[0].lower()
            engSentence = unicodeToAscii(engSentence)

            fraSentence = sentences[1].lower()
            fraSentence = unicodeToAscii(fraSentence)

            engSentence = re.sub(r"([.!?])", r" \1", engSentence)
            engSentence = re.sub(r"[^a-zA-Z.!?]+", r" ", engSentence)

            fraSentence = re.sub(r"([.!?])", r" \1", fraSentence)
            fraSentence = re.sub(r"[^a-zA-Z.!?]+", r" ", fraSentence)

            for engWord in engSentence.split(' '):
                if engWord not in engLang.word2index:
                    engLang.word2index.append(engWord)
                    engLang.index2word.update({engWord: engLang.word2index.index(engWord)})

            for fraWord in fraSentence.split(' '):
                if fraWord not in fraLang.word2index:
                    fraLang.word2index.append(fraWord)
                    fraLang.index2word.update({fraWord: fraLang.word2index.index(fraWord)})

            # engSentences.append(engSentence)
            # fraSentences.append(fraSentence)

            fraToEng.append([fraSentence, engSentence])

        # print(len(engSentences))
        # print(len(fraSentences))

    return engLang, fraLang, fraToEng


def synonymsPreprocessing():
    processedData = []
    engLang = Lang()

    with open('./data/synonyms.json', 'r') as f:
        data = json.load(f)

        for key, item in tqdm(data.items(),
                               total=len(data.keys())):
            k = key.split(':')[0]
            values = item.split(';')

            if k not in engLang.word2index:
                engLang.word2index.append(k)

            for value in values:
                if '|' in value:
                    value = value.split('|')[0]

                if value not in engLang.word2index:
                    engLang.word2index.append(value)

                processedData.append([k, value])

    engLang.word2index = list(set(engLang.word2index))
    engLang.index2word.update({i: word for i, word in enumerate(engLang.word2index)})

    return processedData, engLang


if __name__ == '__main__':
    synonyms, engLang = synonymsPreprocessing()

    np.save('./data/synonyms.npy', synonyms)
    np.save('./data/engLangWord2Index.npy', engLang.word2index)

    with open('./data/engLangIndex2Word.json', 'w') as f:
        json.dump(engLang.index2word, f)
