import torchaudio

if __name__ == '__main__':
    trainingLibriSpeech = torchaudio.datasets.LIBRISPEECH('./data',
                                                          url='train-clean-100',
                                                          download=False)
    evaluatingLibriSpeech = torchaudio.datasets.LIBRISPEECH('./data',
                                                            url='test-clean',
                                                            download=False)

    words = []
    datasetCap = len(trainingLibriSpeech) // 3

    train = True
    EPOCHS = 32
    record_no = 75
    nTest = 100

    evaluatingDataset = []
    trainingDataset = []
    validationSet = []

    maxLenX = 0
    maxLenY = 0

    # for i, data in tqdm(enumerate(trainingLibriSpeech),
    #                     desc='Zippy Zip Zip',
    #                     total=len(trainingLibriSpeech)):
        # x = librosa.feature.mfcc(y=data[0].numpy(), sr=data[1]).flatten()
        # y = data[2]
        # vectorizedY = []

        # for word in y:
        #     if word not in words:
        #         words.append(word)
        #     vectorizedY.append(words.index(word))
        #
        # if i < datasetCap:
        #     trainingDataset.append([x, np.array(vectorizedY, dtype=np.float64)])
        # else:
        #     evaluatingDataset.append([x, np.array(vectorizedY, dtype=np.float64)])
        #
        # if len(vectorizedY) > maxLenY:
        #     maxLenY = len(vectorizedY)
        #
        # if x.shape[-1] > maxLenX:
        #     maxLenX = x.shape[0]

    # for i in tqdm(range(len(trainingDataset))):
    #     trainingDataset[i][0] = np.pad(trainingDataset[i][0],
    #                                    (0, maxLenX - len(trainingDataset[i][0])),
    #                                    constant_values=0)
    #
    #     trainingDataset[i][1] = np.pad(trainingDataset[i][1],
    #                                    (0, maxLenY - len(trainingDataset[i][1])),
    #                                    constant_values=0)
    #
    # for i in tqdm(range(len(evaluatingDataset))):
    #     evaluatingDataset[i][0] = np.pad(evaluatingDataset[i][0],
    #                                      (0, maxLenX - len(evaluatingDataset[i][0])),
    #                                      constant_values=0)
    #
    #     evaluatingDataset[i][1] = np.pad(evaluatingDataset[i][1],
    #                                      (0, maxLenY - len(evaluatingDataset[i][1])),
    #                                      constant_values=0)

    # print(len(trainingDataset))
    # print(len(evaluatingDataset))

    # np.save('./trainingLibriSpeech.npy', trainingDataset)
    # np.save('./evaluatingLibriSpeech.npy', evaluatingDataset)

    # trainingDataset = list(np.load('./trainingLibriSpeech.npy', allow_pickle=True))
    # evaluatingDataset = list(np.load('./evaluatingLibriSpeech.npy', allow_pickle=True))
    #
    # nInputs = trainingDataset[0][0].shape[0]
    # nHiddens = 100  # NOQA
    # nOutputs = trainingDataset[0][1].shape[0]
    #
    # maxInputPerNode = nInputs + nHiddens
    # maxOutputPerNode = nOutputs + nHiddens
    #
    # network = Network(
    #     nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
    #     trainingDataset=trainingDataset, evaluatingDataset=evaluatingDataset,
    #     epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
    #     minRadius=100, maxRadius=300, save=True,
    #     etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=0.1,
    #     width=5000, height=5000, depth=5000, bias=-0.5,
    #     hiddenZoneOffset=1200, outputZoneOffset=300,
    #     maxInputPerNode=maxInputPerNode, minInputPerNode=maxInputPerNode // 2,
    #     maxOutputPerNode=maxOutputPerNode, minOutputPerNode=maxOutputPerNode // 2,
    #     activationFunc='tanh', datasetName='libriSpeech', train=train
    # )
    #
    # if train:
    #     startTime = time.perf_counter()
    #     network.train()
    #     accuracy = network.evaluateForSequence()
    #     print(f'Accuracy: {accuracy}')
    #
    #     totalTime = time.perf_counter() - startTime
    #     print(f'Total Established Time: {totalTime} (sec)')
    #
    #     network.save_result()
    #     network.save_config(totalTime, accuracy=accuracy)
    #     network.plot()
    # else:
    #     network.load_simulation(f'records/3d_nodes_simulation_{record_no}')
    #
    #     for wordIdx in evaluatingDataset[0][1]:
    #         print(words[wordIdx], end=' ')
    #
    #     print('\n')
    #
    #     for output in network.predictForClassification(evaluatingDataset[0][0]):
    #         print(words[int(output)], end=' ')
