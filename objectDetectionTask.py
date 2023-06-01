import time
import cv2
import numpy as np
import json as js

from tqdm import tqdm
from models.GNN import Network


# TODO: NEED A BIGGER RAM STICK (256GB DUAL-CHANNELS)
if __name__ == '__main__':
    nInputs = 400 * 400
    nHiddens = 500  # NOQA
    nOutputs = 4
    N = nInputs + nHiddens + nOutputs

    annotationsFP = './data/number_ops/train/_annotations.coco.json'

    maxInputPerNode = nInputs + nHiddens
    maxOutputPerNode = nOutputs + nHiddens

    train = True
    EPOCHS = 24
    datasetCap = 3000
    ratio = 1 / 2
    record_no = 69
    nTest = 100

    classes = np.zeros((nOutputs,), dtype=np.int64)
    evaluatingDataset = []
    trainingDataset = []
    validationSet = []

    if train:
        with open(annotationsFP, 'r') as f:
            details = js.load(f)
            numberOfSamples = len(details['annotations'])

            for i, detail in tqdm(enumerate(details['annotations']),
                                  desc='Preprocessing Training & Evaluating Data',
                                  colour='green',
                                  total=len(details['annotations'])):
                imageID = detail['id']
                label = detail['category_id']

                bbox = detail['bbox']

                try:
                    maxbboxX = max(bbox[0], bbox[2])
                    minbboxX = min(bbox[0], bbox[2])

                    bbox[0] = (bbox[0] - minbboxX) / (maxbboxX - minbboxX)
                    bbox[2] = (bbox[2] - minbboxX) / (maxbboxX - minbboxX)

                    maxbboxY = max(bbox[1], bbox[3])
                    minbboxY = min(bbox[1], bbox[3])

                    bbox[1] = (bbox[1] - minbboxY) / (maxbboxY - minbboxY)
                    bbox[3] = (bbox[3] - minbboxY) / (maxbboxY - minbboxY)
                except ZeroDivisionError:
                    continue

                bbox = np.array(bbox, dtype=np.float64)

                imageName = details['images'][imageID]['file_name']
                imagePath = f'./data/number_ops/train/{imageName}'

                u = cv2.imread(imagePath)
                u = cv2.cvtColor(u, cv2.COLOR_RGB2GRAY)

                u = np.array(u) / np.max(u)
                u = u.flatten()

                if i < datasetCap:
                    trainingDataset.append([u, bbox])
                else:
                    evaluatingDataset.append([u, bbox])

    network = Network(
        nInputs=nInputs, nHiddens=nHiddens, nOutputs=nOutputs,
        trainingDataset=trainingDataset, evaluatingDataset=evaluatingDataset,
        epochs=EPOCHS, datasetCap=datasetCap, frequency=4,
        minRadius=500, maxRadius=800, save=True,
        etaW=6e-2, etaP=6e-2, etaR=6e-2, decay=0.1,
        width=8000, height=8000, depth=8000, bias=-0.5,
        hiddenZoneOffset=3000, outputZoneOffset=600,
        maxInputPerNode=maxInputPerNode, minInputPerNode=maxInputPerNode // 2,
        maxOutputPerNode=maxOutputPerNode, minOutputPerNode=maxOutputPerNode // 2,
        activationFunc='sigmoid', datasetName='roboflow', train=train
    )

    if train:
        startTime = time.perf_counter()
        network.train()
        confusionMatrix = network.evaluate()
        sumOfDiag = np.sum(np.diag(confusionMatrix))
        accuracy = sumOfDiag * 100. / len(evaluatingDataset)
        print(f'Accuracy: {accuracy}%')

        totalTime = time.perf_counter() - startTime
        print(f'Total Established Time: {totalTime} (sec)')
        print(f'Precision:\n{network.getPrecisionForEachClass()}')

        network.save_result()
        network.save_config(trainingTime=totalTime,
                            accuracy=accuracy)
        network.plot()
    else:
        network.load_simulation(f'records/3d_nodes_simulation_{record_no}')