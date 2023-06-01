import numpy as np
import torchvision.datasets
import matplotlib.pyplot as plt


# Folder's name format: {bestAccuracy:.2f}-{N}-{inputPerNode}-{outputPerNode}-{cap}-{strEta}-{epochs}-{bias}-{decay}-{frequency}
from tqdm import tqdm

PATH = '62.46-800-2-18-8000-7.504464823200006e.08-3-1-0.04218000000000001-2'
_, N, inputPerNode, outputPerNode, _, _, _, bias, _, _ = PATH.split('-')

N = int(N)
inputPerNode = int(inputPerNode)
outputPerNode = int(outputPerNode)
bias = int(bias)

nOutputs = 10
nInputs = 28 * 28


if __name__ == '__main__':
    inputIdc = [i for i in range(nInputs)]
    outputIdc = [N - i for i in range(1, nOutputs + 1)]
    hiddenIdc = []

    copiedInputIdc = inputIdc.copy()
    copiedInputIdc.extend(outputIdc)

    for i in range(N):
        if i not in copiedInputIdc:
            hiddenIdc.append(i)

    print(f'InputIdc  -> Start Idx: {min(inputIdc)} , End Idx: {max(inputIdc)}')
    print(f'HiddenIdc -> Start Idx: {min(hiddenIdc)}, End Idx: {max(hiddenIdc)}')
    print(f'OutputIdc -> Start Idx: {min(outputIdc)}, End Idx: {max(outputIdc)}')

    C = np.load(f'{PATH}/connections.npy')
    cFilter = np.zeros((len(inputIdc),))

    for inputIdx in inputIdc:
        connectedIdc = np.where(C[inputIdx] == 1)[0]

        if len(connectedIdc) > 0:
            cFilter[inputIdx] = 1

    cFilter = cFilter.reshape((28, 28))

    mnistDataset = torchvision.datasets.MNIST('./data', train=False)
    stacked = np.array(mnistDataset[0][0]) * cFilter

    for i, (u, _) in enumerate(tqdm(mnistDataset, desc='Preprocessing: ', colour='green')):
        if i > 0:
            stacked = np.hstack((stacked, np.array(u) * cFilter))

        if (i + 1) % 20 == 0:
            break

    plt.imshow(stacked, cmap='gray')
    plt.show()

