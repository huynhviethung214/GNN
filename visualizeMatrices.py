import matplotlib.pyplot as plt
import numpy as np
import json as js
import torchvision

from modules import forward, sigmoid


def mag(v0, v1):
    return np.sqrt((v0[0] - v1[0]) ** 2
                   + (v0[1] - v1[1]) ** 2
                   + (v0[2] - v1[2]) ** 2)


no = 37
with open(f'./records/3d_nodes_simulation_{no}/configs.json') as f:
    configs = js.load(f)

nInputs = configs['nInputs']
nHiddens = configs['nHiddens']
nOutputs = configs['nOutputs']
N = nInputs + nHiddens + nOutputs

W = np.load(f'./records/3d_nodes_simulation_{no}/w.npy')

# plt.imshow(W[0: nInputs, nInputs: nInputs + nHiddens], cmap='gray')
# plt.show()

# for i in range(W.shape[1]):
#     if len(np.where(W[0: nInputs, i])[0]) > 0:
#         plt.imshow(W[0: nInputs, i].reshape(28, 28), cmap='gray')
#         plt.show()

B = np.load(f'./records/3d_nodes_simulation_{no}/b.npy')
R = np.load(f'./records/3d_nodes_simulation_{no}/r.npy')
P = np.load(f'./records/3d_nodes_simulation_{no}/p.npy')


u, _ = torchvision.datasets.MNIST('./data', train=True, download=True)[1]
u = np.array(u)
u = u / 255.
u = u.flatten()

# plt.imshow(u, cmap='gray')
# plt.show()

inputIdc = np.array([i for i in range(nInputs)], dtype=np.int64).reshape((nInputs,))
hiddenIdc = np.array([i + nInputs for i in range(nHiddens)], dtype=np.int64).reshape((nHiddens,))
outputIdc = np.array([i + nInputs + nHiddens for i in range(nOutputs)], dtype=np.int64).reshape((nOutputs,))

I = np.zeros((N,), dtype=np.float64)
I[inputIdc] = u[inputIdc]
O = np.zeros((N,), dtype=np.float64)
I, O = forward(W, I, O, P, R, B, inputIdc, hiddenIdc, outputIdc, u, sigmoid)

# plt.imshow(O[nInputs: nInputs + nHiddens].reshape(14, 10), cmap='gray')
# plt.show()


# TODO: CALCULATE MAGNITUDE OF CONNECTIONS
magnitudes = np.zeros(W.shape, dtype=np.float64)
for nodeIdx in range(W.shape[0]):
    toIdc = np.where(W[nodeIdx, :] != 0)[0]

    for toIdx in toIdc:
        vectorMagnitude = mag(P[toIdx], P[nodeIdx])

        signalPreservation = R[nodeIdx] / vectorMagnitude
        # magnitudes[nodeIdx, toIdx] = signalPreservation

        if 0.01 <= signalPreservation <= 1.0:
            magnitudes[nodeIdx, toIdx] = signalPreservation * 100.
        else:
            magnitudes[nodeIdx, toIdx] = 0.

# Normalize W
# maxW = np.max(W)
# minW = np.min(W)
# W = (W - minW) / (maxW - minW)

# Normalize R
# maxR = np.max(R)
# minR = np.min(R)
# R = (R - minR) / (maxR - minR)

# Normalize Magnitudes
# maxMagnitudes = np.max(magnitudes)
# minMagnitudes = np.min(magnitudes)
# magnitudes = (magnitudes - minMagnitudes) / (maxMagnitudes - minMagnitudes)
# d = np.diag(magnitudes)

magnitudes[np.where(magnitudes == np.nan)] = 0.
# magnitudes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))
# magnitudes[magnitudes == 0.] = 1

# W[W == -0.] = 0
# W = np.abs(W)
# W[W != 0] += 200
# maxW = np.max(W)
# minW = np.min(W)
# W = (W - minW) / (maxW - minW)

# TODO: NEW CODE
# C = np.asarray(W).copy()
# C[C != 0] = 1.

# ROverMag = ((R / magnitudes) * W)
# # ROverMag = np.abs(ROverMag)
# maxROverMag = np.max(ROverMag)
# minROverMag = np.min(ROverMag)
# ROverMag = (ROverMag - minROverMag) * 100 / (maxROverMag - minROverMag)

# radiusOfNeurons = np.ones(C.shape) * R

# maxRadiusOfNeurons = np.max(radiusOfNeurons)
# minRadiusOfNeurons = np.min(radiusOfNeurons)
# radiusOfNeurons = (radiusOfNeurons - minRadiusOfNeurons) * 100 \
#                   / (maxRadiusOfNeurons - minRadiusOfNeurons)

# r = (W.T * (R / magnitudes)).T
# r = np.abs(r)
# rMax = np.max(r)
# rMin = np.min(r)
# print(rMax, rMin)
# r = (r - rMin) / (rMax - rMin)

# mask = np.asarray(r).copy()
# mask[mask != 0] = True
# mask[mask == 0] = False
# newArrayWithMask = ma.array(r, mask=mask)

# r[r >= (np.max(r) - np.min(r)) / 2] += 200
# magnitudes[np.where((np.min(R) <= magnitudes) <= np.max(R))] *= 1.

fig = plt.figure()
plt.xticks([], [])
plt.yticks([], [])

# plt.ylabel('Hidden Neurons and Output Neurons')
# plt.xlabel('Input Neurons')

plt.ylabel('Hidden Neurons')
plt.xlabel('Hidden Neurons')

# plt.xlabel(f'Node (i + {nInputs})-th')
# plt.ylabel(f'Node (j + {nInputs})-th')
# plt.xlabel(f'Neuron j-th')
# plt.ylabel(f'Node i-th')
# plt.title('Radius of Neurons')
# plt.imshow(radiusOfNeurons, cmap='hot')
plt.title('Percentage of Signal Preservation (%)')
# plt.title('Magnitude of Connections')
# W = np.abs(W)

# magnitudes = magnitudes[0: nInputs, nInputs:].T
# magnitudes = magnitudes[:, 390: 410].T
plt.imshow(magnitudes[nInputs: nInputs + nHiddens, nInputs:], cmap='gray')
# plt.imshow(magnitudes[nInputs: nInputs + nHiddens, nInputs: nInputs + nHiddens], cmap='gray')
# plt.imshow(ROverMag[0: nInputs, nInputs: nInputs + nHiddens].T, cmap='hot')
# plt.imshow(ROverMag[nInputs: nInputs + nHiddens, nInputs: nInputs + nHiddens], cmap='hot')
# plt.imshow(r[0: nInputs, nInputs: nInputs + nHiddens])
# plt.colorbar()
plt.show()
