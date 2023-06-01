# from time import sleep
# from tqdm import tqdm, trange
# from concurrent.futures import ThreadPoolExecutor
#
# L = list(range(9))
#
# def progresser(n):
#     interval = 0.001 / (n + 2)
#     total = 5000
#     text = "#{}, est. {:<04.2}s".format(n, interval * total)
#     for _ in trange(total, desc=text):
#         sleep(interval)
#     if n == 6:
#         tqdm.write("n == 6 completed.")
#         tqdm.write("`tqdm.write()` is thread-safe in py3!")
#
# if __name__ == '__main__':
#     with ThreadPoolExecutor() as p:
#         p.map(progresser, L)

# import networkx as nx
# import matplotlib.pyplot as plt
#
# G = nx.complete_multipartite_graph(28, 16, 10)
# pos = nx.multipartite_layout(G)
# nx.draw(G, pos=pos)
# plt.show()


# import numpy as np
# a = np.array([[0, 1, 0], [0, 1, 1]])
# x = np.random.randint(0, 2, 1)[0]
# print(a)
# print(a[:, 1:])
# print(np.where(a[:, x] == 1)[0], x)
# print(len([None] * 8))
# print(np.random.uniform(0, 1, 1)[0])
# print(np.random.choice(['-', '+', '%'], 1)[0])
# b = 0
#
#
# def a(b):
#     b += 1
#
#
# a(b)
# print(b)

# import itertools
# import matplotlib.pyplot as plt
# import networkx as nx
#
# subset_sizes = [5, 5, 4, 3, 2, 4, 4, 3]
# subset_color = [
#     "gold",
#     "violet",
#     "violet",
#     "violet",
#     "violet",
#     "limegreen",
#     "limegreen",
#     "darkorange",
# ]
#
#
# def multilayered_graph(*subset_sizes):
#     extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
#     layers = [range(start, end) for start, end in extents]
#     G = nx.Graph()
#     for (i, layer) in enumerate(layers):
#         G.add_nodes_from(layer, layer=i)
#     for layer1, layer2 in nx.utils.pairwise(layers):
#         G.add_edges_from(itertools.product(layer1, layer2))
#     return G
#
#
# G = multilayered_graph(*subset_sizes)
# color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
# pos = nx.multipartite_layout(G, subset_key="layer")
# plt.figure(figsize=(8, 8))
# nx.draw(G, pos, node_color=color, with_labels=False)
# plt.axis("equal")
# plt.show()

# import numpy as np
#
# a = np.zeros((4, 4))
# a[2, 2] = 1
# a[2, 3] = 1
# a[1, 3] = 1
#
# print(a)
# print(np.where(a == 1 and a[:, [i for i in range(4)]] == 1)[0])

# import torch
#
# tensor1 = torch.zeros((5, 1))
# tensor2 = torch.zeros((5, 3))
#
# tensor1[0, 0] = 1
# tensor1[1, 0] = 1
# tensor1[2, 0] = 1
#
# tensor2[0, 0] = 1
# tensor2[1, 1] = 0
# tensor2[2, 2] = 1

# a = tensor1 * tensor2
# print(a.size())
# print(a)

# import numpy as np
#
# a = np.ones((200, 1)) * 4
# b = np.ones((1, 894, 3)) * 2
# # c = np.ones((5, 1)) * 4
#
# # print(a * a * b * c)
# print(a @ b)
#
# c = [1, 2, 3]
# print(c[::-1])


import torch
import numpy as np

w = torch.tensor(np.random.random((10, 10)), dtype=torch.float64)
c = torch.tensor(np.random.random((10, 10)), dtype=torch.float64)
r = torch.tensor(np.random.random((10, 1)), dtype=torch.float64)
vm = torch.tensor(np.random.random((10, 1)), dtype=torch.float64)
o = torch.tensor(np.random.random((2, 10)), dtype=torch.float64)
gPr = torch.tensor(np.random.random((2, 5)), dtype=torch.float64)
fP = torch.tensor(np.random.random((2, 10)), dtype=torch.float64)
umv = torch.tensor(np.random.random((10, 3)), dtype=torch.float64)

gU = torch.zeros((2, 10, 10), dtype=torch.float64)
gR = torch.zeros((2, 10), dtype=torch.float64)
gP = torch.zeros((2, 10, 3), dtype=torch.float64)
lwho = torch.zeros((2, 10), dtype=torch.float64)


# print((((gPr[:, 0] * fP[:, 0]).view(-1, 1)
#        * (o * (r / vm).view(1, -1))) * c[:, 0]).size())
gU[:, :, 0] += ((gPr[:, 0] * fP[:, 0]).view(-1, 1)
                * (o * (r / vm).view(1, -1))) * c[:, 0]
print(gU)

# print(((o / vm.view(1, -1))
#        * (gPr[:, 0] * fP[:, 0]).view(-1, 1)).size())
gR += ((o / vm.view(1, -1))
       * (gPr[:, 0] * fP[:, 0]).view(-1, 1)) \
      * w[:, 0]
print(gR)

gP[:, 0, 0] += ((o * (r / vm).view(1, -1))
               * (gPr[:, 0] * fP[:, 0]).view(-1, 1)) \
              @ (w[:, 0] * -umv[:, 0])
gP[:, 0, 1] += ((o * (r / vm).view(1, -1))
               * (gPr[:, 0] * fP[:, 0]).view(-1, 1)) \
              @ (w[:, 0] * -umv[:, 1])
gP[:, 0, 2] += ((o * (r / vm).view(1, -1))
               * (gPr[:, 0] * fP[:, 0]).view(-1, 1)) \
              @ (w[:, 0] * -umv[:, 2])
print(gP[:, 0])

gP[:, :, 0] += ((o * (r / vm).view(1, -1))
               * (gPr[:, 0] * fP[:, 0]).view(-1, 1)) \
              * (w[:, 0] * umv[:, 0])
print(gP)

# print(((r[a, 0] / vm[a, 0]).view(1, -1)
#        * (gPr[:, 0] * fP[:, 0]).view(-1, 1)).size())
lwho += ((r / vm).view(1, -1)
         * (gPr[:, 0] * fP[:, 0]).view(-1, 1)) \
        * w[:, 0]
print(lwho)

arr2D = np.array([[11, 12, 13, 22], [21, 7, 23, 14], [31, 10, 33, 7]])
columnIndex = 0
# Sort 2D numpy array by 2nd Column
sortedArr = arr2D[arr2D[:,columnIndex].argsort()]
print('Sorted 2D Numpy Array')
print(sortedArr)

a = '12345'
print(a[0], a[1])

# # print(i[:, 0].size())
# # print(((o[:, a] * (r[a, 0] / vm[a, 0]).view(1, -1)) @ w[a, 0]).size())
# i = torch.tensor(np.random.random((2, 10)), dtype=torch.float64)
# i[:, 0] = ((o[:, a] * (r[a, 0] / vm[a, 0]).view(1, -1)) @ w[a, 0])
# print(i)

# import numpy as np
#
# a = np.zeros((3, 3), dtype=np.float64)
# a[0, 0] = 1.
# a[1, 1] = 1.
# a[2, 1] = 1.
#
# b = np.eye(3, dtype=np.float64)
#
# print(a)
# print(np.argmax(a, axis=1))
#
# print(b)
#
# print(np.argmax(b, axis=1))
#
# print(np.argmax(a, axis=1) == np.argmax(b, axis=1))
# print(np.where((np.argmax(a, axis=1) == np.argmax(b, axis=1)) == True)[0])
# print(len(np.where((np.argmax(a, axis=1) == np.argmax(b, axis=1)) == True)[0]))


# NUMBA CUDA PROGRAMMING
# import numpy as np
# import math
# from numba import cuda
#
# @cuda.jit()
# def a(v, array, r):
#     # x, y, z = cuda.grid(3)
#     x = cuda.threadIdx.x
#     y = cuda.threadIdx.y
#
#     if x < array.shape[0] and y < array.shape[1]:
#         cumsum = 0
#         for z in range(array.shape[2]):
#             cumsum += (v[x, 0, z] - array[x, y, z]) ** 2
#
#         r[x, y] = math.sqrt(cumsum)
#
#
# B = 1
# N = 16
#
# A = np.ones((B, N, 3)) * 4
# A[:, 0, 0] = 5
#
# v = A[:, 0].reshape((B, 1, 3))
# v = np.ascontiguousarray(v)
#
# r = np.zeros((B, N))
#
# r = cuda.to_device(r)
# v = cuda.to_device(v)
# A = cuda.to_device(A)
#
# threadsPerBlock = (32, 32)
#
# blocksPerGridX = math.ceil(A.shape[1] / threadsPerBlock[0])
# blocksPerGridY = math.ceil(A.shape[2] / threadsPerBlock[1])
# blocksPerGrid = (blocksPerGridX, blocksPerGridY)
#
# a[blocksPerGrid, threadsPerBlock](v, A, r)
#
# print(r.copy_to_host())

# import matplotlib.pyplot as plt
# import numpy as np

a = torch.tensor([[0, 1, 0, 0],
                  [1, 0, 0, 0]])
print(f'a: {torch.argmax(a, dim=1)}')

# from scipy.ndimage import zoom
#
# a = np.zeros((4, 4))
# a[1, 1] = 1.5
# a[1, 2] = 1.
# a[2, 1] = 1.5
# a[2, 2] = 1.
#
# plt.imshow(a, cmap='gray')
# plt.show()
#
# a = zoom(a, 2, mode='mirror')
# plt.imshow(a, cmap='gray')
# plt.show()
