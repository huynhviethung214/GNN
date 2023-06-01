from numba import cuda
import numpy as np


@cuda.jit
def f(I, O, u):
    x, y = cuda.grid(2)

    if x <= I.shape[0] and y <= I.shape[1]:
        I[x, y] = u[x, y]
        O[x, y] = u[x, y]


I = np.zeros((10, 5))
O = np.zeros((10, 5))
u = np.random.uniform(0, 1, (10, 5))

print(u)

I = cuda.to_device(I)
u = cuda.to_device(u)

f[(32, 32), (16, 16), 0, 2560](I, O, u)

print(np.array(I[:, 1]))
print(np.array(O[:, 1]))
print(np.array(u[:, 1]))
