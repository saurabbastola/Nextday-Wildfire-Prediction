import numpy as np

def avg_neighbors(array_in):
    if array_in.shape[0] != array_in.shape[1]:
        raise Exception('Only square arrays make sense here, since you\'re analyzing square arrays.')
    n = array_in.shape[0]
    array_out = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sum_neighbors = 0
            count_neighbors = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        sum_neighbors += array_in[ni, nj]
                        count_neighbors += 1
            array_out[i, j] = sum_neighbors / count_neighbors

    return array_out
