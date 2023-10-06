# import numpy as np

# import matplotlib.pyplot as plt

# def dct(v):
#     """
#         DCT for one dimension np.array.
#         v: Is a np.array of one dimension [1:N]
#         return: Discrete cosine transform of v
#     """
#     N = v.shape[0]
#     c = np.zeros(N)  # [0:N-1]

#     sum = 0
#     for j in range(0, N):
#         sum = sum + (np.cos(0 * np.pi * (2 * (j + 1) - 1) / (2 * N)) * v[j])

#     c[0] = np.sqrt(1 / N) * sum

#     for k in range(1, N):
#         sum = 0
#         for j in range(0, N):
#             sum = sum + (np.cos(k * np.pi * (2 * (j + 1) - 1) / (2 * N)) * v[j])

#         c[k] = np.sqrt(2 / N) * sum

#     return c




# def dct2(matrix):
#     """
#         DCT for two dimension np.array.
#         matrix: Is a np.array of one dimension [M:N]
#         return: Discrete cosine transform of matrix
#     """
#     N = matrix.shape[0]
#     M = matrix.shape[1]
#     C1 = np.zeros(shape=(N, N))
#     C2 = np.zeros(shape=(M, M))
#     Z = np.zeros(matrix.shape)

#     for j in range(N):
#         C1[0, j] = np.sqrt(1 / N)

#     for k in range(1, N):
#         for j in range(N):
#             C1[k, j] = np.cos(k * np.pi * (2 * (j + 1) - 1) / (2 * N)) * np.sqrt(2 / N)


#     for j in range(M):
#         C2[j, 0] = np.sqrt(1 / M)

#     for k in range(1, M):
#         for j in range(M):
#             C2[j, k] = np.cos(k * np.pi * (2 * (j + 1) - 1) / (2 * M)) * np.sqrt(2 / M)


#     Z = np.dot(C1, matrix)
#     Z = np.dot(Z, C2)
#     C_t = np.linalg.inv(C1)
#     return Z





# def idct(c):
#     """
#         IDCT for one dimension np.array.
#         c: Is a np.array of one dimension [0:N-1]
#         return: Inverse discrete cosine transform of c
#     """
#     N = c.shape[0]
#     v = np.zeros(N)
#     for j in range(0, N):
#         sum = 0
#         for k in range(0, N):
#             if k == 0:
#                 a_k = np.sqrt(1 / N)
#             else:
#                 a_k = np.sqrt(2 / N)

#             sum = sum + (np.cos(k * np.pi * (2 * (j + 1) - 1) / (2 * N)) * c[k] * a_k)


#         v[j] = sum
#     return v




# def idct2(matrix):
#     """
#         IDCT for two dimension np.array.
#         matrix: Is a np.array of one dimension [M:N]
#         return: Inverse discrete cosine transform of matrix
#     """
#     M = matrix.shape[0]
#     N = matrix.shape[1]
#     matrix_r = np.zeros(shape=(M, N), dtype=np.float64)  # To store the discrete cosine transform
#     for i in range(0, M):
#         matrix_r[i] = idct(matrix[i])

#     for j in range(0, N):
#         temp = idct(matrix_r[:, j])
#         for k in range(0, M):
#             matrix_r[k, j] = temp[k]

#     return matrix_r



#==================================================================================
import numpy as np


def dct2(matrix):
    """
        DCT for two dimension np.array.
        matrix: Is a np.array of one dimension [M:N]
        return: Discrete cosine transform of matrix
    """
    """
    计算所需参数值：Q,f_k1,fs,B
    """
    Q = 1.0 / (np.power(2, 1.0 / 24) - 1)
    fs = 16000
    f_min = 21
    B = 24
    N = matrix.shape[0]
    M = matrix.shape[1]
    global C1
    global C2
    C1 = np.zeros(shape=(N, N))
    C2 = np.zeros(shape=(M, M))

    for k in range(0, N):
        for j in range(N):
            C1[k, j] = ((f_min * np.power(2, k / B)) / (Q * fs)) * (
                np.cos(((2 * j + 1) * np.pi * f_min * np.power(2, k / B)) / (2 * fs)))

    for k in range(0, M):
        for j in range(M):
            C2[j, k] = ((f_min * np.power(2, k / B)) / (Q * fs)) * (
                np.cos(((2 * j + 1) * np.pi * f_min * np.power(2, k / B)) / (2 * fs)))
    Z = np.dot(C1, matrix)
    Z = np.dot(Z, C2)
    return Z


def idct2(matrix):
    """
        IDCT for two dimension np.array.
        matrix: Is a np.array of one dimension [M:N]
        return: Inverse discrete cosine transform of matrix
    """
    global C1
    global C2
    C1_inv = np.linalg.inv(C1)
    C2_inv = np.linalg.inv(C2)

    Z = np.dot(C1_inv, matrix)
    Z = np.dot(Z, C2_inv)
    return Z


if __name__ == '__main__':
    a = np.array(
        [[1, 2], [4, 4]])
    dct_a = dct2(a)

    i_a = idct2(dct_a)
    print("i_a = ", i_a)
