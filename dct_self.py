import numpy as np

global C1
C1 = np.array([])

global C2
C2 = np.array([])


def dct2(matrix):
    """
        DCT for two dimension np.array.
        matrix: Is a np.array of one dimension [M:N]
        return: Discrete cosine transform of matrix
    """
    N = matrix.shape[0]
    M = matrix.shape[1]
    global C1
    C1 = np.zeros(shape=(N, N))
    global C2
    C2 = np.zeros(shape=(M, M))

    for j in range(N):
        C1[0, j] = np.sqrt(1 / N)

    for k in range(1, N):
        for j in range(N):
            C1[k, j] = np.cos(k * np.pi * (2 * (j + 1) - 1) / (2 * N)) * np.sqrt(2 / N)

    for j in range(M):
        C2[j, 0] = np.sqrt(1 / M)

    for k in range(1, M):
        for j in range(M):
            C2[j, k] = np.cos(k * np.pi * (2 * (j + 1) - 1) / (2 * M)) * np.sqrt(2 / M)

    Z = np.dot(C1, matrix)
    Z = np.dot(Z, C2)
    return Z


def idct2(matrix):
    """
        IDCT for two dimension np.array.
        matrix: Is a np.array of one dimension [M:N]
        return: Inverse discrete cosine transform of matrix
    """
    C1_inv = np.linalg.inv(C1)
    C2_inv = np.linalg.inv(C2)
    Z = np.dot(C1_inv, matrix)
    Z = np.dot(Z, C2_inv)
    return Z


if __name__ == '__main__':
    a = np.array([[1, 2, 3,3,3,3,3,3,3], [4, 5, 6,3,3,3,3,4,3]])
    print(a)
    dct_a = dct2(a)
    print("dct_a = ", dct_a)
    idct_a = idct2(dct_a)
    print("idct_a = ", idct_a)
