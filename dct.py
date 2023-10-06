import numpy as np

B = 24
Q = 1.0 / (np.power(2, 1.0 / B) - 1)
f1 = 21
fs = 16000


def N_k(k):
    N_k = (fs * Q) / (f1 * np.power(2, k / B))
    return N_k


def dct2(matrix):
    M = matrix.shape[0]
    N = matrix.shape[1]
    global C1
    global C2
    C1 = np.zeros(shape=(M, M))
    C2 = np.zeros(shape=(N, N))

    for i in range(M):
        N_k1 = N_k(0)
        C1[0,i] = (1.0 / N_k1) * np.cos(np.pi * Q * (i + 0.5) / N_k1) 
    for i in range(1, M):
        N_k1 = N_k(i)
        # Nk1 = np.minimum(int(np.floor(N_k1)), M)
        for n1 in range(0,M):
            C1[i, n1] = (1.0 / N_k1) * np.cos(np.pi * Q * (n1 + 0.5) / N_k1)

    for i in range(N):
        N_k2 = N_k(0)
        C2[0,i] = (1.0 / N_k2) * np.cos(np.pi * Q * (i + 0.5) / N_k2)
    for j in range(1, N):
        N_k2 = N_k(j)
        # Nk2 = np.minimum(int(np.floor(N_k2)), N)
        for n2 in range(0,N):
            C2[j, n2] = (1.0 / N_k2) * np.cos(np.pi * Q * (n2 + 0.5) / N_k2)
    Z = np.dot(C1, matrix)
    Z = np.dot(Z, C2)
    return Z


def idct2(matrix):
    C1_inv = np.linalg.inv(C1)
    C2_inv = np.linalg.inv(C2)
    Z = np.dot(C1_inv, matrix)
    Z = np.dot(Z, C2_inv)
    return Z


if __name__ == "__main__":
    a = np.random.randint(-5,5,size=(3,4))
    print("a = ", a)
    a_dct = dct2(a)
    newA = idct2(a_dct)
    print("newA = ", newA)
