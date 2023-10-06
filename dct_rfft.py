# 非求和方式
import torch
import numpy as np
import pandas as pd



def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(x, norm=None):
    x1 = idct(x, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def _rfft(x, signal_ndim=1, onesided=True):
    odd_shape1 = (x.shape[1] % 2 != 0)
    x = torch.fft.rfft(x)
    x = torch.cat([x.real.unsqueeze(dim=2), x.imag.unsqueeze(dim=2)], dim=2)
    if onesided == False:
        _x = x[:, 1:, :].flip(dims=[1]).clone() if odd_shape1 else x[:, 1:-1, :].flip(dims=[1]).clone()
        _x[:, :, 1] = -1 * _x[:, :, 1]
        x = torch.cat([x, _x], dim=1)
    return x


def _irfft(x, signal_ndim=1, onesided=True):
    if onesided == False:
        res_shape1 = x.shape[1]
        x = x[:, :(x.shape[1] // 2 + 1), :]
        x = torch.complex(x[:, :, 0].float(), x[:, :, 1].float())
        x = torch.fft.irfft(x, n=res_shape1)
    else:
        x = torch.complex(x[:, :, 0].float(), x[:, :, 1].float())
        x = torch.fft.irfft(x)
    return x


def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = _rfft(v, 1, onesided=False)
#-----------------------------------------------------
    # Q = 1 / (pow(2, 1.0 / 24) - 1)
    # k = - torch.full([1,N],Q)* np.pi / (2 * N)
    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
#----------------------------------------------------
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
#-------------------------------------------------------------------
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    # Q = 1 / (pow(2, 1.0 / 24) - 1)
    # k = torch.full([1,x_shape[-1]],Q)* np.pi / (2 * N)
#-------------------------------------------------------------------
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


if __name__ == '__main__':
    x = torch.rand(100, 1024)
    print("x = ", x)
    y = dct(x)
    print("y = ", y)
    z = idct(y)
    print("z = ", z)
