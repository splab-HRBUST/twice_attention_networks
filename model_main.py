"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import argparse
import sys
import os
from librosa.core import spectrum
from scipy.interpolate.fitpack2 import SmoothBivariateSpline
import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from models import SpectrogramModel, CQCCModel, resnet18_cbam, MFCCModel
# from models import SpectrogramModel,CQCCModel,MFCCModel
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from eval_metrics import compute_eer
from scipy.signal import medfilt
from scipy import signal
# from dct import dct2,idct2
from dct_self import dct2,idct2
#=============================
import time
from splice_scores import cat_scores

#==========================对cqt语谱图的低频和高频进行重新采样，并进行一、二阶差分=============================
def diff(matrix): 
    print("matrix.shape = ",matrix.shape)
    b1 = np.diff(matrix,axis=1)
    b2 = np.diff(b1,axis =1)
    b3 = np.append(matrix,b1,axis=1)
    c = np.append(b3,b2,axis= 1)
    return c
#=======================================================


#========================cqt语谱图====================================
def cqtgram_true(y,sr = 16000, hop_length=256, octave_bins=24, 
n_octaves=8, fmin=21, perceptual_weighting=False):


    s_complex = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
    )
    specgram = np.abs(s_complex)
    # if代码块可以不要。
    if perceptual_weighting:
       # 功率谱的感知权重：S_p[f] = frequency_weighting(f, 'A') + 10*log(S[f] / ref);
        freqs = librosa.cqt_frequencies(specgram.shape[0], fmin=fmin, bins_per_octave=octave_bins)#返回每一个cqt频率带的中心频率。
        specgram = librosa.perceptual_weighting(specgram ** 2, freqs, ref=np.max)#功率谱的感知加权。
    else:
        specgram = librosa.amplitude_to_db(specgram, ref=np.max)#将振幅谱转为用分贝表示的谱图。
    return specgram
# ======================================计算gd_gram============================================
def cqtgram(y,sr = 16000, hop_length=256, octave_bins=24, 
n_octaves=8, fmin=21, perceptual_weighting=False):
    rho=0.4
    gamma=0.9
    n_xn = y*range(1,len(y)+1)
    X = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
        window = "hamming"
    )
    Y = librosa.cqt(
        n_xn,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=octave_bins,
        n_bins=octave_bins * n_octaves,
        fmin=fmin,
        window = "hamming"
    )
    Xr, Xi = np.real(X), np.imag(X)
    Yr, Yi = np.real(Y), np.imag(Y)
    magnitude,_ = librosa.magphase(X,1)# magnitude:幅度，_:相位
    S = np.square(np.abs(magnitude)) # powerspectrum, S =   (192, 126)
    """
    medifilt中值滤波：
    中值滤波的基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，
    让周围的像素值接近真实值，从而消除孤立的噪声点
    """
    a = medfilt(S, 5) #a.shape =  (192, 251)
    dct_spec = dct2(a) # dct_spec.shape =  (192, 251)
    smooth_spec = np.abs(idct2(dct_spec[:,:291]))# smooth_spec.shape =  (192, 251)
    # smooth_spec = np.abs(a)
    gd = (Xr*Yr + Xi*Yi)/np.power(smooth_spec+1e-05,rho)#对振幅的每个值都进行0.4次方处理。
    mgd = gd/(np.abs(gd)*np.power(np.abs(gd),gamma)+1e-10)
    mgd = mgd/np.max(mgd)
    cep = np.log2(np.abs(mgd)+1e-08)
    return cep
    

def stftgram(audio_data):
    sr = 16000
    hop_len_ms = 0.010
    win_len_ms = 0.025
    n_fft=1024
    rho=0.4
    gamma=0.9
    n_xn = audio_data*range(1,len(audio_data)+1)
    X = librosa.stft(audio_data, n_fft=n_fft, win_length = int(win_len_ms*sr), hop_length = int(hop_len_ms*sr))
    Y = librosa.stft(n_xn, n_fft=n_fft, win_length = int(win_len_ms*sr), hop_length = int(hop_len_ms*sr))
    # X = librosa.stft(audio_data,center=False)
    # Y = librosa.stft(n_xn,center=False)
    Xr, Xi = np.real(X), np.imag(X)
    Yr, Yi = np.real(Y), np.imag(Y)
    magnitude,_ = librosa.magphase(X,1)# magnitude:幅度，_:相位
    S = np.square(np.abs(magnitude)) # powerspectrum, S.shape =  (513, 291)
    """
    区别：
    1）是stft中的参数设置不同；
    2）是对振幅的处理不同；
    3）对中值滤波后的振幅进行dct和idct
    medifilt中值滤波：
    中值滤波的基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替，
    让周围的像素值接近真实值，从而消除孤立的噪声点
    """
    dct_spec = dct2(medfilt(S, 5)) # dct_spec = (513, 401) 
    smooth_spec = np.abs(idct2(dct_spec))
    gd = (Xr*Yr + Xi*Yi)/np.power(smooth_spec+1e-05,rho)#对振幅的每个值都进行0.4次方处理。
    mgd = gd/(np.abs(gd)*np.power(np.abs(gd),gamma)+1e-10)
    mgd = mgd/np.max(mgd)
    cep = np.log2(np.abs(mgd)+1e-08)
    cep = np.nan_to_num(cep)
    return cep.T
# ======================================================================================
def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def evaluate_accuracy(data_loader, model, device):
    dev_scores = []
    dev_y = []
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()




#============================================================
        batch_score_dev = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().tolist()
        batch_y = batch_y.data.cpu().numpy().tolist()
        dev_scores.extend(batch_score_dev)
        dev_y.extend(batch_y)
    length = len(dev_y)
    dev_target_scores,dev_nontarget_scores = cat_scores(dev_scores,dev_y,length)
    dev_eer,threshold = compute_eer(dev_target_scores,dev_nontarget_scores)
    print("dev_eer = ",dev_eer)
    print("dev_threshold = ",threshold)
#==============================================================
    return 100 * (num_correct / num_total)


def produce_evaluation_file(dataset, model, device, save_path):
    eval_scores = []
    eval_y = []
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    for batch_x, batch_y, batch_meta in data_loader:
        '''
        batch_meta =  ASVFile(speaker_id, file_name, path, sys_id, key)
        '''

        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()
        
        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())
#============================================================
        batch_score_eval = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().tolist()
        batch_y = batch_y.data.cpu().numpy().tolist()
        eval_scores.extend(batch_score_eval)
        eval_y.extend(batch_y)
    length = len(eval_y)
    dev_target_scores,dev_nontarget_scores = cat_scores(eval_scores,eval_y,length)
    dev_eer,threshold = compute_eer(dev_target_scores,dev_nontarget_scores)
    print("eval_eer = ",dev_eer)
    print("eval_threshold = ",threshold)
#==============================================================
    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            if not dataset.is_eval:
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
            else:
                 fh.write('{} {} {} {}\n'.format(f, s, k, cm))
    print('Result saved to {}'.format(save_path))

def train_epoch(data_loader, model, lr, device):
    train_scores = []
    train_y = []
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train() # 作用是启用batch normalization和drop out
    optim = torch.optim.Adam(model.parameters(), lr=lr) # 优化算法
    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    # criterion = nn.NLLLoss(weight=weight)
    criterion = nn.CrossEntropyLoss(weight=weight)
    start_time = time.time()
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)# batch_size=32
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().ravel()
        batch_loss = criterion(batch_out, batch_y)

        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            # 输出正确率
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        # ==============EER==============
        batch_score_train = (batch_out[:, 1] - batch_out[:, 0]).data.cpu().numpy().tolist()
        batch_y = batch_y.data.cpu().numpy().tolist()
        train_scores.extend(batch_score_train)
        train_y.extend(batch_y)
        # ================================
    end_time = time.time()
    print("\n一轮的训练时间：",end_time-start_time)
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
# =============EER======================
    length = len(train_y)
    train_target_scores,train_nontarget_scores = cat_scores(train_scores,train_y,length)
    train_eer,threshold = compute_eer(train_target_scores,train_nontarget_scores)
    print("train_eer = ",train_eer)
    print("train_threshold = ",threshold)
    return running_loss, train_accuracy
# ======================================

def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s)**2
    #melspect = librosa.feature.melspectrogram(S=a)
    feat = librosa.power_to_db(a)
    return feat


def compute_mfcc_feats(x):
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)
    return feats

if __name__ == '__main__':
    print("开始执行!")
    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved mdoel')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='spect')
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    # device = [0,1,2,3,4,5,6,7] if torch.cuda.is_available() else 'cpu'
    device = "cuda:6" if torch.cuda.is_available() else 'cpu'
    print("device:",device)
    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    track = args.track # track = logical
    assert args.features in ['mfcc', 'spect', 'cqcc'], 'Not supported feature'
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.features, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)    
    print("model_save_path = ",model_save_path)
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')
    # 该if语句不执行
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    if args.features == 'mfcc':
        feature_fn = compute_mfcc_feats
        model_cls = MFCCModel
    elif args.features == 'spect':
        # feature_fn = get_log_spectrum
        feature_fn = cqtgram_true
        # feature_fn = cqtgram
        # feature_fn = stftgram
        model_cls = resnet18_cbam
        # model_cls = SpectrogramModel
    elif args.features == 'cqcc':
        feature_fn = None  # cqcc feature is extracted in Matlab script
        # model_cls = CQCCModel
        model_cls = resnet18_cbam
        # model_cls = SpectrogramModel



    transforms = transforms.Compose([
        lambda x: pad(x),
        # lambda x: librosa.util.normalize(x),
        lambda x: feature_fn(x),
        lambda x: Tensor(x)
    ])

    dev_set = data_utils.ASVDataset(is_train=False, is_logical=is_logical,
                                    transform=transforms,
                                    feature_name=args.features, is_eval=args.eval, eval_part=args.eval_part)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,pin_memory=True,num_workers=64)
    model = model_cls().to(device) 
   

    if args.model_path:
        a = torch.load(args.model_path,map_location="cuda:6")
        model.load_state_dict(a)
        print('Model loaded : {}'.format(args.model_path))
    if args.eval:
        assert args.eval_output is not None, 'You must provide an output path'
        assert args.model_path is not None, 'You must provide model checkpoint'
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0) # 无错误退出，1是有错误退出
    train_set = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms,
                                      feature_name=args.features)
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,pin_memory=False,num_workers=64)
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss, train_accuracy = train_epoch(         
            train_loader, model, args.lr, device) 
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
