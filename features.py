#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:32:29 2020

@author: krishna
"""

import os
import numpy as np
import librosa
from base import Proprocessor
import pumpp
from scipy.signal import medfilt
from scipy.fft import dct, idct
import numpy as np


class Features(Proprocessor):
    def __init__(self, sr, win_len_ms: float, hop_len_ms: float, n_fft: int, pre_em: bool, vad: bool):
        super().__init__(sr, n_fft, win_len_ms, hop_len_ms)
        self.sr = sr
        self.pre_em = pre_em
        self.vad = vad
        
    def _preprocessing(self, audio_file):
        x,fs = self._load_audio(audio_file)
        if self.pre_em:
            x = self._pre_emp(x)
        if self.vad:
            x = self._trim_silence(x)
        return x
    
    def _mfcc(self, audio_data, n_mels = 13):
        features =  librosa.feature.mfcc(audio_data, sr=self.sr,win_length = self.win_len, hop_length = self.hop_len, n_mels=n_mels)
        return features
    
    def _logmel(self, audio_data, n_mels = 128):
        features = librosa.feature.melspectrogram(audio_data, sr=self.sr,win_length = self.win_len, hop_length = self.hop_len, n_mels = n_mels)
        features = np.log(features+1e-07)
        return features
    
    def _cqt(self, audio_data):
        p_cqt = pumpp.feature.CQT(name='cqt', sr=self.sr, hop_length=256)
        out = p_cqt.transform_audio(audio_data)
        C = out['mag']
        log_cqt = np.log2(np.power(np.abs(C),2))
        #log_cqt = np.power(np.abs(C),2)
        #cqt_feats = np.abs(dct(C)[:,:30]).T
        return log_cqt.T
            
    
    
    def _ifgram(self, audio_data):
        frequencies, _,_ = librosa.reassigned_spectrogram(audio_data, sr=self.sr, n_fft=self.n_fft, win_length=self.win_len, hop_length=self.hop_len)
        mag = np.nan_to_num(frequencies)
        return mag
        
    
    def _mdgf(self, audio_data, rho=0.4, gamma=0.9):
        n_xn = audio_data*range(1,len(audio_data)+1)
        X = self._linear_spectogram(audio_data)
        Y = self._linear_spectogram(n_xn)
        Xr, Xi = np.real(X), np.imag(X)
        Yr, Yi = np.real(Y), np.imag(Y)
        magnitude,_ = librosa.magphase(X,1)
        S = np.square(np.abs(magnitude)) # powerspectrum
        dct_spec = dct(medfilt(S, 5));
        smooth_spec = np.abs(idct(x=dct_spec[:,:30],n=int((self.n_fft/2)+1))) 
        gd = (Xr*Yr + Xi*Yi)/np.power(smooth_spec+1e-05,rho)
        mgd = gd/np.abs(gd)*np.power(np.abs(gd),gamma)
        mgd = np.nan_to_num(mgd)
        mgd = mgd/np.max(mgd)
        cep = np.log2(np.abs(mgd)+1e-08)
        cep = np.nan_to_num(cep)
        return cep.T
        
        
    def _logspec(self,audio_data):
        features = self._linear_spectogram(audio_data)
        magnitude,_ = librosa.magphase(features,2)
        log_spec = np.log(magnitude+1e-05)
        return log_spec.T
    
    
    
    
        
 
class FeatureExtractor(Features):
    def __init__(self, sr, win_len_ms: float, hop_len_ms: float, n_fft: int, pre_em: bool, vad: bool, feat_type:str):
        super().__init__(sr, win_len_ms, hop_len_ms, n_fft, pre_em, vad)
        self.sr = sr
        self.n_fft = n_fft
        self.pre_em = pre_em
        self.vad = vad
        self.feature_type = feat_type
        
        
    def _feature_extraction(self, audio_file):
        x = self._preprocessing(audio_file)
        if self.feature_type=='MFCC':
            feats = self._mfcc(x)
            return feats
        
        elif self.feature_type=='LogSpec':
            feats = self._logspec(x)
            return feats
            
        elif self.feature_type=='MGDF':
            feats = self._mdgf(x)
            return feats
            
        elif self.feature_type=='IFGram':
            feats = self._ifgram(x)
            return feats
            
        elif self.feature_type=='LogMel':
            feats = self._logmel(x)
            return feats
            
        elif self.feature_type=='CQT':
            feats = self._cqt(x)
            return feats
            
        else:
            print(f' Mentioned feature {self.feature_type} is not implemented')
            

if __name__=='__main__':
    sr = 22050
    hop_len_ms = 0.010
    win_len_ms = 0.025
    n_fft = 1024
    audio_file = '/media/newhd/AntiSpoofing2019/PA/ASVspoof2019_PA_eval/flac/PA_E_0000021.flac'
    feats = Features(sr, win_len_ms, hop_len_ms, n_fft, True,True)
    feats = FeatureExtractor(sr, win_len_ms, hop_len_ms, n_fft, True,True,'MGDF')
    out = feats._feature_extraction(audio_file)
    import matplotlib.pyplot as plt
    print(out.shape)
    plt.imshow(out)
    
    