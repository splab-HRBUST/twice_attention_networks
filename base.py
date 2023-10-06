#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 23:17:05 2020

@author: krishna
"""

import librosa
import numpy as np


class Proprocessor:
    def __init__(self, sr: int, n_fft: int, win_len_ms: float, hop_len_ms: float):
        self.sr = sr
        self.win_len_ms = win_len_ms
        self.hop_len_ms = hop_len_ms
        self.n_fft = n_fft
      
        
        if self.sr:
            self.win_len = int(self.win_len_ms*self.sr)
            self.hop_len = int(self.hop_len_ms*self.sr)
        else:
            print('Please provide the sampling rate')
            
    
    def _load_audio(self,file):
        audio_data, fs = librosa.load(file, sr=self.sr)
        return audio_data, fs
    
    
    def _trim_silence(self,audio, threshold=0.05, frame_length=2048):
        if audio.size < frame_length:
            frame_length = audio.size
        energy = librosa.feature.rms(audio, frame_length=frame_length)
        frames = np.nonzero(energy > threshold)
        indices = librosa.core.frames_to_samples(frames)[1]
        return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]
        

    def _linear_spectogram(self, audio_data ):
        if self.n_fft:
            linear = librosa.stft(audio_data, n_fft=self.n_fft, win_length=self.win_len, hop_length=self.hop_len) # linear spectrogram
        else:
            print('Mention Number of FFT bins')
            raise
            
        return linear.T
        
    
    def _pre_emp(self,audio_data):
        '''
        Apply pre-emphasis to given utterance.
        x: audio signal
        '''
        return np.append(audio_data[0], np.asarray(audio_data[1:] - 0.97 * audio_data[:-1], dtype=np.float32))
    
    
    
    
    