#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import sys
import librosa
import torch as th
import IPython.display as ipd
import numpy as np
import torch.utils.data
import torch.nn as nn
from pystoi import utils # https://github.com/mpariente/pystoi/tree/b243f8e5e98a45cc731916c8d45415a845f90f3f/pystoi


""" Short term objective intelligibility
    Computes the Stoi loss combined with a MSE-based loss as regularization term.
    first computes the STOI (See [1][2]) of a denoised signal compared to a clean
    signal, The output is expected to have a monotonic relation with the
    subjective speech-intelligibility, where a higher score denotes better
    speech intelligibility.
    # Arguments
        xi (th.Tensor): clean original speech magnitude spectrum
        yi (th.Tensor): denoised speech magnitude spectrum
    # Returns
        loss value(tensor): loss value based on Short time objective intelligibility measure between clean and denoised speech combined with a MSE-based loss as regularization term 
    # Reference
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
            Objective Intelligibility Measure for Time-Frequency Weighted Noisy
            Speech', ICASSP 2010, Texas, Dallas.
        [2] Yan Zhao, Buye Xu, Ritwik Giri, Tao Zhang 'Perceptually Guided Speech 
            Enhancement Using Deep Neural Networks',
            ICASSP 2018, Calgary, AB, Canada.
    # We utilized the one-third octave bands matrix calculation from https://github.com/mpariente/pystoi/blob/b243f8e5e98a45cc731916c8d45415a845f90f3f/pystoi/utils.py
    """


class stoiLoss(nn.Module):
    def __init__(self):
        super(stoiLoss, self).__init__()
    def forward(self, inputs, targets):
        N = 24                              # N. frames for intermediate intelligibility
        FS = 16000                          # Sampling frequency
        N_FRAME = 512                       # Window support
        BETA = -15.                         # Lower SDR bound as described in [2] Eq. (3)
        NFFT = 512                          # FFT Size
        NUMBAND = 15                        # Number of 1/3 octave band
        MINFREQ = 150                       # Center frequency of 1st octave band (Hz)
        bs = inputs.shape[0] # batch size of input signal
        xi = targets     #set xi as the clean target signal magnitude spectrum
        yi = inputs      #set yi as the denoised input signal magnitude spectrum
        cc = th.linalg.norm(xi-yi) #calculate the Frobenius norm of denoised and clean signal. will be used as regularization term in loss function
        OBM, CF = utils.thirdoct(FS, NFFT, NUMBAND, MINFREQ)  # Get 1/3 octave band matrix
        OBM = th.tensor(OBM).float()
        #Apply Octave Bands matrix to the spectrograms as in [2] Eq. (1)
        x_segments = th.sqrt(th.matmul(OBM,th.square(xi))) 
        y_segments = th.sqrt(th.matmul(OBM,th.square(yi)))
        #Find normalization constants and normalize as in [2] Eq. (3)
        normalization_consts = (
            th.linalg.norm(x_segments, axis=2, keepdims=True) /
            th.linalg.norm(y_segments, axis=2, keepdims=True))
        y_segments_normalized = y_segments * normalization_consts

        # Clip as described in [2] Eq. (3)
        clip_value = 10 ** (-BETA / 20)
        y_primes = th.minimum(y_segments_normalized, x_segments * (1 + clip_value))

        # Subtract mean vectors in [2] Eq. (4)
        y_primes = y_primes - th.mean(y_primes, axis=2, keepdims=True)
        x_segments = x_segments - th.mean(x_segments, axis=2, keepdims=True)

        # Divide by their norms in [2] Eq. (4)
        y_primes = y_primes/th.linalg.norm(y_primes, axis=2, keepdims=True) 
        x_segments = x_segments/th.linalg.norm(x_segments, axis=2, keepdims=True) 
        # Find a matrix with entries summing to sum of correlations of vectors
        correlations_components = y_primes * x_segments
        J = x_segments.shape[0] # size of the one-third octave bands, (here 15 according to [2])
        M = x_segments.shape[1] # number of time frames used for stoi calculation (here refers to the batch size of the input signal)
        # Find the mean of all correlations as in [2] Eq. (5)
        d = th.sum(correlations_components) / (J*M)
        print('stoi',d)
        sss = (1-d)**2 #compute the stoi loss term 
        d = sss+ 0.01*cc/(24*M) #calculate the stoi loss with L2 regularization, as described in [2] Eq. (7), here, 0.01 is a tunable hyper-parameter used to balance the two terms in the loss function 
        return d


# In[37]:





# In[ ]:




