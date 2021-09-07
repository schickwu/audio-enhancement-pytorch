#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import librosa
import torch as th
import IPython.display as ipd
import numpy as np
import scipy
from scipy import signal as sgn

# In[ ]:
""" Information theory intelligibility based loss function 
    Computes the Information theory intelligibility loss combined with a MSE-based loss as regularization term.
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
        [1] Steven Van Kuyk, W. Bastiaan Kleijn, Richard C. Hendriks 'An intelligibility metric based on a simple model of speech communication',
            IWAENC 2016, Xi'an, China.
        [2] Yan Zhao, Buye Xu, Ritwik Giri, Tao Zhang 'Perceptually Guided Speech 
            Enhancement Using Deep Neural Networks',
            ICASSP 2018, Calgary, AB, Canada.
        [3] Jesper Jensen, Cees H. Taal 'Speech Intelligibility Prediction Based on Mutual Information'
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, Feb. 2014
        [4] Steven Van Kuyk, W. Bastiaan Kleijn, Richard C. Hendriks 'An Instrumental Intelligibility Metric Based on Information Theory',
            IEEE Signal Processing Letters, Jan. 2018.
    """



def erb_point(low_freq, high_freq, fraction):
    # compute the central frequency of the gammatone filterbank
    ear_q = 9.26449 # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1
    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear Filter
    # Bank." See pages 33-34.
    erb_point = (
        -ear_q * min_bw
        + np.exp(
            fraction * (
                -np.log(high_freq + ear_q * min_bw)
                + np.log(low_freq + ear_q * min_bw)
                )
        ) *
        (high_freq + ear_q * min_bw)
    )
    
    return erb_point




class myLoss(th.nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, inputs, targets):
        bs=inputs.shape[0]  #get the batch size of input signal
        xi = targets        #set xi as the clean target signal magnitude spectrum
        yi = inputs         #set yi as the denoised input signal magnitude spectrum
        cc = th.linalg.norm(xi-yi) #calculate the Frobenius norm of denoised and clean signal. will be used as regularization term in loss function
 
        xi = th.square(xi)  
        yi = th.square(yi)
        Fs = 16000 # Sampling frequency
        #compute the central frequency of the gammatone filterbank
        fre = erb_point(100,5056,th.arange(1, 25 + 1) / 25)
        fre = th.flipud(fre)
        ear_q = 9.26449 # Glasberg and Moore Parameters
        min_bw = 24.7
        fhop = 16000/512 #frequency hop size(frequency resolution) of spectrums(stft)
        #compute the freedom degrees of clean speech signal spectrum as in [3]. section B, with the assumption that it follows a chi distribution, and these degrees will be used in [1] Eq. (9)
        d = th.zeros(25)
        eb = th.zeros(25)
        for t in range(25):
            eb[t] = ((fre[t]/ear_q) + min_bw)
            d[t] = 2*th.floor((eb[t])/fhop)
        #compute the frequency response of 25-gammatone filterbank between 100 Hz and 4500 Hz on the ERB-rate scale 
        Ht = np.zeros((25,257),dtype = 'complex_')
        for idx in range(25):
            fil,filb= sgn.gammatone(fre[idx],'iir',fs=16000)
            w,Ht[idx,:]= sgn.freqz(fil,filb,257)
        Ht = th.Tensor(Ht)
        #Apply gammatone filterbank matrix to the spectrograms as in [1] Eq. (1)
        X= th.matmul(th.square(th.abs(Ht)),xi)
        Y= th.matmul(th.square(th.abs(Ht)),yi)
        X= th.sqrt(X)
        Y= th.sqrt(Y)
        e = th.exp(th.tensor([1.0])) #exponential value
        pi= np.pi                    #  pi value
        #compute the information rate of the speech production based on [4] Eq. (10)
        R = th.ones(bs,25)
        R = R*(-0.5*np.log2((1-0.75*0.75))/1.44)
        R = th.tensor(R)
        I = th.ones(bs,25) # initialize a tensor to store the calculated information rates later
        S = Y*X #compute the X*Y
        X2 =X**2 #compute the X^2
        Y2 =Y**2 #compute the Y^2
        #compute the lower bound on information rate between clean and noisy signals as in [1] Eq. (9)
        g0= th.lgamma(d/2)+0.5*(d-th.log(th.tensor(2.0))-(d-1)*th.digamma(d/2))-0.5*th.log(2*pi*e*(d-2*(th.pow(th.exp(th.lgamma((d+1)/2))/th.exp(th.lgamma(d/2)),2))))
        g0 = g0.repeat(bs,1)
        aaaa=(th.mean(S,axis=2)-th.mean(X,axis=2)*th.mean(Y,axis=2))/th.sqrt((th.mean(X2,axis=2)-th.mean(X,axis=2)**2)*(th.mean(Y2,axis=2)-th.mean(Y,axis=2)**2)) #compute the correlation coefficients between clean and noisy spectrum, based on [1] Eq. (10)
        corre2 = -th.log(1-th.square(aaaa))/2 # in [1] Eq. (9) last term
        iou = corre2+g0 
        #compute the intelligibility value as in [1] Eq. (11)
        iou = th.minimum(iou,R)
        ff = th.sum(iou) 
        ff = ff/th.sum(R)
        print('ff',ff)
        ff = (1-ff)**2 #compute the info theory loss term
        lo = ff+ 0.005*cc/(48*bs)  #calculate the info theory loss with L2 regularization, as described in [2] Eq. (7), here, 0.005 is a tunable hyper-parameter used to balance the two terms in the loss function 

        return lo

