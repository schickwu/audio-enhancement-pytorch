#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import librosa
import torch as th
import IPython.display as ipd
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import soundfile as sf
import pystoi  #stoi value computing from https://github.com/mpariente/pystoi/tree/b243f8e5e98a45cc731916c8d45415a845f90f3f/pystoi
from pesq import pesq #pesq value computing from https://github.com/ludlows/python-pesq


# In[3]:


class Testset(Dataset):
    def __init__(self, train_clean_audio_path, train_noisy_audio_path):
        fc = open(train_clean_audio_path, 'r')
        fn = open(train_noisy_audio_path, 'r')

        audioc = []
        audion = []
        linesc = fc.readlines()
        linesn = fn.readlines()
        
        for line in linesc:
            line = line.rstrip('\n')
            audioc.append(line)
            self.audioc = audioc
        for line in linesn:
            line = line.rstrip('\n')
            audion.append(line)
            self.audion = audion
        fc.close()
        fn.close()
    def __getitem__(self, index):
        Fs = 16000
        fc = self.audioc[index]
        fn = self.audion[index]
        N_FRAME = 512                       # Window support
        NFFT = 512                          # FFT Size
        x, Fs = librosa.load(fn, sr=Fs)
        y, Fs = librosa.load(fc, sr=Fs)
        stoibe = pystoi.stoi(y, x, 16000, extended=False) #compute the stoi value before enhancement
        pesqbe = pesq(16000, y, x, 'wb')  #compute the pesq value before enhancement
        x_spec = librosa.stft(x,n_fft=512, hop_length=256, win_length=512, window='hann')
        y_spec = librosa.stft(y,n_fft=512, hop_length=256, win_length=512, window='hann')
        magnitude, phase = librosa.magphase(x_spec) #get the phase information of noisy signal
        ymagnitude, yphase = librosa.magphase(y_spec)  #get the phase information of clean signal
        shape = x_spec.shape[1]
        x_spec = np.abs(x_spec)
        y_spec = np.abs(y_spec)
        x_norm = np.log(x_spec**2) #compute the log power spectrum
        x_norm = np.array(
            [x_norm[:, (m-2) :(m+3)] for m in range(2,shape-2 )])
        x_norm= np.reshape(x_norm,(shape-4,1285))
        #normlized the input features to zero mean and unit variance
        x_norm= (x_norm- np.mean(x_norm,axis=1, keepdims=True))/np.std(x_norm,axis=1,keepdims=True)
        x_spec = th.Tensor(x_spec)
        y_spec = th.Tensor(y_spec)    
        x_norm = th.Tensor(x_norm)
        return x_spec,y_spec,x_norm,stoibe,phase,pesqbe,yphase
    def __len__(self):
        return len(self.audioc)


# In[4]:



testc = "/Users/schicksal/Desktop/intelligibility enhancement/testclean16k&2.5snr.txt" #clean test dataset
testn = "/Users/schicksal/Desktop/intelligibility enhancement/testnoisy16k&2.5snr.txt" #noisy test dataset
testset = Testset( train_clean_audio_path = testc, train_noisy_audio_path = testn)


# In[13]:


class Net(nn.Module):  
    # structure of the neural network for masking: three hidden layers with 1024 exponential linea units(ELUs)
    # each layer with a dropout layer (for dropout regularization) with dropout rate of 0.3,
    # for the output layer, sigmoid activation units was imployed to bound the value between 0 and 1
    # the input of the network is 5-frame continuous log power spectrums of noisy speech signal
    # output of the network is the predicted 1-frame ideal ratio mask (IRM)
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden  = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.output  = nn.Linear(n_hidden,n_output)        
    def forward(self,x):
        m = nn.ELU()   #ELUs for hidden layers
        si = nn.Sigmoid()  #sigmoid unit for output layer
        dp = nn.Dropout(p=0.3)   #dropout layer with 0.3 dropout rate
        x= self.hidden(x)  
        x= dp(x)
        x= m(x)
        x= self.hidden2(x)
        x= dp(x)
        x= m(x)
        x= self.hidden3(x)
        x= dp(x)
        x= m(x)
        x= self.output(x)
        x= dp(x)
        x= si(x)
        return x


# In[11]:


def MaskTest(epochs, testLoader, model, device = "cpu"):
    losst=0
    lossbb=0
    pesqbb=0
    pesqt=0
    for i, (x,y,z, stoibe,phase,pesqbe,yphase) in enumerate(test_loader):
        sh = y.shape[2]
        mask = model(z) #generate mask for noisy signal based on the log power specturm of noisy signal
        mask = th.reshape(mask,(sh-4,257))
        mask = th.transpose(mask,1,0)
        x = th.reshape(x,(257,sh))
        y = th.reshape(y,(257,sh))
        x[:,2:sh-2] = mask*x[:,2:sh-2] # apply the estimated mask to the noisy speech spectrum
        x=x.detach().numpy()
        y=y.detach().numpy()
        phase=phase.detach().numpy()  #phase information of the noisy signal        
        yphase=yphase.detach().numpy() #phase information of the clean signal
        length = phase.shape[2]
        phase = np.reshape(phase,(257,length))
        yphase = np.reshape(yphase,(257,length))
        xox = x*phase #reconstruct the noisy stft from magnitude spectrum   
        yoy = y*yphase  #reconstruct the clean stft from magnitude spectrum
        xx =librosa.istft(xox,hop_length=256, win_length=512, window='hann')
        yy =librosa.istft(yoy,hop_length=256, win_length=512, window='hann')
        pesqaf = pesq(16000, yy, xx, 'wb') #compute the pesq value after enhancement
        stoiaf = pystoi.stoi(yy, xx, 16000, extended=False) #compute the stoi value after enhancement
        lossbb = lossbb+stoibe
        losst=losst+stoiaf
        pesqbb = pesqbb+pesqbe
        pesqt=pesqt+pesqaf
        print(' Testloss: {}',i,stoibe,stoiaf,pesqbe,pesqaf)
    print('befor&after',lossbb/i,losst/i,pesqbb/i,pesqt/i)


# In[15]:


model1 = torch.load('beststoi20bsnew.pth') #load the tained model and parameters for mask estimation
batch_size= 1
test_db= testset
test_loader = th.utils.data.DataLoader(
    test_db,
    batch_size=batch_size, shuffle=False,drop_last=True) 
ss = MaskTest(1,test_loader,model1)


# In[ ]:




