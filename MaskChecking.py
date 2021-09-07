#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import librosa
import torch as th
import IPython.display as ipd
import numpy as np
import soundfile as sf
import pystoi  #stoi value computing from https://github.com/mpariente/pystoi/tree/b243f8e5e98a45cc731916c8d45415a845f90f3f/pystoi
from pesq import pesq  #pesq value computing from https://github.com/ludlows/python-pesq


# In[3]:


class Net(th.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden  = th.nn.Linear(n_feature, n_hidden)
        self.hidden2 = th.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = th.nn.Linear(n_hidden, n_hidden)
        self.output  = th.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        m = th.nn.ELU()
        si = th.nn.Sigmoid()
        dp = th.nn.Dropout(p=0.3)
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
    


# In[4]:


N_FRAME = 256                     # Window support
NFFT = 512            
nameto = 'p232_001.wav' #speech file to do enhancement
sp_wavx = os.path.join('DS_10283_2791', 'noisy_testset_wav', nameto)
sp_wavy = os.path.join('DS_10283_2791', 'clean_testset_wav', nameto)
Fs = 16000
model = th.load('beststoi20bsnew.pth')
x, Fs = librosa.load(sp_wavx, sr=Fs)#noisy
y, Fs = librosa.load(sp_wavy, sr=Fs)#clean
stoibe = pystoi.stoi(y, x, 16000, extended=False)

savename="/Users/schicksal/Desktop/intelligibility enhancement/stoi/"  
save1 = savename+'unprocessed'+ nameto
sf.write(save1, x, 16000, 'PCM_24')
x_spec = librosa.stft(x,n_fft=512, hop_length=256, win_length=512, window='hann')
y_spec = librosa.stft(y,n_fft=512, hop_length=256, win_length=512, window='hann')
pesqbe = pesq(16000, y, x, 'wb')
magnitude, phase = librosa.magphase(x_spec) #get the noisy phase information
ymagnitude, yphase = librosa.magphase(y_spec) #get the clean phase information
x_spec = np.abs(x_spec)
y_spec = np.abs(y_spec)
shape = x_spec.shape[1]
x_norm = 2*np.log(x_spec) #get the log power spectrum of noisy signal
x_norm = np.array(
    [x_norm[:, m-2 :m+3] for m in range(2,shape-2 )])
x_norm= np.reshape(x_norm,(shape-4,1285))
x_norm= (x_norm- np.mean(x_norm,axis=1, keepdims=True))/np.std(x_norm,axis=1,keepdims=True)   #normlized the input features to zero mean and unit variance
x_norm = th.Tensor(x_norm)
sh = y_spec.shape[1]
mask = model(x_norm) #generate masks through trained network
mask = th.reshape(mask,(sh-4,257))
mask = th.transpose(mask,1,0)
mask = mask.detach().numpy()
x_spec[:,2:sh-2] = x_spec[:,2:sh-2]*mask #apply the mask to the noisy magnitude spectrum 
xox = x_spec*phase    
yoy = y_spec*yphase    
xx =librosa.istft(xox,hop_length=256, win_length=512, window='hann')
yy =librosa.istft(yoy,hop_length=256, win_length=512, window='hann')
pesqaf = pesq(16000, yy, xx, 'wb')
stoiaf = pystoi.stoi(yy, xx, 16000, extended=False)
save2 = savename+'processed'+ nameto
sf.write(save2, xx, 16000, 'PCM_24')
stoiaf = pystoi.stoi(yy, xx, 16000, extended=False)
print(stoibe,stoiaf,pesqbe,pesqaf)
ipd.Audio(xx, rate=16000)


# In[ ]:




