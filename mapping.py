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
from torch.autograd import Variable
from matplotlib import pyplot as plt


# In[30]:


class Trainset(Dataset):
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
        N=5
        x, Fs = librosa.load(fn, sr=Fs)
        y, Fs = librosa.load(fc, sr=Fs)
        x_spec = librosa.stft(x,n_fft=512, hop_length=256, win_length=512, window='hann')
        y_spec = librosa.stft(y,n_fft=512, hop_length=256, win_length=512, window='hann')
        x_spec = np.abs(x_spec)
        y_spec = np.abs(y_spec)
        rng = np.random.default_rng()
        shape = x_spec.shape[1]-3
        ran= rng.integers(2,shape)
        x_norm = np.log(x_spec**2)
        y_norm = np.log(y_spec**2)
        y_norm = np.array(
            y_norm[:, ran])
        x_norm = np.array(
            x_norm[:, ran-2 :ran+3] )
        x_norm= np.reshape(x_norm,(1285))
        x_norm= (x_norm- np.mean(x_norm))/np.std(x_norm)
        x_norm = th.Tensor(x_norm)
        y_norm = th.Tensor(y_norm)    

        return x_norm,y_norm
    def __len__(self):
        return len(self.audioc)


# In[31]:




trainc = "/Users/schicksal/Desktop/intelligibility enhancement/16k_train_clean_audio.txt"
trainn = "/Users/schicksal/Desktop/intelligibility enhancement/16k_train_noisy_audio.txt"
trainset = Trainset( train_clean_audio_path = trainc, train_noisy_audio_path = trainn)
valc = "/Users/schicksal/Desktop/intelligibility enhancement/16k_val_clean_audio.txt"
valn = "/Users/schicksal/Desktop/intelligibility enhancement/16k_val_noisy_audio.txt"
valset = Trainset( train_clean_audio_path = valc, train_noisy_audio_path = valn)


# In[36]:


# structure of the neural network for mapping: three hidden layers with 1024 exponential linea units(ELUs) units
# each layer with a dropout layer (for dropout regularization) with dropout rate of 0.3,
# for the output layer, linear activation units(ELUs)
# the input of the network is 5-frame continuous log power spectrums of noisy speech signal
# output of the network is the predicted 1-frame predicted log power spectrum
class MappingNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MappingNet, self).__init__()
        self.hidden  = nn.Linear(n_feature, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.output  = nn.Linear(n_hidden,n_output)        
    def forward(self,x):
        m = nn.ELU()
        dp = nn.Dropout(p=0.3)
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
        return x
    


# In[33]:


def Mse_train(epochs, trainLoader,valLoader, model, Lr, device = "cpu"): # training procedure for MSEloss
    mse = nn.MSELoss() #loss function of MSEloss
    optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
    losst = 0 #trining loss of every epoch
    lossvb =1000
    lossti = np.zeros(epochs)
    lossvi = np.zeros(epochs)
    scheduler =th.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.9)
    
    for e in range(epochs):
        losst=0
        for i, (x_norm,y_norm) in enumerate(train_loader):
            mapping = model(x_norm) #get a mapping function based on each log-magnitude spectrogram 
            loss =mse(mapping,y_norm) #calculate loss based on the estimated log power spectrum and clean log power spectrum
            print(i,loss)
            losst = losst+loss
            optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
            loss.backward()
            optimizer.step()
        losst = losst/(i+1)
        lossti[e] = losst
        print('epoch: {}, Trainloss: {}'.format(e + 1, losst))
        lossv = 0
        for i,(x_norm,y_norm) in enumerate(val_loader):
            mapping = model(x_norm) #get a mapping function based on each log-magnitude spectrogram 
            lossv =lossv +mse(mapping,y_norm)   #calculate loss based on the estimated log power spectrum and clean log power spectrum
        lossv = lossv/(i+1)
        lossvi[e] = lossv   
        if lossvb >lossv:
            lossvb = lossv
            torch.save(model, 'mappingbest.pth') # save net model and parameters of the model with best validation performance
        elif e>=15:
            torch.save(model, 'mappinglast.pth') # save net model and parameters, early stopping patience 15
            break
        print('epoch: {}, Valloss: {}, Trainloss: {}'.format(e + 1, lossv, losst))
        scheduler.step()
    torch.save(model, 'mappinglast.pth') # save the last net model and parameters
    return lossti,lossvi


# In[37]:


#model1 = th.load('mappingbest1.pth')
model1 = MappingNet(1285,1024,257)
batch_size= 20
train_loader = th.utils.data.DataLoader(
    trainset,
    batch_size=batch_size, shuffle=True,drop_last=True) 
val_loader = th.utils.data.DataLoader(
    valset,
    batch_size=batch_size, shuffle=True,drop_last=True)
lossti,lossvi = Mse_train(50,train_loader,val_loader,model1,0.0001)


# In[ ]:




