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
import stoiloss_16k as stsp  # import stoi loss function
import infoloss_16k as my       # import info theory loss function


# In[2]:


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
        # Do stft to the both clean and noisy speech signals 
        # randomly take a continuous 24(stoi loss and Mse loss) or 48(info loss) time frames segment
        # 
        Fs = 16000                          #sampling frequency fixed to 16000
        fc = self.audioc[index]
        fn = self.audion[index]
        N_FRAME = 512                       # Window size of stft support
        NFFT = 512                          # FFT Size
        N=5                                 # using 5 time frames context window to do prediction
        Frames = 48                         # time frames for different intelligibility metrics calculation(stoi and mse 24, info_theory 48)
        x, Fs = librosa.load(fn, sr=Fs)
        y, Fs = librosa.load(fc, sr=Fs)
        x_spec = librosa.stft(x,n_fft=512, hop_length=256, win_length=512, window='hann')#stft to the noisy speech
        y_spec = librosa.stft(y,n_fft=512, hop_length=256, win_length=512, window='hann')#stft to the clean speech
        #take the magnitude of both clean and noisy speech spectrum
        x_spec = np.abs(x_spec)
        y_spec = np.abs(y_spec)
        rng = np.random.default_rng() # define a random number generator
        shape = x_spec.shape[1]-(Frames+3)    #set the upper bound of the random number
        ran= rng.integers(2,shape) #generate a random number which represents the index of randomly chosen time frame
        x_norm = np.log(x_spec**2) #comput the log power magnitude spectrum of noisy magnitude specturm
        #segment the continuous 24(stoi and mse) or 48(info) time frames magnitude spectrum of noisy and clean signal
        x_spec = np.array(
            x_spec[:, ran :ran+Frames] )
        y_spec = np.array(
            y_spec[:, ran :ran+Frames])
        #segment the corresponding log power spectrum of noisy signal 
        x_norm = np.array(
            [x_norm[:, m-2 :m+3] for m in range(ran,ran+Frames )])
        x_norm= np.reshape(x_norm,(Frames,1285))
        #normlized the input features to zero mean and unit variance
        x_norm= (x_norm- np.mean(x_norm,axis=1, keepdims=True))/np.std(x_norm,axis=1,keepdims=True)
        x_norm= np.reshape(x_norm,(-1))
        x_spec = th.Tensor(x_spec)
        y_spec = th.Tensor(y_spec)    
        x_norm = th.Tensor(x_norm)
        return x_spec,y_spec,x_norm
    def __len__(self):
        return len(self.audioc)


# In[3]:


#dataset from https://datashare.ed.ac.uk/handle/10283/2791, and take one feamle and one male speaker's speech as validation set
#the sampling rate of the original signals are 48k, before training, we resample the to 16k to decrease the training time
trainc = "/Users/schicksal/Desktop/intelligibility enhancement/16k_train_clean_audio.txt"
trainn = "/Users/schicksal/Desktop/intelligibility enhancement/16k_train_noisy_audio.txt"
trainset = Trainset( train_clean_audio_path = trainc, train_noisy_audio_path = trainn)
valc = "/Users/schicksal/Desktop/intelligibility enhancement/16k_val_clean_audio.txt"
valn = "/Users/schicksal/Desktop/intelligibility enhancement/16k_val_noisy_audio.txt"
valset = Trainset( train_clean_audio_path = valc, train_noisy_audio_path = valn)


# In[4]:


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
    
    


# In[5]:


def My_train(epochs, trainLoader,valLoader, model, Lr, device = "cpu"):# training procedure for information theory loss metric
    cccss=my.myLoss()#loss function of intelligibility metric based on information theory
    optimizer = torch.optim.Adam(model.parameters(), lr=Lr) #Adam optimizer was used here
    lossvb =1000 # initilaize the best validation loss
    lossti = np.zeros(epochs) # initialize a numpy array to store the training loss in every epoch
    lossvi = np.zeros(epochs) # initialize a numpy array to store the validation loss in every epoch
    scheduler =th.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.9) #learning rate decay of patience 5, and decays to 0.9 each time    
    for e in range(epochs):
        losst=0 # initialize the trining loss of every epoch
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        for i, (x,y,z) in enumerate(train_loader):
            z=th.reshape(z,(960,1285))
            mask = model(z) #utilize network to get a mask based on each log-magnitude spectrogram 
            mask = th.reshape(mask,(20,48,257))
            mask = th.transpose(mask,2,1)
            x = x*mask #apply mask on the magnitude spectrogram
            loss =cccss(x,y) #calculate loss
            print(i,loss)
            losst = losst+loss
            optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
            loss.backward()
            optimizer.step()
        losst = losst/(i+1) #the average training loss
        lossti[e] = losst
        print('epoch: {}, Trainloss: {}'.format(e + 1, losst))
        lossv = 0
        for i,(x,y,z) in enumerate(val_loader):
            z=th.reshape(z,(960,1285))
            mask = model(z) #get a mask based on each log-magnitude spectrogram 
            mask = th.reshape(mask,(20,48,257))
            mask = th.transpose(mask,2,1)
            x = x*mask #apply mask on the magnitude spectrogram
            lossv =lossv +cccss(x,y)
        lossv = lossv/(i+1) #the average validation loss
        lossvi[e] = lossv   
        if lossvb >lossv:
            lossvb = lossv
            torch.save(model, 'bestinfo.pth') # save net model and parameters of the model with best validation performance
        elif e>=10:
            torch.save(model, 'lastinfo.pth') # save net model and parameters, early stopping patience 10
            break
        print('epoch: {}, Valloss: {}, Trainloss: {}'.format(e + 1, lossv, losst))
        scheduler.step()

    torch.save(model, 'lastinfo.pth') # save net model and parameters
    return lossti,lossvi    #return the average training loss and validation loss


# In[6]:


def Stoi_train(epochs, trainLoader,valLoader, model, Lr, device = "cpu"): # training procedure for stoi loss metric
    ccrr =stsp.stoiLoss() #loss function of intelligibility metric based on stoi
    optimizer = torch.optim.Adam(model.parameters(), lr=Lr) #Adam optimizer was used here
    lossvb =1000 # initilaize the best validation loss
    lossti = np.zeros(epochs) # initialize a numpy array to store the training loss in every epoch
    lossvi = np.zeros(epochs) # initialize a numpy array to store the validation loss in every epoch
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    scheduler =th.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.9)
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    
    for e in range(epochs):
        losst=0 # initialize the trining loss of every epoch
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        for i, (x,y,z) in enumerate(train_loader):
            z=th.reshape(z,(480,1285))
            mask = model(z) #utilize network to get a mask based on each log-magnitude spectrogram 
            mask = th.reshape(mask,(20,24,257))
            mask = th.transpose(mask,2,1)
            x = x*mask #apply mask on the magnitude spectrogram
            loss =ccrr(x,y) #calculate loss
            print(i,loss)
            losst = losst+loss
            optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
            loss.backward()
            optimizer.step()
        losst = losst/(i+1) #the average training loss
        lossti[e] = losst
        print('epoch: {}, Trainloss: {}'.format(e + 1, losst))
        lossv = 0
        for i,(x,y,z) in enumerate(val_loader):
            z=th.reshape(z,(480,1285))
            mask = model(z) #get a mask based on each log-magnitude spectrogram 
            mask = th.reshape(mask,(20,24,257))
            mask = th.transpose(mask,2,1)
            x = x*mask #apply mask on the magnitude spectrogram
            lossv =lossv +ccrr(x,y)

        lossv = lossv/(i+1)#the average validation loss
        lossvi[e] = lossv   
        if lossvb >lossv:
            lossvb = lossv
            torch.save(model, 'beststoi.pth') # save net model and parameters of the model with best validation performance
        elif e>=10:
            torch.save(model, 'laststoi.pth') # save net model and parameters, early stopping patience 10
            break
        print('epoch: {}, Valloss: {}, Trainloss: {}'.format(e + 1, lossv, losst))
        scheduler.step()

    torch.save(model, 'laststoi.pth') # save net model and parameters
    return lossti,lossvi    #return the average training loss and validation loss


# In[7]:


def Mse_train(epochs, trainLoader,valLoader, model, Lr, device = "cpu"): # training procedure for MSEloss
    mse = nn.MSELoss() #loss function of MSEloss
    optimizer = torch.optim.Adam(model.parameters(), lr=Lr) #Adam optimizer was used here
    lossvb =1000 # initilaize the best validation loss
    lossti = np.zeros(epochs) # initialize a numpy array to store the average training loss in every epoch
    lossvi = np.zeros(epochs) # initialize a numpy array to store the average validation loss in every epoch
    scheduler =th.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.9)    
    for e in range(epochs):
        losst=0 # initialize the trining loss of every epoch
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        for i, (x,y,z) in enumerate(train_loader):
            z=th.reshape(z,(480,1285))
            mask = model(z) #utilize network to get a mask based on each log-magnitude spectrogram 
            mask = th.reshape(mask,(20,24,257))
            mask = th.transpose(mask,2,1)
            x = x*mask #apply mask on the magnitude spectrogram
            loss =mse(x,y) #calculate loss
            print(i,loss)
            losst = losst+loss
            optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
            loss.backward()
            optimizer.step()
        losst = losst/(i+1)#the average training loss
        lossti[e] = losst
        print('epoch: {}, Trainloss: {}'.format(e + 1, losst))
        lossv = 0
        for i,(x,y,z,y_norm) in enumerate(val_loader):
            z=th.reshape(z,(480,1285))
            mask = model(z) #get a mask based on each log-magnitude spectrogram 
            mask = th.reshape(mask,(20,24,257))
            mask = th.transpose(mask,2,1)
            x = x*mask #apply mask on the magnitude spectrogram
            lossv =lossv +mse(x,y)
        lossv = lossv/(i+1)#the average validation loss
        lossvi[e] = lossv   
        if lossvb >lossv:
            lossvb = lossv
            torch.save(model, 'bestmse.pth') # save net model and parameters of the model with best validation performance
        elif e>=10:
            torch.save(model, 'lastmse.pth') # save net model and parameters, early stopping patience 10
            break
        print('epoch: {}, Valloss: {}, Trainloss: {}'.format(e + 1, lossv, losst))
        scheduler.step()
    torch.save(model, 'lastmse.pth') # save net model and parameters
    return lossti,lossvi   #return the average training loss and validation loss


# In[8]:


model1 = Net(1285,1024,257)   #initialize a new network model with 1285 input units, 1024 hidden units, and 257 output units
batch_size= 20 #flexible setting the batch size, we use 10 or 20 here
train_loader = th.utils.data.DataLoader(
    trainset,
    batch_size=batch_size, shuffle=True,drop_last=True) #generate a training dataset loader
val_loader = th.utils.data.DataLoader(
    valset,
    batch_size=batch_size, shuffle=True,drop_last=True) #generate a validation dataset loader
lossti,lossvi = Mse_train(50,train_loader,val_loader,model1,0.0001)  # do the MSE loss training with 0.0001 initial learning rate 
lossti,lossvi = Stoi_train(50,train_loader,val_loader,model1,0.0001) # do the Stoi loss training with 0.0001 initial learning rate
lossti,lossvi = My_train(50,train_loader,val_loader,model1,0.0001)   # do the info theory loss training with 0.0001 initial learning rate


# In[ ]:





# In[ ]:




