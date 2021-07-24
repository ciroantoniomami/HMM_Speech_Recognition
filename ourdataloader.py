from sklearn.linear_model import LogisticRegressionCV
from hmmlearn.base import _BaseHMM, ConvergenceMonitor
from hmmlearn.utils import iter_from_X_lengths, normalize
from librosa import feature
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
import torch
from torch import nn
from torch.autograd import Variable
import librosa 
import os
from python_speech_features import mfcc, delta
from hmmlearn import hmm
import pickle
import time
from scipy.io import wavfile
update_dnn = True
fast_update = False
forward_backward = False
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix










def read_wav(fpath):
    sample_rate, signal = wavfile.read(fpath)
    return sample_rate, signal

def get_features(signal, sample_rate, num_delta=5, add_mfcc_delta = True, add_mfcc_delta_delta = True):
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=13)
    #wav_features = np.empty(shape=[mfcc_features.shape[0], 0])
    #if add_mfcc_delta:
    #    delta_features = delta(mfcc_features, num_delta)
    #    wav_features = np.append(wav_features, delta_features, 1)
    #if add_mfcc_delta_delta:
    #    delta_delta_features = librosa.feature.delta(mfcc_features, order=2)
    #    wav_features = np.append(wav_features, delta_delta_features, 1)
    #wav_features = np.append(mfcc_features, wav_features, 1)
    return mfcc_features

def read_wav_get_features(eval=False, num_delta=5):

    fpaths = [f for f in os.listdir('train/audio2/') if os.path.splitext(f)[1] == '.wav']
    labels = [file.split("_")[-1][1:-4] for file in fpaths]
    spoken = list(set(labels))
    spoken.sort()
    len_per_label = {key:0 for key in spoken}

    
    features = []
    for n, file in enumerate(fpaths):
        
        sample_rate, signal = read_wav('train/audio2/' + file)
        
        
        file_features = get_features(signal, sample_rate, num_delta)
        len_per_label[labels[n]] +=1
        features.append(file_features)
    if eval:
        return features
    
    c = list(zip(features, labels))
    #np.random.shuffle(c)
    features, labels = zip(*c)
     
        
    print(f'nr of features {len(features)}')
    print(f'nr of labels {len(labels)}')
    return features, labels, spoken, len_per_label


if __name__ == "__main__":


    features, labels, spoken, len_per_label = read_wav_get_features()
    
    print(len_per_label)
    spoken = ["down","go","left","no","off","on","right","stop","up","yes"]
    print(spoken)
    for word in spoken:
        name = word + "train.txt"
        file = open(name,"a")

        val_i_end = int(0.2*len(features))
        count = 0
        for i in range(val_i_end, len(features[val_i_end:])):
            
            if labels[i] == word:
                count +=1
                if count == 301:
                    break
                for line in features[i]:
                    for j in range(len(line)):
                        file.write(str(line[j])+ " ")
                    file.write('\n')
                file.write('\n')
    
    for word in spoken:
        name = word + "test.txt"
        file = open(name,"a")

        val_i_end = int(0.2*len(features))
        count = 0
        for i in range(0,val_i_end):
            
            if labels[i] == word:
                count +=1
                if count == 51:
                    break
                for line in features[i]:
                    for j in range(len(line)):
                        file.write(str(line[j])+ " ")
                    file.write('\n')
                file.write('\n')

        

    file2 = open("train.txt","a")
    for word in spoken:
        name = word + "train.txt"
        file = open(name,"r")
        file2.write(file.read())
    
    file3 = open("test.txt","a")
    for word in spoken:
        name = word + "test.txt"
        file = open(name,"r")
        file3.write(file.read())


