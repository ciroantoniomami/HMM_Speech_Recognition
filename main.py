import numpy as np
from hmmlearn.hmm import GMMHMM
import warnings
import os
from python_speech_features import mfcc, delta
import pickle
from pathlib import Path
import librosa
from scipy.io import wavfile

def extract_mfcc(full_audio_path, num_delta=5, add_mfcc_delta=True, add_mfcc_delta_delta=True):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave,sample_rate, 0.025, 
    0.01,numcep=20,nfft = 1200, appendEnergy = True)   
    print(mfcc_features.shape)
    wav_features = np.empty(shape=[mfcc_features.shape[0], 0])
    if add_mfcc_delta:
        delta_features = delta(mfcc_features, num_delta)
        wav_features = np.append(wav_features, delta_features, 1)
    if add_mfcc_delta_delta:
        delta_delta_features = librosa.feature.delta(mfcc_features, order=2)
        wav_features = np.append(wav_features, delta_delta_features, 1)
    wav_features = np.append(mfcc_features, wav_features, 1)
       
    
    return wav_features

def get_feature_list(path):
    fileList = list(Path(path).rglob('*.wav'))
    labels = [file.parent.stem for file in fileList]
    features = []
    for f in fileList:
        features.append(extract_mfcc(f))
    
    c = list(zip(features, labels))
    #np.random.shuffle(c)
    features, labels = zip(*c)

    print(f'nr of features {len(features)}')
    print(f'nr of labels {len(labels)}')

    return features, labels
    



class SpeechModel:
    def __init__(self, Class, label, transmatPrior, startprobPrior, m_n_iter=10, n_features_traindata=20):
        self.Class = Class
        self.label = label 
        self.model = GMMHMM(n_components=3, n_mix=7,
                                transmat_prior=transmatPrior, startprob_prior=startprobPrior,
                                covariance_type='diag', n_iter=m_n_iter)
        self.traindata = np.zero((0, n_features_traindata))


def train( features, labels, bakisLevel=2):
    words = set(labels)
    wordmodel = [None]*len(words)
    transmatPrior, startprobPrior = initByBakis(3,bakisLevel)
    for i in range(len(words)):
        wordmodel[i]= SpeechModel(i, words[i], transmatPrior, startprobPrior)

    for i in range(features):
        for j in range(len(wordmodel)):
            if wordmodel[j].label == labels[i]:
                wordmodel[j].traindata= np.concatenate(wordmodel[j].traindata,
                 features[i])

    for model in wordmodel:
        model.fit(model.traindata)
    
    n_spoken = len(words)
        print(f'Training completed -- {n_spoken} GMM-HMM models are built for {n_spoken} different types of words')



def getTransmatPrior(inumstates, bakisLevel):
        transmatPrior = (1 / float(bakisLevel)) * np.eye(inumstates)

        for i in range(inumstates - (bakisLevel - 1)):
            for j in range(bakisLevel - 1):
                transmatPrior[i, i + j + 1] = 1. / bakisLevel

        for i in range(inumstates - bakisLevel + 1, inumstates):
            for j in range(inumstates - i - j):
                transmatPrior[i, i + j] = 1. / (inumstates - i)

        return transmatPrior

def initByBakis(self, inumstates, ibakisLevel):
        startprobPrior = np.zeros(inumstates)
        startprobPrior[0: ibakisLevel - 1] = 1 / float((ibakisLevel - 1))
        transmatPrior = self.getTransmatPrior(inumstates, ibakisLevel)
        return startprobPrior, transmatPrior






def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    


if __name__ == "__main__":
    #features = get_feature_list("train/audio")
    #save_obj(features, "featureslist")
    #print(getTransmatPrior(3,2))

