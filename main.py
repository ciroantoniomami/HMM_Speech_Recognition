import numpy as np
from sklearn.hmm import GMMHMM
import warnings
import os
import python_speech_features as mfcc, delta
import pickle
import librosa
from scipy.io import wavfile

def extract_mfcc(full_audio_path, num_delta=5, add_mfcc_delta=True, add_mfcc_delta_delta=True):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc.mfcc(wave,sample_rate, 0.025, 
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
    fileList = list(path.rglob('*.wav'))
    labels = [file.parent.stem for file in fileList]
    features = []
    for f in fileList:
        features.append(extract_mfcc(f))
    
    c = list(zip(features, labels))
    np.random.shuffle(c)
    features, labels = zip(*c)

    print(f'nr of features {len(features)}')
    print(f'nr of labels {len(labels)}')

    return features, labels
    



class SpeechModel:
    def __init__(self, class, label, m_transmatPrior, m_startprobPrior, m_n_iter=10, n_features_traindata=6):
        self.class = class
        self.label = label 
        self.model = GMMHMM(n_components=3, n_mix=7,
                                transmat_prior=m_transmatPrior, startprob_prior=m_startprobPrior,
                                covariance_type='diag', n_iter=m_n_iter)
        self.traindata = np.zero((0, n_features_traindata))




def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    


if __name__ == "__main__":
    features = get_feature_list("train/audio")

