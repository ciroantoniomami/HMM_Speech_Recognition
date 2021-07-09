from __future__ import print_function
import warnings
import os
from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
from pathlib import Path
import pickle
from operator import itemgetter
import collections
warnings.filterwarnings('ignore')
def extract_mfcc(full_audio_path):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave,sample_rate, 0.025, 
    0.01,numcep=12,nfft = 1200, appendEnergy = True) 
    return mfcc_features

def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    fileList = list(Path(dir).rglob('*.wav'))
    dataset = {}
    for fileName in fileList:
        #tmp = fileName.split('.')[0]
        label = fileName.parent.stem 
        feature = extract_mfcc(fileName)
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
    return dataset

def train_GMMHMM(dataset):
    GMMHMM_Models = {}
    states_num = 5
    GMM_mix_num = 3
    tmp_p = 1.0/(states_num-2)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                               [0, tmp_p, tmp_p, tmp_p , 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0.5, 0.5], \
                               [0, 0, 0, 0, 1]],dtype=np.float)


    startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

    for label in dataset.keys():
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m].shape[0]
        trainData = np.vstack(trainData)
        print("1")
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)



def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def predict(wordmodel, files):
        features = []
        features.append(extract_mfcc(files))
        Model_confidence = collections.namedtuple('model_prediction', ('name', 'score'))
        predicted_labels_confs = []

        for i in range(0, len(features)):
            file_scores_confs = []
            for key,value in wordmodel.items():
                score = value.score(features[i])
                label = key
                file_scores_confs.append(Model_confidence(name=label, score=score))
            file_scores_confs = sorted(file_scores_confs, key=itemgetter(1), reverse=True)
            predicted_labels_confs.append(file_scores_confs[0])

        return predicted_labels_confs

if __name__ == "__main__":
    #trainDir = "train/audio"
    #trainDataSet = buildDataSet(trainDir)
    #print("Finish prepare the training data")
#
    #hmmModels = train_GMMHMM(trainDataSet)
    #save_obj(hmmModels, "modelslist")
    #print("Finish training of the GMM_HMM models for digits 0-9")

    Models = load_obj("modelslist")
    print(predict(Models,"yes.wav"))



  