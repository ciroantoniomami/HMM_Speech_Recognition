from __future__ import print_function
from pyexpat import features
import warnings
import os
from python_speech_features import mfcc, delta
import librosa
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
from pathlib import Path
import pickle
from operator import itemgetter
import collections
warnings.filterwarnings('ignore')
def extract_mfcc(full_audio_path,add_mfcc_delta=True,add_mfcc_delta_delta=True,num_delta=5):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc(wave,sample_rate, 0.025, 
    0.01,numcep=14,nfft = 1200, appendEnergy = True)
    wav_features = np.empty(shape=[mfcc_features.shape[0], 0])
    if add_mfcc_delta:
        delta_features = delta(mfcc_features, num_delta)
        wav_features = np.append(wav_features, delta_features, 1) 
    if add_mfcc_delta_delta:
        delta_delta_features = librosa.feature.delta(mfcc_features, order=2)
        wav_features = np.append(wav_features, delta_delta_features, 1)
    wav_features = np.append(mfcc_features, wav_features, 1)
    
    return wav_features



def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        #tmp = fileName.split('.')[0]
        label = fileName.split("_")[-2]
        feature = extract_mfcc(dir+fileName)
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
    GMM_mix_num = 17
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
    trainDir = "Crema/trainemotion/"
    trainDataSet = buildDataSet(trainDir)
    save_obj(trainDataSet,"trainsetemotion")
    trainDataSet = load_obj("trainsetemotion")
    hmmModels = train_GMMHMM(trainDataSet)
    save_obj(hmmModels, "modelslist")
    print("Finish training of the GMM_HMM models for digits 0-9")
    hmmModels = load_obj("modelslist")
    testDir = "Crema/testemotion/"
    testDataSet = buildDataSet(testDir)
    score_cnt = 0
    tot = 0
    for label in testDataSet.keys():
        label_cont = 0
        feature = testDataSet[label]
        scoreList = {}
        label_sc = 0
        for f in feature:
            label_cont+=1
            tot += 1
            for model_label in hmmModels.keys():
                model = hmmModels[model_label]
                score = model.score(f)
                scoreList[model_label] = score
            predict = max(scoreList, key=scoreList.get)
            #print("Test on true label ", label, ": predict result label is ", predict)
            if predict == label:
                label_sc +=1
                score_cnt+=1
        print("score rate for",label,"is:%.2f"%(100.0*label_sc/label_cont), "%")
    print("Final recognition rate is %.2f"%(100.0*score_cnt/tot), "%")


  