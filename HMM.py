import warnings
import os
import python_speech_features as mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
import pickle


warnings.filterwarnings('ignore')
def extract_mfcc(full_audio_path):
    sample_rate, wave =  wavfile.read(full_audio_path)
    mfcc_features = mfcc.mfcc(wave,sample_rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)   
    return mfcc_features

def buildDataSet(dir):
    # Filter out the wav audio files under the dir
    fileList = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    for fileName in fileList:
        label = fileName.split('.')[0]
        #label = tmp.split('_')[1]
        feature = extract_mfcc(dir+fileName)
        if label not in dataset.keys():
            dataset[label] = feature

        
    return dataset

def training(dataset):
    np.random.seed(42)
   
   
    startprob = np.random.rand(5)
    transmat = np.random.rand(5,5)
    

    for i in range(5):
        for j in range(i):
            transmat[i][j]=0
    
    model = hmm.GMMHMM(n_components=5, n_mix=3, 
                           transmat_prior= transmat, startprob_prior=startprob, 
                           covariance_type='diag', n_iter=10)
    
    
    lengths = []
    
    first = True
    for label in dataset.keys():
        
        if first == True: 
            traindata = dataset[label]
            first = False
        else:
            
            
            traindata = np.concatenate([traindata, dataset[label]])
           
        lengths.append(dataset[label].shape[0])

    model.fit(traindata, lengths )


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":

    #d = buildDataSet("dataset/")
    #save_obj(d, "dataset")
    d = load_obj("dataset")

    



    training(d)