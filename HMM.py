import warnings
import os
import python_speech_features as mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np

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



if __name__ == "__main__":

    data = buildDataSet("dataset/")
    print(data["mic_F01_sa2"].shape[1])