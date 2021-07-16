from model import read_wav_get_features, save_list, load_list, MLP, hmm_dnn
from HMM2 import load_obj, save_obj
from librosa import feature
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import librosa 
from hmmlearn import hmm
import pickle
import time
update_dnn = True
fast_update = False
forward_backward = False
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool


if __name__ == "__main__":

    tmp_p = 1.0/3
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                           [0, tmp_p, tmp_p, tmp_p , 0], \
                           [0, 0, tmp_p, tmp_p,tmp_p], \
                           [0, 0, 0, 0.5, 0.5], \
                           [0, 0, 0, 0, 1]],dtype=np.float64)

    startprobPrior = np.array([1, 0, 0, 0, 0],dtype=np.float64)

    features, labels, spoken = read_wav_get_features()
    save_obj(features,'featurelist')
    save_obj(labels,'labelslist')
    for w in spoken:
        print(w)
    traindata = [None]*len(spoken)
    for i in range(len(traindata)):
        traindata[i] = np.zeros((0,36))
    val_i_end = int(0.2*len(features))
    for i in range(val_i_end, len(features[val_i_end:])):
            for j in range(0, len(spoken)):
                if spoken[j] == labels[i]:
                    traindata[j] = np.concatenate((traindata[j], features[i]))
    

    #traindata = load_obj('featurelist')

    gmmhmm_module_list = []
    seq_mapper = []

    for i in range(len(spoken)):
            gmmhmm_module_list.append(hmm.GMMHMM(n_components=5, n_mix=18,
                                                 covariance_type='diag', transmat_prior=transmatPrior, startprob_prior=startprobPrior, n_iter=10))

    for i, module in enumerate(gmmhmm_module_list):
        module.fit(traindata[i])

    
    for i, module in enumerate(gmmhmm_module_list):
        #for data in traindata[i]:

        prob, path = module.decode(traindata[i])
        seq_mapper.append((i, traindata[i], path))

        save_list(seq_mapper, 'path.dict')

    

    
    phonetic_train_data = [np.zeros((0, 36))] * len(spoken)
    phonetic_train_label = [np.zeros((0, 1))] * len(spoken)

    

    for label, data, seq in seq_mapper:
        phonetic_train_data[label] = np.vstack([phonetic_train_data[label], np.array(data)])
        phonetic_train_label[label] = np.vstack([phonetic_train_label[label], np.array(seq).reshape(-1, 1)])

        

    # train DNN network
    #############################################################################
    print('---- training DNN network')
    dnn_module_list = []
    for i in range(len(spoken)):
        dnn_module_list.append(MLP(36, 5))

    for i, module in enumerate(dnn_module_list):
        module.train(phonetic_train_data[i], phonetic_train_label[i])
    
    print('number of DNN:',len(dnn_module_list))

    # create hmm dnn modules
    #############################################################################
    hmm_dnn_module_list = []
    for i in range(len(spoken)):
        hmm_dnn_module_list.append(
            hmm_dnn(dnn_module_list[i],
                    n_components = 5,
                    startprob_prior=gmmhmm_module_list[i].startprob_, transmat_prior=gmmhmm_module_list[i].transmat_,
                    n_iter=10))

    print('number of HMM-DNN:',len(hmm_dnn_module_list))
    # train hmm dnn modules
    #############################################################################
    print('---- training HMM-DNN')
    start_train_time = time.time()
    for i, module in enumerate(hmm_dnn_module_list):
        print('HMM_DNN number:',i)
        module.fit(traindata[i])
    print("Train Time: ", time.time() - start_train_time)

    save_obj(hmm_dnn_module_list,'hmm_dnn_list')

