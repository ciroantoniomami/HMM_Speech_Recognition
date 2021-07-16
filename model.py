from HMM2 import load_obj, save_obj
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
import pandas as pd
from multiprocessing import Pool





def read_wav(fpath):
    sample_rate, signal = wavfile.read(fpath)
    return sample_rate, signal

def get_features(signal, sample_rate, num_delta=5, add_mfcc_delta = True, add_mfcc_delta_delta = True):
    mfcc_features = mfcc(signal, samplerate=sample_rate, numcep=12)
    wav_features = np.empty(shape=[mfcc_features.shape[0], 0])
    if add_mfcc_delta:
        delta_features = delta(mfcc_features, num_delta)
        wav_features = np.append(wav_features, delta_features, 1)
    if add_mfcc_delta_delta:
        delta_delta_features = librosa.feature.delta(mfcc_features, order=2)
        wav_features = np.append(wav_features, delta_delta_features, 1)
    wav_features = np.append(mfcc_features, wav_features, 1)
    return wav_features

def read_wav_get_features(eval=False, num_delta=5):

    fpaths = [f for f in os.listdir('train/audio3/') if os.path.splitext(f)[1] == '.wav']
    labels = [file.split("_")[-1][1:-4] for file in fpaths]
    spoken = list(set(labels))
    
    
    features = []
    for n, file in enumerate(fpaths):
        
        sample_rate, signal = read_wav('train/audio3/' + file)
        
        
        file_features = get_features(signal, sample_rate, num_delta)
        features.append(file_features)
    if eval:
        return features
    
    c = list(zip(features, labels))
    np.random.shuffle(c)
    features, labels = zip(*c)
     
        
    print(f'nr of features {len(features)}')
    print(f'nr of labels {len(labels)}')
    return features, labels, spoken



class hmm_dnn(_BaseHMM):

    def __init__(self, mlp,  n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=True,
                 params="stmc", init_params="stmc"):

        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.mlp = mlp
        self.mlp.info()

    def _compute_log_likelihood(self, X):
        prob = self.mlp.log_probablity(X).astype(type(X[0, 0]))


        return prob

    def _accumulate_sufficient_statistics(self, stats, X, epsilon, gamma, path, bwdlattice):

        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += gamma[0]
        if 't' in self.params:
            n_samples = X.shape[0]

            if n_samples <= 1:
                return

            a = np.zeros((self.n_components, self.n_components))

            for i in range(self.n_components):
                for j in range(self.n_components):
                    a[i, j] = np.sum(epsilon[:, i, j]) / (np.sum(gamma[:, i]) + (np.sum(gamma[:, i]) == 0))

            stats['trans'] += a

    def fit(self, X, lengths=None):

        X = check_array(X)
        self._init(X, lengths=lengths)
        self._check()

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        for iter in range(self.n_iter):
            print('iteration: {}'.format(iter))
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            tt = 0
            path_list = list()

            for i, j in iter_from_X_lengths(X, lengths):
                logprob, state_sequence = self.decode(X[i:j], algorithm="viterbi")

                curr_logprob += logprob

                epsilon = np.zeros((state_sequence.shape[0] - 1, self.n_components, self.n_components))
                gamma = np.zeros((state_sequence.shape[0], self.n_components))

                for t in range(state_sequence.shape[0] - 1):
                    epsilon[t, state_sequence[t], state_sequence[t + 1]] = 1

                for t in range(state_sequence.shape[0]):
                    for i in range(self.n_components):
                        if t != (state_sequence.shape[0] - 1):
                            gamma[t, i] = np.sum(epsilon[t, i])
                        else:
                            gamma[t, i] = gamma[t-1, i]

                path_list.append(state_sequence)
                self._accumulate_sufficient_statistics(stats, X[i:j], epsilon, gamma, state_sequence, None)
                tt += 1

            print('average loss: {}'.format(curr_logprob / tt))

            if not fast_update:
                stats['start'] /= tt
                stats['trans'] /= tt

                self._do_mstep1(stats)
                if update_dnn:
                    temp_path = np.zeros((0, 1))
                    for k, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
                        temp_path = np.vstack([temp_path, np.array(path_list[k]).reshape(-1, 1)])
                    self.mlp.train(X, temp_path, 20)

        

            self.monitor_.report(curr_logprob)
            if self.monitor_.iter == self.monitor_.n_iter or \
                    (len(self.monitor_.history) == 2 and
                     abs(self.monitor_.history[1] - self.monitor_.history[0]) < self.monitor_.tol * abs(
                                self.monitor_.history[1])):
                break

        print('----------------------------------------------')
        return self

    def _do_mstep1(self, stats):
        if 's' in self.params:
            startprob_ = stats['start']
            self.startprob_ = np.where(self.startprob_ == 0.0,
                                       self.startprob_, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = stats['trans']
            self.transmat_ = np.where(self.transmat_ == 0.0,
                                      self.transmat_, transmat_)
            normalize(self.transmat_, axis=1)

            for i, row in enumerate(self.transmat_):
                if not np.any(row):
                    self.transmat_[i][i] = 1


def fit1(model, X, lengths=None):

        X = check_array(X)
        model._init(X, lengths=lengths)
        model._check()
        model.monitor_ = ConvergenceMonitor(model.tol, model.n_iter, model.verbose)
        for iter in range(model.n_iter):
            print('iteration: {}'.format(iter))
            stats = model._initialize_sufficient_statistics()
            curr_logprob = 0
            tt = 0
            path_list = list()

            for i, j in iter_from_X_lengths(X, lengths):
                logprob, state_sequence = model.decode(X[i:j], algorithm="viterbi")

                curr_logprob += logprob

                epsilon = np.zeros((state_sequence.shape[0] - 1, model.n_components, model.n_components))
                gamma = np.zeros((state_sequence.shape[0], model.n_components))

                for t in range(state_sequence.shape[0] - 1):
                    epsilon[t, state_sequence[t], state_sequence[t + 1]] = 1

                for t in range(state_sequence.shape[0]):
                    for i in range(model.n_components):
                        if t != (state_sequence.shape[0] - 1):
                            gamma[t, i] = np.sum(epsilon[t, i])
                        else:
                            gamma[t, i] = gamma[t-1, i]

                path_list.append(state_sequence)
                model._accumulate_sufficient_statistics(stats, X[i:j], epsilon, gamma, state_sequence, None)
                tt += 1

            print('average loss: {}'.format(curr_logprob / tt))

            if not fast_update:
                stats['start'] /= tt
                stats['trans'] /= tt

                model._do_mstep(stats)
                if update_dnn:
                    temp_path = np.zeros((0, 1))
                    for k, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
                        temp_path = np.vstack([temp_path, np.array(path_list[k]).reshape(-1, 1)])
                    model.mlp.train(X, temp_path, 20)

        

            model.monitor_.report(curr_logprob)
            if model.monitor_.iter == model.monitor_.n_iter or \
                    (len(model.monitor_.history) == 2 and
                     abs(model.monitor_.history[1] - model.monitor_.history[0]) < model.monitor_.tol * abs(
                                model.monitor_.history[1])):
                break

        print('----------------------------------------------')
        return model


class NeuralNetwork(nn.Module):

    def __init__(self, feature_size, class_count):
        super(NeuralNetwork, self).__init__()

        mid1_neuron = 70
        mid2_neuron = 100
        mid3_neuron = 70

        self.layer1 = nn.Sequential(
            nn.Linear(feature_size, mid1_neuron)
        )

        self.layer1_post = nn.Sequential(
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(mid1_neuron, mid2_neuron)
        )

        self.layer2_post = nn.Sequential(
            nn.LeakyReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Linear(mid2_neuron, mid3_neuron)
        )

        self.layer3_post = nn.Sequential(
            nn.LeakyReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(mid3_neuron, class_count)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer1_post(out)
        out = self.layer2(out)
        out = self.layer2_post(out)
        out = self.layer3(out)
        out = self.layer3_post(out)
        out = self.layer4(out)
        return out


class MLP:

    def info(self):
        print('trained: ', self.trained)

    def __init__(self, feature_size, class_count):
        self.net = NeuralNetwork(feature_size, class_count)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.epoch_count = 100
        self.trained = False
         

    def train(self, data, label, epoch=None):
        self.trained = True
        self.net.train()
        number_of_epuchs = self.epoch_count if epoch is None else epoch
        for epoch in range(number_of_epuchs):
            accuracy_meter = AverageMeter()

            for i, (batch_data, batch_label) in enumerate(zip(data, label)):
                #print(batch_data.shape)
                batch_data = batch_data.reshape(1, -1)
                #print(batch_data.shape)

                batch_data = Variable(torch.from_numpy(batch_data)).float()
                batch_label = Variable(torch.from_numpy(batch_label)).long()

                self.optimizer.zero_grad()
                score = self.net(batch_data)

                loss = self.loss_function(score, batch_label)

                loss.backward()
                self.optimizer.step()

                acc = accuracy(score, batch_label)
        
                accuracy_meter.update(val=acc, n=batch_data.shape[0])
            print(f"Epoch {epoch+1} completed. Accuracy: {accuracy_meter.avg}")

    def log_probablity(self, data):
        self.net.eval()

        data = Variable(torch.from_numpy(data)).float()

        scores = self.net(data)
        softmax_module = nn.LogSoftmax(dim=-1)
        prob = softmax_module(scores)
        return prob.data.numpy()

    def predict(self, data):
        self.net.eval()

        data = Variable(torch.from_numpy(data)).float()

        scores = self.net(data)
        _, predicted = torch.max(scores.data, 1)
        return predicted

def save_list(my_list, file_name):
    with open(file_name, "wb") as fp:   #Pickling
        pickle.dump(my_list, fp)


def load_list(file_name):
    with open(file_name, "rb") as fp:   # Unpickling
        return pickle.load(fp)

def train1(mlp, data, label, epoch=None):
        mlp.trained = True
        mlp.net.train()
        number_of_epuchs = mlp.epoch_count if epoch is None else epoch
        for epoch in range(number_of_epuchs):

            accuracy_meter = AverageMeter()
            for i, (batch_data, batch_label) in enumerate(zip(data, label)):
                batch_data = batch_data.reshape(1, -1)

                batch_data = Variable(torch.from_numpy(batch_data)).float()
                batch_label = Variable(torch.from_numpy(batch_label)).long()

                mlp.optimizer.zero_grad()
                score = mlp.net(batch_data)

                loss = mlp.loss_function(score, batch_label)

                loss.backward()
                mlp.optimizer.step()

                acc = accuracy(score, batch_label)
        
                accuracy_meter.update(val=acc, n=batch_data.shape[0])
                print(f"Epoch {epoch+1} completed. Accuracy: {accuracy_meter.avg}")



class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(y_hat, y):
    '''
    y_hat is the model output - a Tensor of shape (n x num_classes)
    y is the ground truth

    How can we implement this function?
    '''
    classes_prediction = y_hat.argmax(dim=1)
    match_ground_truth = classes_prediction == y # -> tensor of booleans
    correct_matches = match_ground_truth.sum()
    return (correct_matches / y_hat.shape[0]).item()

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

    ##################    
    traindata = [[] for i in range(len(spoken))]
    trainlength = [[] for i in range(len(spoken))]

    val_i_end = int(0.2*len(features))
    for i in range(val_i_end, len(features[val_i_end:])):
            for j in range(0, len(spoken)):
                if spoken[j] == labels[i]:
                    traindata[j].append(features[i])
                    trainlength[j].append(len(features[i]))
                    print(f"For model {j} \n Traindata.shape = {(np.concatenate(traindata[j])).shape}  sum= {(sum(np.array(trainlength[j])))}\n")
    
    
    ##################
    #traindata = load_obj('featurelist')

    gmmhmm_module_list = []
    seq_mapper = []

    for i in range(len(spoken)):
            gmmhmm_module_list.append(hmm.GMMHMM(n_components=5, n_mix=18,
                                                 covariance_type='diag', transmat_prior=transmatPrior, startprob_prior=startprobPrior, n_iter=10))


    ##################
    for i, module in enumerate(gmmhmm_module_list):
        module.fit(np.concatenate(traindata[i]),np.array(trainlength[i]))
    ##################
    
    ##################
    for i, module in enumerate(gmmhmm_module_list):
        for data in traindata[i]:
                prob, path = module.decode(np.array(data))
                seq_mapper.append((i, data, path))

    save_list(seq_mapper, 'path.dict')
    ##################

    

    
    phonetic_train_data = [np.zeros((0, 36))] * len(spoken)
    phonetic_train_label = [np.zeros((0, 1))] * len(spoken)

    #language_model = [np.zeros(states_count) for _ in range(len(spoken))]

    for label, data, seq in seq_mapper:
        phonetic_train_data[label] = np.vstack([phonetic_train_data[label], np.array(data)])
        phonetic_train_label[label] = np.vstack([phonetic_train_label[label], np.array(seq).reshape(-1, 1)])

        

    # train DNN network
    #############################################################################
    print('---- training DNN network')
    dnn_module_list = []
    for i in range(len(spoken)):
        dnn_module_list.append(MLP(36, 5))

    inputmlp = [None]*len(dnn_module_list)
    for i in range(len(dnn_module_list)):
        inputmlp[i] = (dnn_module_list[i],phonetic_train_data[i], phonetic_train_label[i])
    
    with Pool(13) as p:
        p.starmap(train1, inputmlp)
    
    
    print('number of DNN:',len(dnn_module_list))

    # create hmm dnn modules
    #############################################################################
    hmm_dnn_module_list = []
    for i in range(len(spoken)):
        hmm_dnn_module_list.append(
            hmm_dnn(dnn_module_list[i],
                    n_components = 5,
                    startprob_prior=gmmhmm_module_list[i].startprob_, transmat_prior=gmmhmm_module_list[i].transmat_,
                    n_iter=50))

    print('number of HMM-DNN:',len(hmm_dnn_module_list))
    # train hmm dnn modules
    #############################################################################
    print('---- training HMM-DNN')
    start_train_time = time.time()

    
    ##################
    input = [None]*len(hmm_dnn_module_list)
    for i in range(len(hmm_dnn_module_list)):
        input[i] = (hmm_dnn_module_list[i],np.concatenate(traindata[i]),np.array(trainlength[i]))
        
    ##################
    with Pool(13) as p:
        hmm_dnn_module_list = p.starmap(fit1, input)
    
    #for i, module in enumerate(hmm_dnn_module_list):
    #    print('HMM_DNN number:',i)
    #    module.fit(traindata[i])
    print("Train Time: ", time.time() - start_train_time)

    save_obj(hmm_dnn_module_list,'hmm_dnn_list_parallel2')

 