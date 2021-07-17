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
def CCR(test_labels, classifier_labels):
    CCR_Val = 0
    for label_idx in range(len(test_labels)):
        if test_labels[label_idx] == classifier_labels[label_idx]:
            CCR_Val += 1

    return 100.0 * CCR_Val / len(test_labels)



def plot_confusion_matrix(test_labels, classifier_labels, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()

    cm = confusion_matrix(test_labels, classifier_labels)

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    title += '\nCCR is = ' + str(CCR(test_labels, classifier_labels))

    print(title)

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):

        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        #self.aucoustic_model = aucoustic_model
        #self.observation_count = observation_count
        self.mlp = mlp
        self.mlp.info()

    def _compute_log_likelihood(self, X):
        prob = self.mlp.log_probablity(X).astype(type(X[0, 0]))

        #prob = prob - np.log(self.observation_count)
        #prob = prob - np.log(self.aucoustic_model + (self.aucoustic_model == 0))

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

                self._do_mstep(stats)
                if update_dnn:
                    temp_path = np.zeros((0, 1))
                    for k, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
                        temp_path = np.vstack([temp_path, np.array(path_list[k]).reshape(-1, 1)])
                    self.mlp.train(X, temp_path, 20)

        #        acoustic_model = np.zeros(self.n_components)
        #        for i, j in iter_from_X_lengths(X, lengths):
        #            logprob, state_sequence = self.decode(X[i:j], algorithm="viterbi")
        #            for state in state_sequence:
        #                acoustic_model[state] += 1
        #        self.aucoustic_model = acoustic_model / np.sum(acoustic_model)

            self.monitor_.report(curr_logprob)
            if self.monitor_.iter == self.monitor_.n_iter or \
                    (len(self.monitor_.history) == 2 and
                     abs(self.monitor_.history[1] - self.monitor_.history[0]) < self.monitor_.tol * abs(
                                self.monitor_.history[1])):
                break

        print('----------------------------------------------')
        return self

    def _do_mstep(self, stats):
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





class NeuralNetwork(nn.Module):

    def __init__(self, feature_size, class_count):
        super(NeuralNetwork, self).__init__()

        mid1_neuron = 40
        mid2_neuron = 30
        mid3_neuron = 20

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
        self.optimizer = torch.optim.RMSprop(self.net.parameters())
        self.epoch_count = 20
        self.trained = False

    def train(self, data, label, epoch=None):
        self.trained = True
        self.net.train()
        number_of_epuchs = self.epoch_count if epoch is None else epoch
        for epoch in range(number_of_epuchs):

            for i, (batch_data, batch_label) in enumerate(zip(data, label)):
                batch_data = batch_data.reshape(1, -1)

                batch_data = Variable(torch.from_numpy(batch_data)).float()
                batch_label = Variable(torch.from_numpy(batch_label)).long()

                self.optimizer.zero_grad()
                score = self.net(batch_data)

                loss = self.loss_function(score, batch_label)

                loss.backward()
                self.optimizer.step()

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



if __name__ == "__main__":

    features = load_obj('featurelist')
    labels = load_obj('labelslist')
    spoken = list(set(labels))

    hmm_dnn_module_list = load_obj('hmm_dnn_list')

    predicted_label_list = list()

    score_list = [0 for _ in range(len(spoken))]


    val_i_end = int(0.2*len(features))
   
    for j in range(len(features[0:val_i_end])):
        for i, module in enumerate(hmm_dnn_module_list):
            score_list[i], _ = module.decode(features[j])
        predicted_label_list.append(spoken[np.argmax(score_list)])
    
    
    #plot_confusion_matrix(labels[0:val_i_end], predicted_label_list, range(10))
    
    conf_mat = confusion_matrix(labels[0:val_i_end], predicted_label_list, labels=list(set(list(labels[:val_i_end]) )) , normalize="true")
    df_conf_mat = pd.DataFrame(conf_mat)
    df_conf_mat.columns = list(set(list(labels[:val_i_end])))
    df_conf_mat.index = list(set(list(labels[:val_i_end])))
    print(df_conf_mat.to_string())

    print("------------on training--------------")

    for j in range(val_i_end, len(features[val_i_end:])):
        for i, module in enumerate(hmm_dnn_module_list):
            score_list[i], _ = module.decode(features[j])
        predicted_label_list.append(spoken[np.argmax(score_list)])

    
    conf_mat = confusion_matrix(labels[val_i_end:], predicted_label_list, labels=list(set(list(labels[val_i_end:]) )) , normalize="true")
    df_conf_mat = pd.DataFrame(conf_mat)
    df_conf_mat.columns = list(set(list(labels[val_i_end:])))
    df_conf_mat.index = list(set(list(labels[val_i_end:])))
    print(df_conf_mat.to_string())