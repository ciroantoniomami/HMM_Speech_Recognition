import collections
import pickle
from pyexpat import features
from zoneinfo import available_timezones

import librosa
import pandas as pd
import numpy as np
import scipy.stats as sp
from scipy.io import wavfile
from operator import itemgetter
from pathlib import Path
import os
from hmmlearn import hmm
from python_speech_features import mfcc, delta
from sklearn.metrics import classification_report, confusion_matrix



def bic_general(likelihood_fn, k, X):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
           X - array. Data that been fitted upon.
    """
    bic = np.log(len(X))*k - 2*likelihood_fn(X)
    return bic

def aic_general(likelihood_fn, k, X):
    """likelihood_fn: Function. Should take as input X and give out   the log likelihood
                  of the data under the fitted model.
           k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
           X - array. Data that been fitted upon.
    """
    aic = 2*k - 2*likelihood_fn(X)
    return aic

def bic_hmmlearn(X):
    lowest_bic = np.infty
    lowest_aic = np.infty
    bic = []
    aic = []
    n_states_range = range(10,30)
    best_num_states_bic = 1
    best_num_states_aic = 1
    for n_components in n_states_range:
        hmm_curr = hmm.GMMHMM(n_components=n_components, n_mix= 5 ,covariance_type='diag')
        hmm_curr.fit(X)

        # Calculate number of free parameters
        # free_parameters = for_means + for_covars + for_transmat + for_startprob
        # for_means & for_covars = n_features*n_components
        n_features = hmm_curr.n_features
        free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)

        bic_curr = bic_general(hmm_curr.score, free_parameters, X)
        bic.append(bic_curr)
        aic_curr = aic_general(hmm_curr.score, free_parameters, X)
        aic.append(aic_curr)
        if bic_curr < lowest_bic:
            best_num_states_bic = n_components
            lowest_bic = bic_curr
        
        if aic_curr < lowest_aic:
            best_num_states_aic = n_components
            lowest_aic = aic_curr
        best_hmm = hmm_curr

    return (best_hmm, bic, aic,best_num_states_bic,best_num_states_aic)



def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)



def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


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

    fpaths = [f for f in os.listdir('train/audio2/') if os.path.splitext(f)[1] == '.wav']
    labels = [file.split("_")[-1][1:-4] for file in fpaths]
    spoken = list(set(labels))
    
    
    features = []
    for n, file in enumerate(fpaths):
        
        sample_rate, signal = read_wav('train/audio2/' + file)
        
        
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


def cross_validation(features, labels , spoken):
    d = {key:0 for key in range(3,7)}
    for num_states in range(3,7):
        acc = []
        for index in range(5):
            model = HMMSpeechRecog(features, labels , spoken,index)
            model.train(m_num_of_HMMStates=num_states)
            acc.append(model.test())
        print("Mean accuracy for number of states:",num_states)
        print(sum(acc)/len(acc))
        d[num_states] = sum(acc)/len(acc))

    return d 

class HMMSpeechRecog(object):
    def __init__(self,  features,labels,spoken,index,filespath=Path('train/audio/'), val_p=0.2):
        self.filespath = Path(filespath)
        self.val_p = val_p
        self.labels = labels
        self.features = features
        self.spoken = spoken
        #self._get_val_index_end()
        self.left = int(index/5*len(self.features))
        self.right = int((index+1)/5*len(self.features))
        print(self.left,self.right)
        self._get_gmmhmmindex_dict()
        

    def _get_val_index_end(self):
        self.val_i_end = int(len(self.features) * self.val_p)

    def _get_gmmhmmindex_dict(self):
        self.gmmhmmindexdict = {}
        self.indexgmmhmmdict = {}
        index = 0
        for word in self.spoken:
            self.gmmhmmindexdict[word] = index
            self.indexgmmhmmdict[index] = word
            index = index + 1

    
    
    

    def initByBakis(self,m_num_of_HMMStates):

        tmp_p = 1.0/(m_num_of_HMMStates)
        transmatPrior = np.zeros((m_num_of_HMMStates,m_num_of_HMMStates))
        for i in range(0,m_num_of_HMMStates):
            for j in range(0,m_num_of_HMMStates):
                transmatPrior[i][j] = tmp_p
        transmatPrior=np.triu(transmatPrior)



        startprobPrior = np.zeros(m_num_of_HMMStates)
        startprobPrior[0] = 1
        

        return startprobPrior, transmatPrior

    def init_model(self, m_num_of_HMMStates, m_bakisLevel):
        self.m_num_of_HMMStates = m_num_of_HMMStates
        self.m_bakisLevel = m_bakisLevel
        self.m_startprobPrior, self.m_transmatPrior = self.initByBakis( m_num_of_HMMStates)

    def train(self, m_num_of_HMMStates=5, m_bakisLevel=2, m_num_of_mixtures=18, m_covarianceType='diag', m_n_iter=10):
        self.m_num_of_mixtures = m_num_of_mixtures
        self.m_covarianceType = m_covarianceType
        self.m_n_iter = m_n_iter

        self.init_model(m_num_of_HMMStates, m_bakisLevel)
        self.speechmodels = [None] * len(self.spoken)

        for key in self.gmmhmmindexdict:
            self.speechmodels[self.gmmhmmindexdict[key]] = SpeechModel(self.gmmhmmindexdict[key], key,
                                                                       self.m_num_of_HMMStates,
                                                                       self.m_num_of_mixtures, self.m_transmatPrior,
                                                                       self.m_startprobPrior, self.m_covarianceType,
                                                                       self.m_n_iter,
                                                                       self.features[0].shape[1])

        for i in range(len(self.features)):
            if (i == self.left):
                i+= (self.right -self.left)
                continue 
            for j in range(0, len(self.speechmodels)):
                if int(self.speechmodels[j].Class) == int(self.gmmhmmindexdict[self.labels[i]]):
                    self.speechmodels[j].traindata = np.concatenate(
                        (self.speechmodels[j].traindata, self.features[i]))

        for speechmodel in self.speechmodels:
            speechmodel.model.fit(speechmodel.traindata)
        n_spoken = len(self.spoken)
        save_obj(speechmodel, "modelslist2")
        print(f'Training completed -- {n_spoken} GMM-HMM models are built for {n_spoken} different types of words')
        #self.pickle("obj/regoc")

    def get_confusion_matrix(self, real_y, pred_y, labels):
        conf_mat = confusion_matrix(real_y, pred_y, labels=labels , normalize="true")
        df_conf_mat = pd.DataFrame(conf_mat)
        df_conf_mat.columns = labels
        df_conf_mat.index = labels
        return df_conf_mat

    def get_accuracy(self, save_path=None):
        self.accuracy = 0.0
        count = 0
        predicted_labels = []

        print("")
        print("Prediction for test set:")

        for i in range(self.left, self.right):
            predicted_label_i = self.m_PredictionlabelList[i]
            predicted_labels.append(self.indexgmmhmmdict[predicted_label_i])
            if self.gmmhmmindexdict[self.labels[i]] == predicted_label_i:
                count = count + 1

        report = classification_report(self.labels[self.left:self.right], predicted_labels)
        df_conf_mat = self.get_confusion_matrix(self.labels[self.left:self.right], predicted_labels,
                                                labels=self.spoken)
        print(report)
        print(df_conf_mat.to_string())
        if save_path is not None:
            Path(save_path).write_text(f'nr of files in test set: {count}\n{report}'
                                       f'\nConfusion matrix (y-axis real label, x-axis predicted label):\n'
                                       f'{df_conf_mat.to_string()}')

    def test(self, save_path=None):
        # Testing
        self.m_PredictionlabelList = []
        count = 0
        tot = 0
        for i in range(self.left, self.right):
            scores = []
            tot += 1
            for speechmodel in self.speechmodels:
                scores.append(speechmodel.model.score(self.features[i]))
            id = scores.index(max(scores))
            if self.speechmodels[id].label == labels[i]:
                count +=1
            self.m_PredictionlabelList.append(self.speechmodels[id].Class)
            #print(str(np.round(scores, 3)) + " " + str(max(np.round(scores, 3))) + " " + ":" +
            #      self.speechmodels[id].label)
        #self.get_accuracy(save_path=save_path)

        return float(count/tot)

        

    def _predict(self, features):
        Model_confidence = collections.namedtuple('model_prediction', ('name', 'score'))
        predicted_labels_confs = []

        for i in range(0, len(features)):
            file_scores_confs = []
            for speechmodel in self.speechmodels:
                score = speechmodel.model.score(features[i])
                label = speechmodel.label
                file_scores_confs.append(Model_confidence(name=label, score=score))
                file_scores_confs = sorted(file_scores_confs, key=itemgetter(1), reverse=True)
            predicted_labels_confs.append(file_scores_confs)

        return predicted_labels_confs

    def predict_files(self, files):
        features = self._read_wav_get_features(files, eval=True)
        predicted_labels_confs = self._predict(features)
        return predicted_labels_confs

    def predict_signal(self, signal, sample_rate):
        features = self._get_features(signal, sample_rate)
        predicted_labels_confs = self._predict(features)
        return predicted_labels_confs

    # Calcuation of  mean ,entropy and relative entropy parameters
    '''Entropyvalues for the 3 hidden states and 100 samples'''

    def entropy_calculator(self, dataarray, meanvalues, sigmavalues):
        entropyvals = []
        for i in range(0, len(dataarray[0])):
            totallogpdf = 0
            entropy = 0
            for j in range(0, len(dataarray)):
                totallogpdf += sp.norm.logpdf(dataarray[j, i], meanvalues[i], sigmavalues[i])
                entropy = (-1 * totallogpdf) / len(dataarray)
            entropyvals.append(entropy)
        return entropyvals

    '''Relative Entropyvalues for the 6 columns of the given data and sampled values'''

    def relative_entropy_calculator(self, givendata, samplesdata, givendatasigmavals, sampledsigmavals,
                                    givendatameanvals, sampledmeanvals):
        absgivendatasigmavals = [abs(number) for number in givendatasigmavals]
        abssampleddatasigmavals = [abs(number) for number in sampledsigmavals]
        relativeentropyvals = []

        for i in range(0, len(givendata[0])):
            totallogpdf = 0
            relativeentropy = 0
            for j in range(0, len(givendata)):
                totallogpdf += (sp.norm.logpdf(samplesdata[j, i], sampledmeanvals[i],
                                               abssampleddatasigmavals[i]) - sp.norm.logpdf(givendata[j, i],
                                                                                            givendatameanvals[i],
                                                                                            absgivendatasigmavals[i]))
                relativeentropy = (-1 * totallogpdf) / float(len(givendata))
            relativeentropyvals.append(relativeentropy)
        return relativeentropyvals

    def calc_mean_entropy(self):
        for speechmodel in self.speechmodels:
            print("For GMMHMM with label:" + speechmodel.label)
            samplesdata, state_sequence = speechmodel.model.sample(n_samples=len(speechmodel.traindata))

            sigmavals = []
            meanvals = []

            for i in range(0, len(speechmodel.traindata[0])):
                sigmavals.append(np.mean(speechmodel.traindata[:, i]))
                meanvals.append(np.std(speechmodel.traindata[:, i]))

            sampledmeanvals = []
            sampledsigmavals = []

            for i in range(0, len(samplesdata[0])):
                sampledmeanvals.append(np.mean(samplesdata[:, i]))
                sampledsigmavals.append(np.std(samplesdata[:, i]))

            GivenDataEntropyVals = self.entropy_calculator(speechmodel.traindata, meanvals, meanvals)
            SampledValuesEntropyVals = self.entropy_calculator(samplesdata, sampledmeanvals, sampledsigmavals)
            RelativeEntropy = self.relative_entropy_calculator(speechmodel.traindata, samplesdata, sigmavals,
                                                               sampledsigmavals, meanvals, sampledmeanvals)

            print("MeanforGivenDataValues:")
            roundedmeanvals = np.round(meanvals, 3)
            print(str(roundedmeanvals))
            print("")

            print("EntropyforGivenDataValues:")
            roundedentropyvals = np.round(GivenDataEntropyVals, 3)
            print(str(roundedentropyvals))
            print("")

            print("MeanforSampleddatavalues:")
            roundedsampledmeanvals = np.round(sampledmeanvals, 3)
            print(str(roundedsampledmeanvals))
            print("")

            print("EntropyforSampledDataValues:")
            roundedsampledentvals = np.round(SampledValuesEntropyVals, 3)
            print(str(roundedsampledentvals))
            print("")

            print("RelativeEntopyValues:")
            roundedrelativeentvals = np.round(RelativeEntropy, 3)
            print(str(roundedrelativeentvals))
            print("")

    def pickle(self, filename):
        '''save model to file'''
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def unpickle(filename):
        '''read model from file'''
        with open(filename, 'rb') as f:
            return pickle.load(f)


class SpeechModel:
    def __init__(self, Class, label, m_num_of_HMMStates, m_num_of_mixtures, m_transmatPrior, m_startprobPrior,
                 m_covarianceType='diag', m_n_iter=10, n_features_traindata=12):
        self.traindata = np.zeros((0, n_features_traindata))
        self.Class = Class
        self.label = label
        self.model = hmm.GMMHMM(n_components=m_num_of_HMMStates, n_mix=m_num_of_mixtures,
                                transmat_prior=m_transmatPrior, startprob_prior=m_startprobPrior,
                                covariance_type=m_covarianceType, n_iter=m_n_iter)




if __name__ == "__main__":
    features, labels , spoken = read_wav_get_features()
    d = cross_validation(features, labels , spoken)
    print(d)
    #data = {word : np.zeros((0,36)) for word in spoken}
    #for i in range(len(features)):
    #    for word in spoken:
    #        if labels[i] == word:
    #            data[word] = np.concatenate((data[word],features[i]))
    #
    #for word in spoken:
    #    model, bic, aic ,  best_num_states_bic, best_num_states_aic = bic_hmmlearn(data[word])
    #    print("Best num_states for word:",word)
    #    print(bic)
    #    print(aic)
    #    print(best_num_states_bic)
    #    print(best_num_states_aic)
    #model = HMMSpeechRecog(load=False)
    #model = HMMSpeechRecog()
    #model.train()
    #model.test()
    #predicted_labels = model.predict_files(['data/test_data/apple15.wav'])
    #print(f'predicted label {predicted_labels}')