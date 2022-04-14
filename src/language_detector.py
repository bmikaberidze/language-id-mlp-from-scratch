from deep_neural_network import DNN

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

class LanguageDetector():

    #Configuration file path
    config_file             = './config.json' 

    #Folder paths
    data_path               = '../data'         
    figures_path            = '../figures'   

    #Raw dataset colamn names
    LANG_COL                = 'lang'
    TEXT_COL                = 'text'
    #Raw dataset file path
    raw_ds_file             = f'{data_path}/sentences.csv'

    #Dataset files paths
    train_ds_file           = f'{data_path}/train.csv' 
    valid_ds_file           = f'{data_path}/valid.csv'
    test_ds_file            = f'{data_path}/test.csv'

    #Featured dataset files pats
    train_feat_ds_file      = f'{data_path}/train_feat.csv'
    valid_feat_ds_file      = f'{data_path}/valid_feat.csv'
    test_feat_ds_file       = f'{data_path}/test_feat.csv'

    #Evaluated model image names
    loss_plot_filename      = 'loss'
    conf_matrix_filename    = 'conf_matrix'  

    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        """
        Initialises language detection model 
        """

        #Read and set Configuration
        self.config = self._json_file_to_simple_nsp(self.config_file)

    # ---------------------------------------------------------------------------------------------------------------------
    def preprocess(self):
        """
        Preprocess dataset: filter, split and generate features
        """
        filtered_ds                                         = self._filter_dataset()
        self.train_ds,   self.valid_ds,   self.test_ds      = self._split_dataset(filtered_ds)
        self.train_feat, self.valid_feat, self.test_feat    = self._generate_features()

    # ---------------------------------------------------------------------------------------------------------------------
    def _filter_dataset(self):
        """
        Filter dataset by sentence length and language. 
        Select the particular number of sentences for each language.
        """

        #Get sentence filtering config
        sent = self.config.preprocessing.filter.sentence

        #Read full dataset
        data = pd.read_csv( self.raw_ds_file,
                            sep                 = '\t', 
                            encoding            = 'utf8', 
                            index_col           = 0,
                            names               = [ self.LANG_COL, self.TEXT_COL ])

        #Filter by text length
        len_cond = [True if sent.len.min <= len(s) <= sent.len.max else False for s in data[self.TEXT_COL]]
        data = data[len_cond]

        #Filter by text language
        data = data[data[self.LANG_COL].isin(self.config.languages)]

        #Select N sentences for each language
        data_trim = pd.DataFrame(columns=[self.LANG_COL, self.TEXT_COL])
        #
        for l in self.config.languages:
            lang_trim = data[data[self.LANG_COL] == l].sample(sent.count, random_state = 100)
            data_trim = pd.concat([data_trim, lang_trim])

        return data_trim

    # ---------------------------------------------------------------------------------------------------------------------
    def _split_dataset(self, data):
        """
        Split dataset into training, validation and testing sub-datasets
        """

        #Get dataset splitting config
        split = self.config.preprocessing.split

        #Create a random train, valid, test splits
        data_shuffle    = data.sample(frac=1)
        #
        data_len        = len(data_shuffle)
        train_indx      = int(data_len * split.train / 100)
        valid_indx      = int(data_len * split.valid / 100) + train_indx
        test_indx       = data_len
        #
        train_ds        = data_shuffle[ 0          : train_indx ]
        valid_ds        = data_shuffle[ train_indx : valid_indx ]
        test_ds         = data_shuffle[ valid_indx : test_indx  ]

        return train_ds, valid_ds, test_ds

    # ---------------------------------------------------------------------------------------------------------------------
    def _generate_features(self):
        """
        Generate n-gram bag-of-words features from datasets
        """

        #Get vocabulary of common trigrams by language
        feature_vocab, feature_names = self._get_feature_vocab()
        
        #Get features generation config
        feat = self.config.preprocessing.features

        #Train count vectoriser using vocabulary
        vectorizer = CountVectorizer(analyzer       = 'char',
                                     ngram_range    = (feat.ngrams.min, feat.ngrams.max),
                                     vocabulary     = feature_vocab)
        
        #Create feature matrices
        train_feat    = self._get_feature_matrix(self.train_ds, feature_names, vectorizer)
        valid_feat    = self._get_feature_matrix(self.valid_ds, feature_names, vectorizer)
        test_feat     = self._get_feature_matrix(self.test_ds,  feature_names, vectorizer)

        #Scale feature matrices (data normalization)
        train_min     = train_feat.min()
        train_max     = train_feat.max()
        train_feat    = self._scale_feature_matrix(train_feat, train_min, train_max)
        valid_feat    = self._scale_feature_matrix(valid_feat, train_min, train_max)
        test_feat     = self._scale_feature_matrix(test_feat,  train_min, train_max)
        
        #Add target variables 
        train_feat    = self._label_feature_matrix(self.train_ds, train_feat)
        valid_feat    = self._label_feature_matrix(self.valid_ds, valid_feat)
        test_feat     = self._label_feature_matrix(self.test_ds,  test_feat)
        
        #Save train, valid, test split
        train_feat.to_csv(self.train_feat_ds_file)
        valid_feat.to_csv(self.valid_feat_ds_file)
        test_feat.to_csv( self.test_feat_ds_file ) 
        
        #Print details
        print('Number of unique features: ', len(train_feat.columns))
        print('Numbers of training, validation and test sentences: ', len(train_feat), len(valid_feat), len(test_feat))

        return train_feat, valid_feat, test_feat

    # ---------------------------------------------------------------------------------------------------------------------
    def _import_features(self):
        """
        Import features datasets
        """

        self.train_feat = pd.read_csv(self.train_feat_ds_file, index_col =0)
        self.valid_feat = pd.read_csv(self.valid_feat_ds_file, index_col =0)
        self.test_feat  = pd.read_csv(self.test_feat_ds_file,  index_col =0)

        #Print details
        print('Number of unique features: ', len(self.train_feat.columns))
        print('Numbers of training, validation and test sentences: ', len(self.train_feat), len(self.valid_feat), len(self.test_feat))
        
    # ---------------------------------------------------------------------------------------------------------------------
    def _get_feature_vocab(self):
        """
        Get vocabulary of common trigrams by language
        """
        
        #Obtain trigrams from each language
        features = {}
        features_set = set()

        for l in self.config.languages:

            #get corpus filtered by language
            corpus = self.train_ds[self.train_ds.lang==l][self.TEXT_COL]

            #get most frequent trigrams from language corpuse
            trigrams = self._get_corp_feature_names(corpus)

            #add to dict and set
            features[l] = trigrams 
            features_set.update(trigrams)

        #create vocabulary list using feature set
        vocab = dict()
        for i,f in enumerate(features_set):
            vocab[f]=i
        
        return vocab, features_set
    
    # ---------------------------------------------------------------------------------------------------------------------
    def _get_corp_feature_names(self, corpus, max_features = 0):
        """
        Returns a list of the N most common character trigrams from a list of sentences
        -------
        Params:
            corpus (matrix): list of sentences
            max_features (integer): max number of features for each language
        """

        #Get features generation config
        feat            = self.config.preprocessing.features
        max_features    = max_features or feat.max

        #Fit the n-gram model
        vectorizer = CountVectorizer(analyzer       = 'char',
                                     ngram_range    = (feat.ngrams.min, feat.ngrams.max),
                                     max_features   = max_features)

        X = vectorizer.fit_transform(corpus)

        #Get model feature names
        feature_names = vectorizer.get_feature_names_out()

        return feature_names
    
    # ---------------------------------------------------------------------------------------------------------------------
    def _get_feature_matrix(self, corpus, feature_names, vectorizer):
        """
        Returns a beg-of-words matrix of given corpus (istead of words, we have n-gram features) 
        -------
        Params:
            corpus (matrix): list of sentences
            feature_names (string array):  list of trigrams
            vectorizer (object): CountVectorizer's object
        """

        X = vectorizer.fit_transform(corpus[self.TEXT_COL])
        corpus_feat = pd.DataFrame(data=X.toarray(), columns=feature_names)

        return corpus_feat

    # ---------------------------------------------------------------------------------------------------------------------
    def _scale_feature_matrix(self, matrix, matrix_min, matrix_max):
        """
        Returns min-max scaled matrix
        -------
        Params:
            corpus (matrix)
            matrix_min (integer)
            matrix_min (integer)
        """
        return (matrix - matrix_min) / (matrix_max - matrix_min)

    # ---------------------------------------------------------------------------------------------------------------------
    def _label_feature_matrix(self, corpus, corpus_feat):
        """
        Label dataset of features (add target column) 
        -------
        Params:
            corpus (matrix): list of sentences
            corpus_feat (matrix):  list of sentences as trigram frequences
        """
        corpus_feat[self.LANG_COL] = list(corpus[self.LANG_COL])
        return corpus_feat
    
    # ---------------------------------------------------------------------------------------------------------------------
    def define( self, 
                hidden_layers = None, 
                activation = None, 
                classifier = None, 
                learn_rate = None, 
                optim_rate = None, 
                regul_rate = None):
        """
        Defines the deep neural network model
        -------
        Params:
            hidden_layers (integer array): dense layer dimensions
            activation (string): nonliniariti function name
            classifier (string): classifier function name
            learn_rate (float): learning rate
            optim_rate (float): optimaizer rate
            regul_rate (float): regularisation rate
        """

        #Import features datasets if needed
        if not hasattr(self, 'train_feat'):
            self._import_features()

        #Get model config
        cnf             = self.config.model
        #
        hidden_layers   = hidden_layers or cnf.hidden_layers
        activation      = activation    or cnf.activation
        classifier      = classifier    or cnf.classifier
        learn_rate      = learn_rate    or cnf.train.learn_rate
        optim_rate      = optim_rate    or cnf.train.optim_rate
        regul_rate      = regul_rate    or cnf.train.regul_rate

        #Define model
        #
        model_input_dim     = len(self.train_feat.columns) - 1
        model_output_dim    = len(self.config.languages)
        #
        model = DNN(model_input_dim, learn_rate, optim_rate, regul_rate)
        #
        [ model.add(hidden_layer, activation) for hidden_layer in hidden_layers ]            
        #
        model.add(model_output_dim, classifier)
        #
        self.model = model

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Old code for defining keras sequential model
        # #
        # from keras.models import Sequential
        # from keras.layers import Dense
        # model = Sequential()
        # model.add(Dense(hidden_layers[0], input_dim=len(self.train_feat.columns)-1, activation=activation))
        # for hidden_layer in hidden_layers[1:]: 
        #     model.add(Dense(hidden_layer, activation=activation))
        # model.add(Dense(len(self.config.languages), activation=classifier))
        # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics= [ "accuracy" ])
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
   
    # ---------------------------------------------------------------------------------------------------------------------
    def train(self, epochs = 0, batch_size = 0, validate = None ):
        """
        Train the model on training features dataaset
        -------      
        Params:            
            epochs (integer): number of training epochs
            batch_size (integer): input vectors batch size
            validate (boolean): calculate validation loss in parallel  
        """

        #Define model if needed
        if not hasattr(self, 'model'):
            self.define()

        #Get model training config
        cnf             = self.config.model.train
        #
        epochs          = epochs        or cnf.epochs
        batch_size      = batch_size    or cnf.batch_size
        #
        validate        = validate if validate is not None else cnf.validate

        #Prepare training inputs and targets
        x, y                = self._get_inputs_and_targets(self.train_feat)
        x_valid, y_valid    = self._get_inputs_and_targets(self.valid_feat) if validate else (None, None)
        
        #Train model
        self.model.train(x, y, epochs, batch_size, x_valid, y_valid)
        # self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

    # ---------------------------------------------------------------------------------------------------------------------
    def tuning(self):
        """
        Hyper-parameter tuning. train on different parameters and validate
        """

        #Get model tuning config
        tun = self.config.model.tuning

        self.tunnin_results = []

        for hl in tun.hidden_layers:
            for epochs in tun.epochs:
                for batch_size in tun.batch_sizes:
                    for learn_rate in tun.learn_rates:
                        for optim_rate in tun.optim_rates:
                            for regul_rate in tun.regul_rates:

                                result = {}

                                self.define(hidden_layers   = hl, 
                                            activation      = self.config.model.activation, 
                                            classifier      = self.config.model.classifier, 
                                            learn_rate      = learn_rate, 
                                            optim_rate      = optim_rate, 
                                            regul_rate      = regul_rate)
                                
                                self.train( epochs          = epochs, 
                                            batch_size      = batch_size, 
                                            validate        = False)

                                result['hidden_layers']     = hl
                                result['learn_rate']        = learn_rate
                                result['optim_rate']        = optim_rate
                                result['regul_rate']        = regul_rate

                                result['epochs']            = epochs
                                result['batch_size']        = batch_size

                                result['train_loss']        = self.model.mean_loss
                                result['valid_loss']        = self.model.valid_mean_loss

                                result['train_acc'], cm     = self.evaluate(self.train_feat)
                                result['valid_acc'], cm     = self.evaluate(self.valid_feat)

                                self.tunnin_results.append(result)

        self.tunnin_results     = pd.DataFrame(self.tunnin_results)
        self.best_tun_res       = self.tunnin_results
        self.best_tun_res       = self.best_tun_res[self.best_tun_res.valid_acc == self.best_tun_res.valid_acc.max()]

        print(self.best_tun_res)
        print(self.tunnin_results)

    # ---------------------------------------------------------------------------------------------------------------------
    def evaluate(self, corpus_feat):
        """        
        Evaluate and return model's accuracy and confusion matrix on given features corpus
        -------      
        Params:            
            corpus_feat (matrix): list of sentences as trigram frequences
        """

        #Prepare prediction inputs and targets
        inputs, targets = self._get_inputs_and_targets(corpus_feat, encode = False)

        #Predict
        predictions     = self.model.predict(inputs) 
        labels          = np.argmax(predictions, axis = 1)
        lang_labels     = self._encoder.inverse_transform(labels)

        #Get models accuracy
        accuracy        = accuracy_score(targets, lang_labels)

        #Get confusion matrix
        conf_matrix     = confusion_matrix(targets, lang_labels)

        return accuracy, conf_matrix

    # ---------------------------------------------------------------------------------------------------------------------
    def test(self):
        """
        Tests model on test features dataset and prints model's accuracy and confusion matrix
        """
            
        #Import features datasets if needed
        if not hasattr(self, 'test_feat'):
            self._import_features()

        #Get accuracy and confusion matrix
        accuracy, conf_matrix = self.evaluate(self.test_feat)
        accuracy = round(accuracy*100, 2)

        #Print accuracy
        print(f'Model\'s accuracy: {accuracy}%') #98.26% | my with keras: 98.83% | my with my model: 98.21%

        #Draw and save loss chart 
        self._draw_loss_chart(accuracy)

        #Draw and save confusion matrix
        self._draw_conf_matrix(conf_matrix, accuracy)

    # ---------------------------------------------------------------------------------------------------------------------
    def _draw_loss_chart(self, accuracy):
        """
        Draw and save loss chart       
        -------      
        Params:  
            accuracy (float): models accuracy
        """
        plt.plot(self.model.losses, linewidth=0.7)
        plt.savefig(f'{self.figures_path}/{accuracy}_{self.loss_plot_filename}.png')
        
    # ---------------------------------------------------------------------------------------------------------------------
    def _draw_conf_matrix(self, conf_matrix, accuracy):
        """
        Draw and save confusion matrix       
        -------      
        Params:  
            conf_matrix (matrix):  models confusion matrix
            accuracy (float): models accuracy
        """

        #Plot confusion matrix heatmap
        conf_matrix_df  = pd.DataFrame(conf_matrix, columns=self.config.languages, index=self.config.languages)
        plt.figure(figsize = (10, 10), facecolor='w', edgecolor='k')
        sns.set(font_scale = 1.5)
        sns.heatmap(conf_matrix_df, cmap='coolwarm', annot=True, fmt='.5g', cbar=False)
        plt.xlabel('Predicted', fontsize=22)
        plt.ylabel('Actual', fontsize=22)
        
        #Save confusion matrix heatmap as image
        plt.savefig(f'{self.figures_path}/{accuracy}_{self.conf_matrix_filename}.png', format='png', dpi=150)

    # ---------------------------------------------------------------------------------------------------------------------
    def _get_inputs_and_targets(self, corpus_feat, encode = True):
        """
        Returns an inputs and targets for a given features dataset       
        -------      
        Params:  
            corpus_feat (matrix):  list of sentences as trigram frequences
            encode (boolean): encode labels as one hot vectors
        """

        #Create encoder if needed
        if not hasattr(self, 'encoder'):
            self._create_encoder()

        inputs  = corpus_feat.drop(self.LANG_COL, axis=1)
        targets = self._encode(corpus_feat[self.LANG_COL]) if encode else corpus_feat[self.LANG_COL] 

        return inputs, targets

    # ---------------------------------------------------------------------------------------------------------------------
    def _encode(self, y):
        """
        Returns a list of one hot encodings      
        -------  
        Params:
            y (string array): list of language labels
        """

        #Create encoder if needed
        if not hasattr(self, 'encoder'):
            self._create_encoder()
        
        #Represent y labels as on hot encodings
        y_encoded = self._encoder.transform(y)
        y_dummy = np_utils.to_categorical(y_encoded)
        
        return y_dummy

    # ---------------------------------------------------------------------------------------------------------------------
    def _create_encoder(self):
        """
        Creates _encoder for one hot encoding 
        """
        #Create and fit _encoder
        self._encoder = LabelEncoder()
        self._encoder.fit(self.config.languages)

    # ---------------------------------------------------------------------------------------------------------------------
    #Copyed from prev project for configuration reading
    def _json_file_to_simple_nsp(self, json_file):
        with open(json_file) as f:
            return self._dict_to_simple_nsp(json.load(f))
    def _dict_to_simple_nsp(self, dictionary = {}):
        return self._json_load_simple_nsp(self._json_dumps(dictionary))
    def _json_load_simple_nsp(self, j):
        return json.loads(j, object_hook=lambda d: SimpleNamespace(**d))
    def _json_dumps(self, obj, **kwargs):
        return json.dumps(obj, sort_keys=False, indent=4, ensure_ascii=False, **kwargs)
    
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
                                                                                                                                                                                                                                            