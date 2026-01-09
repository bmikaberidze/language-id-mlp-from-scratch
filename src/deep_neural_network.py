import numpy as np
from src.dense_layer import DL
import matplotlib.pyplot as plt

import warnings

class DNN:

    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_dim, learn_rate, optim_rate, regul_rate):
        """
        Initialises deep neural network model 
        -------
        Params:
            input_dim (integer): dimention of input vector
            learn_rate (float): learning rate
            optim_rate (float): optimaizer rate
            regul_rate (float): regularisation rate
        """
        self.layers         = []
        self.input_dim      = input_dim
        #
        self.learn_rate     = learn_rate
        self.optim_rate     = optim_rate
        self.regul_rate     = regul_rate
        #
        self.losses         = []
        self.index          = 0 

    # ---------------------------------------------------------------------------------------------------------------------
    def add(self, dim, activation):
        """
        Add dance layer to model 
        -------
        Params:
            dim (integer): dimention of layer vector
            activation (string): activation function name
        """
        input_dim = self.layers[-1].dim if len(self.layers) else self.input_dim
        #
        layer = DL(dim, input_dim, activation)
        #
        self.layers.append(layer)

    # ---------------------------------------------------------------------------------------------------------------------
    def predict(self, input):
        """
        Return prediction according input
        -------
        Params:
            input (vector or matrix): neural network's input vector or vector batch
        """        
        input = np.array([input]) if input.ndim == 1 else input            
        input = np.transpose(input)

        return np.transpose(self._feed_forward(input))

    # ---------------------------------------------------------------------------------------------------------------------
    def train(self, inputs, targets, epochs, batch_size, valid_inputs = None, valid_targets = None):
        """
        Train deep neural network
        -------
        Params:
            inputs (matrix): all input vectors as matrix (dataset)
            targets (matrix): labels for all input vectors as one hot vectors matrix
            epochs (integer): number of training epochs
            batch_size (integer): input vectors batch size to feed forward before each loss calculation, backpropagation and weight updates 
            valid_inputs (matrix): all validation vectors as matrix 
            valid_targets (matrix): labels for all validation vectors as one hot vectors matrix
        """

        #Set training parameters
        self.epochs     = epochs
        self.batch_size = batch_size

        #Set validation data
        if valid_inputs is not None:
            self.valid_inputs   = np.transpose(valid_inputs)
            self.valid_targets  = np.transpose(valid_targets)

        #Cycle epoches
        self._cycle_epochs(inputs, targets)

    # ---------------------------------------------------------------------------------------------------------------------
    def _cycle_epochs(self, inputs, targets, epoch_indx = 0):
        """
        Cycle training epoches recursively
        -------
        Params:
            inputs (marix): all input vectors as matrix (dataset)
            targets (matrix): labels for all input vectors as one hot vectors
            epoch_indx (integer): index of training epochs (recursion index)
        """

        if self.epochs > epoch_indx:

            #Decreasing lerning rate through epoches
            self.learn_rate = self.learn_rate / (5 if epoch_indx else 1)

            print(f'Epoch - {epoch_indx + 1}/{self.epochs} - Learning rate: {self.learn_rate} - - - - - - - - - - - - - - # ')

            #Count of input veqtors
            inputs_cnt = inputs.shape[0]

            #Process inputs by batches
            for start in range(0, inputs_cnt, self.batch_size):
                end = start + self.batch_size

                inputs_batch    = inputs[start : end][:]
                targets_batch   = targets[start : end][:]

                self._fit_batch(inputs_batch, targets_batch)

            #Call cycle_epochs recursively
            self._cycle_epochs(inputs, targets, epoch_indx + 1)
    
    # ---------------------------------------------------------------------------------------------------------------------
    def _fit_batch(self, inputs_batch, targets_batch):
        """
        Feed forward all input vectors in batch, calculate loss
        -------
        Params:
            inputs_batch (marix): batche of input vectors as matrix
            targets_batch (matrix): batch of labels for input vectors as one hot vectors
        """

        #Feed forwarde all input vectors batch
        inputs_batch  = np.transpose(inputs_batch)
        targets_batch = np.transpose(targets_batch)        
        predict_batch = self._feed_forward(inputs_batch)

        #Calculate loss and its derivative with predicted vectors batch
        self.loss   = self._calculate_loss(predict_batch, targets_batch)
        loss_deriv  = self._loss_deriv(predict_batch, targets_batch)
        # self.loss   = self._catch_warning(self._calculate_loss, [predict_batch, targets_batch])
        # loss_deriv  = self._catch_warning(self._loss_deriv,     [predict_batch, targets_batch])

        #Back propagate gradient and update weights        
        self._back_propagate(loss_deriv)

        #Gether input batch and validation batch loss means 
        self.valid_mean_loss = 0
        if hasattr(self, 'valid_inputs'):
            valid_predicts  = self._feed_forward(self.valid_inputs)
            self.valid_mean_loss = np.mean(self._calculate_loss(valid_predicts, self.valid_targets))
        self.mean_loss = np.mean(self.loss)
        self.losses.append(self.mean_loss) if not self.valid_mean_loss else self.losses.append([self.mean_loss, self.valid_mean_loss])

        #Print batch loss mean
        self.index += 1
        # print(f'{self.index}. Loss: {self.mean_loss}')
        
    # ---------------------------------------------------------------------------------------------------------------------
    def _feed_forward(self, input_batch, layer_indx = 0):
        """
        Feeds input vectors batch forward recursively through all layers and return prediction
        -------
        Params:
            input_batch (matrix): neural network's input vectors batch
            layer_indx (integer): layer index (recursion index)
        """

        if len(self.layers) > layer_indx:

            layer           = self.layers[layer_indx]
            output_batch    = layer.feed_forward(input_batch)

            return self._feed_forward(output_batch, layer_indx + 1)

        else:
            return input_batch

    # ---------------------------------------------------------------------------------------------------------------------
    def _back_propagate(self, down_loss_deriv, layer_indx = 0):
        """
        Back propagate derivative of loss and update weights
        -------
        Params:
            down_loss_deriv (matrix): loss derivative with current layer vector
            layer_indx (integer): layer index (recursion index)
        """
        
        if len(self.layers) > layer_indx:

            layer               = self.layers[-layer_indx-1]
            down_loss_deriv     = layer.back_propagate(down_loss_deriv, self.learn_rate, self.optim_rate, self.regul_rate)

            self._back_propagate(down_loss_deriv, layer_indx + 1)
    
    # ---------------------------------------------------------------------------------------------------------------------
    def _calculate_loss(self, predict_batch, target_batch):
        """
        Return batch loss
        -------
        Params:
            predict_batch (matrix): batche of predicted vectors as matrix
            targets_batch (matrix): batch of labels for input vectors as one hot vectors
        """
        return -(target_batch*np.log(predict_batch)).sum(axis=0)

    # ---------------------------------------------------------------------------------------------------------------------
    def _loss_deriv(self, predict_batch, target_batch):
        """
        Return derivative of loss function with predicted vectors batch
        -------
        Params:
            predict_batch (matrix): batche of predicted vectors as matrix
            targets_batch (matrix): batch of labels for input vectors as one hot vectors
        """
        return -target_batch/predict_batch

    # ---------------------------------------------------------------------------------------------------------------------
    def _catch_warning(self, callback, params):
        
        warnings.filterwarnings("error")
        try:
            value = callback(*params)
        except RuntimeWarning:
            warnings.filterwarnings("ignore")
            value = callback(*params)
            print(value, *params)

        warnings.filterwarnings("default")

        return value
