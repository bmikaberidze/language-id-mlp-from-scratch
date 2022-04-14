#At first, the class was developed for feeding forward and backpropagating only one input vector, 
#then it was reconstructed to process input vectors batches.
#Hance, consider vectors as vector batches in parameter naming.

import numpy as np
from scipy.special import softmax

class DL:

    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(self, dim, input_dim, activation, prin = False):
        """
        Initialisation of denselayer 
        -------
        Params:
            dim (integer): dimention of layer vector
            input_dim (integer): dimention of input vector
            activation (string): activation function name
            print (boolean): print logs or not
        """

        self._prin              = prin

        self.dim                = dim
        self.input_dim          = input_dim
        self.activation         = activation   

        self.weights            = self._xavier_initialisation(dim, input_dim)
        
        self.last_delta_weights = 0

        self._print(self.weights)

    # ---------------------------------------------------------------------------------------------------------------------
    def _xavier_initialisation(self, x, y): 
        """
        Returns a (x,y) size matrix, initialized with xavier
        -------
        Params:
            x (integer): number of matrix rows
            y (integer): number of matrix columns
        """       

        #For biased member of linear function
        y += 1
        limit = np.sqrt(6 / float(x + y))
        w = np.random.uniform(low=-limit, high=limit, size=(y,x))
        w[-1:] = np.zeros(x)
        return np.transpose(w)

    # ---------------------------------------------------------------------------------------------------------------------
    def feed_forward(self, input_vec, training = False):
        """
        Returns dance layer vector. Additionaly calculate derivatives while training
        -------
        Params:
            input_vec (vector): dance layer input vector
            training (boolean): calculate derivatives if Ture
        """
                            #For biased member of linear function        
        self.input_vec      = np.append(input_vec, [np.ones(input_vec.shape[1])], axis=0)
        self.act_input_vec  = self._weigh(self.input_vec)
        self.vector         = self._activate(self.act_input_vec)

        self._print(self.input_vec)
        self._print(self.act_input_vec)
        self._print(self.vector)

        return self.vector

    # ---------------------------------------------------------------------------------------------------------------------
    def back_propagate(self, down_loss_deriv, learn_rate, optim_rate, regul_rate):
        """
        Returns loss derivative with input vector. Updates weights.
        -------
        Params:
            down_loss_deriv (matrix): loss derivative with current layer vector
            learn_rate (float): learning rate
            optim_rate (float): optimaizer rate
            regul_rate (float): regularisation rate
        """

        deriv_active = self._deriv_active(down_loss_deriv)
        deriv_weight = self._deriv_weight(deriv_active)

        self._update_weights(deriv_weight, learn_rate, optim_rate, regul_rate)

        return self._deriv_input(deriv_active)
    
    # ---------------------------------------------------------------------------------------------------------------------
    def _weigh(self, input_vec):
        """
        Returns weights matrix multiplied by input vector 
        -------
        Params:
            input_vec (vector): dance layer input vector
        """
        return self.weights.dot(input_vec)

    # ---------------------------------------------------------------------------------------------------------------------
    def _activate(self, act_input_vec):
        """
        Returns activated vector
        -------
        Params:
            act_input_vec (vector): weights matrix multiplied by input vector
        """
        activate = getattr(self, f'_{self.activation}')
        return activate(act_input_vec)

    # ---------------------------------------------------------------------------------------------------------------------
    def _relu(self, vector):
        """
        Its a nonliniarity function, Returns activated vector
        -------
        Params:
            vector (vector)
        """
        return np.maximum(vector, 0)

    # ---------------------------------------------------------------------------------------------------------------------
    def _softmax(self, vector):
        """
        Its a classifier function, Returns activated vector
        -------
        Params:
            vector (vector)
        """
        #return np.exp(vector)/sum(np.exp(vector))
        return softmax(vector, axis=0) 

    # ---------------------------------------------------------------------------------------------------------------------
    def _deriv_active(self, down_loss_deriv):
        """
        Returns loss derivative with activation function input vector.
        -------
        Params:
            down_loss_deriv (vector): loss derivative with current layer vector
        """        
        _deriv_active = getattr(self, f'_deriv_{self.activation}')
        return _deriv_active(down_loss_deriv)
        
    # ---------------------------------------------------------------------------------------------------------------------
    def _deriv_relu(self, down_loss_deriv):
        """
        Returns loss derivative with relu function input vector.
        -------
        Params:
            down_loss_deriv (vector): loss derivative with current layer vector
        """        
        return down_loss_deriv * self._relu(np.sign(self.act_input_vec))
        
    # ---------------------------------------------------------------------------------------------------------------------
    def _deriv_softmax(self, down_loss_deriv):
        """
        Returns loss derivative with sowtmax classifier input vector.
        -------
        Params:
            down_loss_deriv (vector): loss derivative with current layer vector
        """
        # x = down_loss_deriv.dot(self.vector)
        x = (down_loss_deriv*self.vector).sum(axis=0)
        # x = down_loss_deriv.sum(axis=0)
        return self.vector*(down_loss_deriv - x)

    # ---------------------------------------------------------------------------------------------------------------------
    def _deriv_weight(self, deriv_active):
        """
        Returns loss derivative with weights.
        -------
        Params:
            deriv_active (vector): loss derivative with activation function's input vector
        """
        # return np.outer(deriv_active, self.input_vec)
        return deriv_active.dot(self.input_vec.transpose())

    # ---------------------------------------------------------------------------------------------------------------------
    def _deriv_input(self, deriv_active):
        """
        Returns loss derivative with input vector.
        -------
        Params:
            deriv_active (vector): loss derivative with activation function's input vector
        """
        return np.transpose(self.weights)[:-1].dot(deriv_active)

    # ---------------------------------------------------------------------------------------------------------------------
    def _update_weights(self, deriv_weight, learn_rate, optim_rate, regul_rate):
        """
        Update training weights
        -------
        Params:
            deriv_weight (matrix): loss derivative withs weigths
            learn_rate (float): learning rate
            optim_rate (float): optimaizer rate
            regul_rate (float): regularisation rate
        """

        # Adam optimimizer
        optim_adam = self.last_delta_weights * optim_rate

        # L2 regularization
        regul_l2 =  2 * self.weights * regul_rate
        
        #Calculate delta weights
        delta_weights = -learn_rate *(deriv_weight + regul_l2) + optim_adam

        #Save last delta weights
        self.last_delta_weights = delta_weights

        #Update weights
        self.weights += delta_weights

    # ---------------------------------------------------------------------------------------------------------------------
    def _print(self, obj):
        """
        Print logs if allowed 
        -------
        Params:
            obj (object): number of matrix rows
        """
        print(obj) if self._prin else False