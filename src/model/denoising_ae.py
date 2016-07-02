# -*- coding: utf-8 -*-
import sys
import logging
import numpy as np

from sklearn.metrics import accuracy_score


from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder
from util.activation_functions import Activation
from model.mlp import MultilayerPerceptron 

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class DenoisingAutoEncoder(AutoEncoder):
    """
    A denoising autoencoder.
    """

    def __init__(self, train, valid, test, learning_rate=0.1, epochs=30):
        """
         Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        self.performances = []

        self.layers = []
        # First hidden layer
        number_of_1st_hidden_layer = 100

        self.layers.append(LogisticLayer(train.input.shape[1], number_of_1st_hidden_layer, None, activation="tanh", is_classifier_layer=False))

            # Output layer
        self.layers.append(LogisticLayer(number_of_1st_hidden_layer, train.input.shape[1], None, activation="tanh", is_classifier_layer=True))
	
        self.MLP = MultilayerPerceptron(self.training_set, self.validation_set, self.test_set, layers = self.layers, learning_rate=0.05, epochs=30)


    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training DAE epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if False:#verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")
        pass

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        for img in self.training_set.input:
            self.noise = 0.1
            noisy = img + self.noise * np.random.uniform(-1.0,1.0)
            normalized = Activation.tanh(noisy)
            self.MLP._feed_forward(normalized)
            self.MLP._compute_error(normalized[1:])
            self.MLP._update_weights()
        pass


    def _get_weights(self):
        """
        Get the weights (after training)
        """
        return self.MLP._get_input_layer().weights
