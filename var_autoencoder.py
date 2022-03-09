import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda

tf.compat.v1.disable_eager_execution()

class VarAutoencoder:
    '''
    This module is for implementing Variational Autoencoder
    The concept is to join two mirrored NN.
    The encoder and decoder
    '''
    def __init__(self,
            input_shape,
            filters,
            kernels,
            strides,
            latent_space_dim):
        #Initial atributes
        self.input_shape = input_shape
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000000

        #Models
        self.encoder = None
        self.decoder = None
        self.model = None

        #Private attributes
        self._num_conv_layers = len(filters)
        self._enc_shape = None #Shape of encoder's last layer
        self._model_input = None

        self._build() 

    #Builds whole setup
    def _build(self):
        self._build_coder('encoder')
        self._build_coder('decoder')
        self._build_autoencoder()

    '''BUILD CODERS'''
    def _build_coder(self, coder):
        _input = self._add_input(coder)
        if coder == 'encoder': #Build encoder
            self._model_input = _input
            conv_layers = self._add_layers(_input, 'conv')
            bottleneck = self._add_bottleneck(conv_layers)
            self.encoder = Model(_input, bottleneck, name = coder)
        elif coder == "decoder":  #Build decoder
            dense_layer = self._add_layers(_input, 'dense')
            reshaped_layer = self._add_layers(dense_layer, 'reshape')
            conv_transpose_layers = self._add_layers(reshaped_layer, 'transpose')
            decoder_output = self._add_decoder_output(conv_transpose_layers)
            self.decoder = Model(_input, decoder_output, name = coder)
        else:
            print('Error: Coder is not recognize!')

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')

    '''BUILD CODERS SUBMETHODS'''
    def _add_input(self, coder):
        if coder == 'encoder':
            return Input(shape = self.input_shape, name="enc_input")
        else:
            return Input(shape = self.latent_space_dim, name='dec_input')

    def _add_layers(self, _input, layer_type):
        if layer_type == 'conv':
            for layer_idx in range(self._num_conv_layers):
                _input = self._add_layer(layer_idx, _input, layer_type)
            return _input
        elif layer_type == 'dense':
            num_neurons = np.prod(self._enc_shape)
            dense_layer = Dense(num_neurons, name='dec_dense')(_input)
            return dense_layer
        elif layer_type == 'reshape':
            reshaped_layer = Reshape(self._enc_shape)(_input)
            return reshaped_layer
        elif layer_type == 'transpose':
            #Add convolutional transpose blocks
            #Loop through all conv layer in reverse.
            #Stop in 1st layer
            for layer_index in reversed(range(1, self._num_conv_layers)):
                _input = self._add_layer(layer_index, _input, layer_type)
            return _input
        else:
            print('Error: Layer is not recognized!')
            return None

    def _add_layer(self, layer_idx, layer, layer_type):
        if layer_type == 'conv':
            layer_num = layer_idx + 1
            add_layer = Conv2D(
                filters = self.filters[layer_idx],
                kernel_size = self.kernels[layer_idx],
                strides = self.strides[layer_idx],
                padding = 'same',
                name = f'enc_conv_layer_{layer_num}'
            )
        elif layer_type == 'transpose':
            layer_num = self._num_conv_layers - layer_idx
            add_layer = Conv2DTranspose(
                filters = self.filters[layer_idx],
                kernel_size = self.kernels[layer_idx],
                strides = self.strides[layer_idx],
                padding = 'same',
                name = f'dec_conv_transpose_layer_{layer_num}'
            )
        else:
            return None
        layer = ReLU(name=f'{layer_type}_relu_{layer_num}')(add_layer(layer))
        return BatchNormalization(name=f'{layer_type}_batchnormalization_{layer_num}')(layer)


    def _add_bottleneck(self, conv_layers):
        #Flatten data and add bottleneck
        #Apply Gaussian sampling z = u + Ee
        self._enc_shape = K.int_shape(conv_layers)[1:]
        conv_layers = Flatten()(conv_layers)
        self.mu = Dense(self.latent_space_dim, name='mu')(conv_layers)
        self.log_var = Dense(self.latent_space_dim, name='log_var')(conv_layers)
        
        #Sample a point from normal distribution
        def sample_point_normal_distribution(args):
            mu, log_var = args
            epsilon = K.random_normal(shape = K.shape(self.mu), mean = 0., stddev =1.)
            sample_point = mu + K.exp(log_var/2)*epsilon
            return sample_point
        return Lambda(sample_point_normal_distribution, name='enc_output')([self.mu, self.log_var])

    def _add_decoder_output(self, layer):
        conv_transpose_layer = Conv2DTranspose(
            filters = 1,
            kernel_size = self.kernels[0],
            strides = self.strides[0],
            padding = 'same',
            name = f'dec_conv_transpose_layer_{self._num_conv_layers}'
        )
        return Activation('sigmoid', name='sigmoid_layer')(conv_transpose_layer(layer))

    #Common functions
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate = learning_rate)
        self.model.compile(optimizer = optimizer, loss = self._calculate_combined_loss)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train , batch_size = batch_size, epochs = num_epochs, shuffle = True )
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self._save_parameters(path)
        self.model.save_weights(os.path.join(path, 'weights.h5'))
        
    
    def _save_parameters(self, path):
        parameters = [
            self.input_shape,
            self.filters,
            self.kernels,
            self.strides,
            self.latent_space_dim
        ]
        with open(os.path.join(path, 'parameters.pkl'), "wb") as f:
            pickle.dump(parameters, f)

    def load_weights(self, path):
        self.model.load_weights(path)

    @classmethod
    def load(cls, path='.'):
        with open(os.path.join(path, 'parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)
        autoencoder = VarAutoencoder(*parameters)
        autoencoder.load_weights(os.path.join(path, 'weights.h5'))
        return autoencoder

    def _calculate_combined_loss(self, y_targ, y_pred):
        reconstruction_loss = self._calculate_reconstruction_loss(y_targ, y_pred)
        kl_loss = self._calculate_kl_loss(y_targ, y_pred)
        # cl = W*rl + kl
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_targ, y_pred):
        error = y_targ - y_pred
        reconstruction_loss = K.mean(K.square(error), axis=[1,2,3]) #mean square error
        return reconstruction_loss

    def _calculate_kl_loss(self, y_targ, y_pred):
        #Kullback-Leibler Divergence
        kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
        return kl_loss

    def reconstruct(self, orig):
        latent_representation = self.encoder.predict(orig)
        reconstructed = self.decoder.predict(latent_representation)
        return reconstructed, latent_representation

   
    





    
