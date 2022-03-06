import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K

import numpy as np

class Autoencoder:
    '''
    Deep Convolutional autoencoder architecture 
    with mirrored encoder and decoder components.
    '''

    def __init__(self,
            input_shape,
            conv_filters,
            conv_kernels,
            conv_strides,
            latent_space_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        #Private attributes
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()
    
    def summary(self):
        # self.encoder.summary()
        # self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate)
        nse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=nse_loss)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train , batch_size = batch_size, epochs = num_epochs, shuffle = True )

    def save(self, path):
        self._create_dir(path)
        self._save_parameters(path)
        self._save_weights(path)

    def _create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _save_parameters(self, path):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        with open(os.path.join(path, 'parameters.pkl'), "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, path):
        self.model.save_weights(os.path.join(path, 'weights.h5'))

    @classmethod
    def load(cls, path='.'):
        with open(os.path.join(path, 'parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        autoencoder.load_weights(os.path.join(path, 'weights.h5'))
        return autoencoder

    def load_weights(self, path):
        self.model.load_weights(path)
    
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    #BUILDING ENCODER -------------------------------------------
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):    
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        #Creates all convolutional blocks in encoder
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        #Creates a convolutional blocks in encoder. Conv 2d, Relu and batch normalization
        layer_num = layer_index + 1
        conv_layer = Conv2D(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = 'same',
            name=f'encoder_conv_layer_{layer_num}'
        )

        x = conv_layer(x)
        x = ReLU(name=f'encoder_relu_{layer_num}')(x)
        x = BatchNormalization(name=f'batch_normalization_{layer_num}')(x)
        return x

    def _add_bottleneck(self, x):
        #Flatten data and add bottleneck
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name=f'encoder_output')(x)
        return x

    #BUILDING DECODER -------------------------------------------
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshaped_layer = self._add_reshaped_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshaped_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_decoder_input(self):
        return Input(shape = self.latent_space_dim, name='decoder_input')

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) 
        dense_layer = Dense(num_neurons, name='decoder_dense')(decoder_input)
        return dense_layer

    def _add_reshaped_layer(self, dense_layer):
        reshaped_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshaped_layer

    def _add_conv_transpose_layers(self, x):
        #Add convolutional transpose blocks
        #Loop through all conv layer in reverse. Stop in 1st layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = 'same',
            name = f'decoder_conv_transpose_layer_{layer_num}'
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f'decoder_relu_{layer_num}')(x)
        x = BatchNormalization(name=f'decoder_batchnormalization_{layer_num}')(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters = 1,
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = 'same',
            name = f'decoder_conv_transpose_layer_{self._num_conv_layers}'
        )
        x = conv_transpose_layer(x)
        output_layer = Activation('sigmoid', name='sigmoid_layer')(x)
        return output_layer

    #BUILDING AUTOENCODER -------------------------------------------
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')
    
    def reconstruct(self, images):
        latent_representation = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representation)
        return reconstructed_images, latent_representation