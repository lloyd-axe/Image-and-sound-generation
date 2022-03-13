# Image and sound generation using Autoencoders

Autoencoders is a system of Neural Networks (NN) that have the same input an output. Autoencoders cosists of 2 NN, the encoder and decoder.
This project is to demonstrate the implementation of autoencoders when generating image and audio.

For image generation, I used the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset. For audio, I used [FSDD](https://www.kaggle.com/joserzapata/free-spoken-digit-dataset-fsdd).

![var mnist](https://user-images.githubusercontent.com/67902015/158052025-7fccb861-b3d2-45bf-895b-d6ac97cd46f9.PNG)
_Original Input written numbers (1st row) vs Generated Output written numbers (2nd row)_

## How it works
* [var_autoencoder.py](https://github.com/lloyd-axe/Image-and-sound-generation/blob/master/var_autoencoder.py) - Variational autoencoder module, Made out of 2 mirrored convolutional neural networks
* [preprocessing_pipeline.py](https://github.com/lloyd-axe/Image-and-sound-generation/blob/master/preprocessing_pipeline.py) - Module for preprocessing data
* [generator.py](https://github.com/lloyd-axe/Image-and-sound-generation/blob/master/generator.py) - Module for generating data


