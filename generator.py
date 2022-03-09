import librosa
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist #MNIST dataset

class ImageGenerator:
    '''Using autoencoder to generate images'''
    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        self.samples = 0

        self.generated_imgs = None
        self.latent_representation = None

    def generate(self, data, labels, samples, plot = True):
        self.samples = samples
        self.sample_imgs, self.sample_labels = self._select_images(data, labels)
        self.generated_imgs, self.latent_representation = self.autoencoder.reconstruct(self.sample_imgs)
        
        if plot:
            self._plot_images()

    def _select_images(self, data, labels):
        idx = np.random.choice(range(len(data)), self.samples)
        return data[idx], labels[idx]

    def _plot_images(self):
        fig = plt.figure(figsize=(16,3))
        num_img = len(self.generated_imgs)
        for i, (img, gen_img) in enumerate(zip(self.sample_imgs, self.generated_imgs)):
            self._generate_image(img, num_img, i + 1, fig)
            self._generate_image(gen_img, num_img, i + num_img + 1, fig)
        plt.show()

    def _generate_image(self, image, num_img, idx, fig):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_img, idx)
        ax.axis('off')
        ax.imshow(image, cmap = 'gray_r')

    @staticmethod
    def load_mnist():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #Normalize
        x_train = x_train.astype('float32') / 255
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.astype('float32')/255
        x_test = x_test.reshape(x_test.shape + (1,))
        return x_train, y_train, x_test, y_test

class SoundGenerator:
    def __init__(self, autoencoder, hop_length):
        self.autoencoder = autoencoder
        self.hop_length = hop_length

        self.normalizer = None

    def generate(self, spectrogram, min_max):
        generated_spec, latent_rep = self.autoencoder.reconstruct(spectrogram)
        signals = self.convert_spec_to_audio(generated_spec, min_max)
        return signals, latent_rep

    def convert_spec_to_audio(self, spectrograms, min_max):
        signals = []
        for spec, mm_val in zip(spectrograms, min_max):
            log_spectrogram = spec[:, :, 0]
            denormalized_spectrogram = self.normalizer.denormalize(log_spectrogram, mm_val['min'], mm_val['max'])
            ampli = librosa.db_to_amplitude(denormalized_spectrogram)
            signals.appen(librosa.istft(ampli, hop_length=self.hop_length))
        return signals
    