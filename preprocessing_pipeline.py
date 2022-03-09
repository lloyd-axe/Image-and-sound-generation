import os
import pickle
import librosa
import numpy as np

'''
This module is created to aid the user when preprocessing
data from any DataSet to use in the auto encoder

Components available in this module are:
1. Loader
2. Saver
3. Padder
4. Normalizer
5. Extractor
'''

class PreprocessingPipeline:
    '''
    Performs all the task necessary for preprocessing any data
    for the purpose of using the preprocessed data in the my 
    variational autoencoder
    '''
    def __init__(self):
        #Major components
        self._loader = None
        self.saver = None
        self.padder = None
        self.extractor = None
        self.normalizer = None

        self.min_max = {}
        self._samples_num = None

    def process(self, data_path):
        #Process each file found in the directory
        for root, _, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f'processed {file}...')
        self.saver.save_min_max(self.min_max) #Save original min-max values

    def _process_file(self, file_path):
        signal = self.loader.load_sound(file_path)
        if len(signal) < self._samples_num:
            pad = self._samples_num - len(signal)
            signal = self.padder._add_padding(signal, 'right', pad)
        features = self.extractor.extract_log_spec(signal)
        normalized_features = self.normalizer.normalize(features)
        save_path = self.saver.save_features(normalized_features, file_path)
        
        self.min_max[save_path] = {
            'min' : features.min(),
            'max' : features.max()
        }

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._samples_num = int(loader.sr * loader.duration)

class Loader:
    '''Loads from directory'''
    def __init__(self, sr = 22050, duration = 3000, mono = True):
        self.sr = sr
        self.duration = duration
        self.mono = mono

    def load_sound(self, file_path):
        return librosa.load(file_path, 
            sr = self.sr, 
            duration = self.duration, 
            mono = self.mono)[0]

    #Loader for other data types will be added in future updates

class Padder:
    '''Applies padding to the data'''
    def __init__(self, mode = 'constant'):
        self.mode = mode

    def _add_padding(self, data, position, pad_num):
        pad = (0, pad_num) if position == 'right' else (pad_num, 0)
        return np.pad(data, pad, mode = self.mode)

class Extractor:
    '''Extracts special features from data'''

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract_log_spec(self, signal):
        stft = librosa.stft(signal, 
            n_fft = self.frame_size, 
            hop_length = self.hop_length)[:-1] #To even the length of the data
        return librosa.amplitude_to_db(np.abs(stft))

class MinMaxNormalizer:
    '''Normalize any data to values between {min} and {max}'''
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def normalize(self, data):
        return ((data - data.min()) / (data.max() - data.min())) + (self.max - self.min) + self.min
    
    def denormalize(self, normalized_data, orig_min, orig_max):
        return ((normalized_data - self.min) / (self.max - self.min)) + (orig_max - orig_min) + orig_min

class Saver:
    '''Saving function ment to save features and other values'''
    def __init__(self, features_path, min_max_path):
        self.features_path = features_path
        self.min_max_path = min_max_path

    def save_features(self, features, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.features_path, file_name + '.npy')
        np.save(save_path, features)
        return save_path

    def save_min_max(self, min_max):
        save_path = os.path.join(self.min_max_path, 'min_max.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(min_max, f)
    
        

    




        