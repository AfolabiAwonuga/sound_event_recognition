import librosa

# Util
import numpy as np 
import json 
import glob
import os



class Loader:
    """
    A signal loader class
    """
    
    def __init__(self, sample_rate, duration, mono):
        """
        Constructs all necessary attributes for the Loader object
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
        
    def load(self, file_path):
        """
        Extracts signal arrays from training data 
        """
        signal = librosa.load(file_path, 
                              sr = self.sample_rate,
                              duration = self.duration, 
                              mono = self.mono)[0]
        return signal
                              
class Padder:
    """
    A signal array passing class 
    """
    
    def __init__(self, duration, sample_rate):
        """
        Constructs all necessary attributes for the Padder object
        """
        self.duration = duration
        self.sample_rate = sample_rate
        
    def pad(self, signal):
        """
        Pads signal array 
        """
        padded_signal = librosa.util.fix_length(signal, 
                                               size = self.duration * self.sample_rate)
        
        return padded_signal                              

class MfccExtractor:
    """
    MFCC feature extractor class 
    """
    
    def __init__(self, sample_rate, n_mfcc, hop_length, n_fft):
        """
        Constructs all necessary attributes for the MfccExtractor object
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract(self, signal):
        """
        Extracts MFCC feature from signals 
        """
        mfcc = np.mean(librosa.feature.mfcc(y = signal,
                                            sr = self.sample_rate,
                                            n_mfcc = self.n_mfcc,
                                            hop_length = self.hop_length,
                                            n_fft = self.n_fft).T, axis = 0).tolist()
        return mfcc        


class Saver:
    """
    A saver class 
    """
    
    def __init__(self, save_dict):
        """
        Constructs all necessary attributes for the Saver object
        """
        self.save_dict = save_dict
       
    def save_feature(self, feature, file_path):
        """
        Saves signal data in a dictionary 
        """
        label = file_path.split('/')[1]
        self.save_dict['label'].append(label)
        self.save_dict['feature'].append(feature)
        self.save_dict['file_path'].append(file_path)        

class PreprocessingPipeline:
    """
    Signal pre-processing class
    """
    
    def __init__(self):
        """
        Constructs all necessary attributes for the PreprocessingPipeline object
        """
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
        
    def process(self, file_dir):
        """
        Implements _process_file method 
        """
        for file_path in glob.glob(file_dir):
            self._process_file(file_path)
        
    def _process_file(self, file_path):
        """
        Loads, Pads(if necessary), and Extract Mfcc feature from signal 
        """
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        self.saver.save_feature(feature, file_path)
        
    def _is_padding_necessary(self, signal):
        """
        Checks if padding a signal array is necessary
        """
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self, signal):
        """
        Applies padding to signal array
        """
        padded_signal = self.padder.pad(signal)
        return padded_signal


if __name__ == '__main__':
    sample_rate = 22050
    duration = 5
    n_mfcc = 13
    hop_length = 512
    n_fft = 2048
    mono = True
    
    
    loader = Loader(sample_rate, duration, mono)
    padder = Padder(duration, sample_rate)
    mfcc_extractor = MfccExtractor(sample_rate, n_mfcc, hop_length, n_fft)
    extractor = mfcc_extractor

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = extractor
    
    classes = ['air conditioner', 'crying baby', 'dog bark', 'door knock',
            'children palying', 'fire', 'alarm']
    
    for i in classes:
        
        save_dict = {
            'label': [],
            'feature' : [],
            'file_path': []
        }
        save = Saver(save_dict)
        preprocessing_pipeline.saver = save
        file_dir = f'combi_set/{i}/*.wav'
        preprocessing_pipeline.process(file_dir)
        with open(os.path.join('mfcc_feature', f'{i}_mfcc_features.json'), 'w') as f:
            json.dump(save_dict, f, indent = 4 )
