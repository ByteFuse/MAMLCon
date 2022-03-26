import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn.functional as F

from src.data.processing import (
    load_and_process_audio,
    raw_audio_to_logspectrogram,
    raw_audio_to_melspectrogram,
    raw_audio_to_mfcc
)

class Flickr8kWordClassification(torch.utils.data.Dataset):
    def __init__(self, meta_path, audio_root, conversion_config, stemming=False, lemmetise=False):
        assert conversion_config['name'] in ['melspec', 'spec', 'raw', 'mfcc'], 'audio_conversion_method must be one of: melspec, mfcc, spec, raw'
        
        metadata = pd.read_csv(meta_path)
        self.audio_files = [os.path.join(audio_root, audio) for audio in tqdm(metadata.file, desc='Loading audio')]

        if stemming:
            self.words = metadata.stem.values
        elif lemmetise:
            self.words = metadata.lem.values
        else:
            self.words = metadata.word.values

        self.labels = LabelEncoder().fit_transform(self.words)
        self.indices_to_labels = {i: label for i, label in enumerate(self.labels)}
        self.create_labels_to_indices()
        
        self.conversion_config = conversion_config
        self.conversions = {
            'melspec': raw_audio_to_melspectrogram,
            'spec': raw_audio_to_logspectrogram,
            'mfcc': raw_audio_to_mfcc,
            'raw': load_and_process_audio,
        }

    def create_labels_to_indices(self):
        unique_labels = np.unique(self.labels)
        l2i = {label:[] for label in unique_labels}

        for i, label in enumerate(self.labels):
            l2i[label].append(i)
        
        self.labels_to_indices = l2i

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]    
        audio = self.conversions[self.conversion_config['name']](audio_path, config=self.conversion_config)
        label = self.labels[idx]

        return audio, label

    def __len__(self):
        return len(self.audio_files)

class GoogleCommandsWordClassification(Flickr8kWordClassification):
    def __init__(self, meta_path, audio_root, conversion_config):
        assert conversion_config['name'] in ['melspec', 'spec', 'raw', 'mfcc'], 'audio_conversion_method must be one of: melspec, mfcc, spec, raw'
        
        metadata = pd.read_csv(meta_path)
        self.audio_files = [os.path.join(audio_root, audio) for audio in tqdm(metadata.file, desc='Loading audio')]
        self.labels = LabelEncoder().fit_transform(metadata.word.values)
        self.indices_to_labels = {i: label for i, label in enumerate(self.labels)}
        self.create_labels_to_indices()
        
        self.conversion_config = conversion_config
        self.conversions = {
            'melspec': raw_audio_to_melspectrogram,
            'spec': raw_audio_to_logspectrogram,
            'mfcc': raw_audio_to_mfcc,
            'raw': load_and_process_audio,
        }