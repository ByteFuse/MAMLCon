import os
from collections import defaultdict
import random
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder

from src.data.processing import raw_audio_to_melspectrogram, raw_audio_to_mfcc


def sample_noise(conversion_cfg):
    #sample random noise
    noise_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_noises/_background_noise_')
    noise_options = os.listdir(noise_root)
    noise_path = os.path.join(noise_root, random.choice(noise_options))

    noise = librosa.load(noise_path, sr=conversion_cfg['sample_rate'])[0]
    random_start_index = random.choice(list(range(len(noise)-16500)))
    noise = noise[random_start_index:random_start_index+16000] #sample 1 second of noise

    if conversion_cfg.name=='mfcc':
        noise = raw_audio_to_mfcc(None, conversion_cfg, noise)
    else:
        noise = raw_audio_to_melspectrogram(None, conversion_cfg, noise)

    return torch.tensor(noise)


def sample_unkown_word(conversion_cfg):
    #sample random word from official test set - ONLY USE FOR TABLE CREATION
    unkown_word = random.choice(["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"])
    unkown_root = os.path.join('../../google_commands/SpeechCommands/speech_commands_v0.02/', unkown_word)
    unkown_options = os.listdir(unkown_root)
    unkown_path = os.path.join(unkown_root, random.choice(unkown_options))

    unkown = librosa.load(unkown_path, sr=conversion_cfg['sample_rate'])[0]

    if conversion_cfg.name=='mfcc':
        unkown = raw_audio_to_mfcc(None, conversion_cfg, unkown)
    else:
        unkown = raw_audio_to_melspectrogram(None, conversion_cfg, unkown)

    return torch.tensor(unkown)

def sample_noise_unknown_words(label, conversion_cfg, k_shot):
    assert label in [-1,-2], 'label must be -1 or -2'
    if label == -2:
        noise = [sample_noise(conversion_cfg) for _ in range(k_shot)]
        return noise
    elif label == -1:
        unkown = [sample_unkown_word(conversion_cfg) for _ in range(k_shot)]
        return unkown
    
class FewShotBatchSampler:
    def __init__(self, dataset_targets, N_way, K_shot, include_query=False, shuffle=True, shuffle_once=False):
        """
        Code is adapted from the tutorials found at the following link:
        https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/12-meta-learning.html
        """
        super().__init__()

        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query

        if self.include_query:
            self.K_shot *= 2
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch

        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        assert self.num_classes >= N_way, 'N way can not be bigger than number of available classes'

        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way

    def __iter__(self):
        # Sample few-shot batches
        start_index = defaultdict(int) # default dict value is now 0
        for it in range(self.iterations):
            class_batch = list(np.random.choice(self.classes, replace=False, size=self.N_way))  # Select N classes for the batch
            index_batch = []

            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                indeces = self.indices_per_class[c][start_index[c] : start_index[c] + self.K_shot]
                index_batch.extend(indeces)
                start_index[c] += self.K_shot
                
                if len(indeces)!=self.K_shot:
                    start_index[c] = 0
                    indeces = self.indices_per_class[c][start_index[c] : start_index[c] + (self.K_shot-len(indeces))]
                    index_batch.extend(indeces)
                    start_index[c] += self.K_shot-len(indeces)

            if self.include_query:  # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations


class SpokenWordTaskBatchSampler:
    def __init__(
        self, 
        dataset_targets, 
        N_way, 
        K_shot, 
        conversion_cfg=None, # must be provided if using noise labels
        noise_labels=None, #-1=unknown, -2=noise
        min_samples=80500,
        max_samples=80500, 
        include_query=False, 
        shuffle=True, 
        constant_size=False, 
        pad_both_sides=False
        ):
        """
        Code is adapted from the brilliant tutorials found at the following link:
        https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/12-meta-learning.html
        """
        super().__init__()

        if noise_labels:
            assert conversion_cfg, 'No conversion method provided for noise labels'

        self.batch_sampler = FewShotBatchSampler(dataset_targets, N_way, K_shot, include_query, shuffle)
        self.task_batch_size = 1
        self.local_batch_size = self.batch_sampler.batch_size
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.constant_size = constant_size
        self.pad_both_sides = pad_both_sides
        self.noise_labels = noise_labels
        self.conversion_cfg = conversion_cfg
        self.k_shot = K_shot
        self.n_way = N_way

    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        for batch_idx, batch in enumerate(self.batch_sampler):
            batch_list.extend(batch)
            if (batch_idx + 1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []

    def __len__(self):
        return len(self.batch_sampler) // self.task_batch_size

    def get_collate_fn(self, item_list):
        min_samples=self.min_samples
        max_samples=self.max_samples
        constant_size=self.constant_size
        pad_both_sides=self.pad_both_sides

        le = LabelEncoder()

        audio = []
        words = []

        for aud, word in item_list:
            if not aud is None:
                words.append(word)
                audio.append(torch.tensor(aud))

        if constant_size:
            max_audio_len = max_samples
        else:
            max_audio_len = max([x.size(-1) for x in audio]) 
            max_audio_len = min_samples if max_audio_len < min_samples else max_audio_len
            max_audio_len = max_samples if max_audio_len > max_samples else max_audio_len

        # sneak in the noise labels here
        if self.noise_labels:
            for label in self.noise_labels:
                noise = sample_noise_unknown_words(label, self.conversion_cfg, self.k_shot*2)
                audio = audio[:self.n_way*self.k_shot] + noise[:self.k_shot] + audio[self.n_way*self.k_shot:] + noise[self.k_shot:]
                words = words[:self.n_way*self.k_shot] + [label] * self.k_shot + words[self.n_way*self.k_shot:] + [label] * self.k_shot

        def pad_audio(x):
            if x.size(-1) > max_audio_len:
                x = x[:,:max_audio_len]
            else:
                if pad_both_sides:
                    pad_lenght = int(max_audio_len-x.size(-1))//2
                    x = F.pad(x, (pad_lenght, pad_lenght+1 if int(max_audio_len-x.size(-1))%2!=0 else pad_lenght), 'constant', 0)
                else:
                    x = F.pad(x, (0, int(max_audio_len-x.size(-1))), 'constant', 0)
            return x

        audio = [pad_audio(x) for x in audio]
        audio = torch.stack(audio, dim=0)
        words =  torch.tensor(le.fit_transform(np.array(words))) # get into numbers makes everything easier
        audio = audio.chunk(self.task_batch_size, dim=0)[0]
        words = words.chunk(self.task_batch_size, dim=0)[0]
       
        return audio, words