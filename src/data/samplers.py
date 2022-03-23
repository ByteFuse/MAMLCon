from collections import defaultdict
import random

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder

class FewShotBatchSampler:
    def __init__(self, dataset_targets, N_way, K_shot, include_query=False, shuffle=True, shuffle_once=False):
        """
        Code is adapted from the brilliant tutorials found at the following link:
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
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_shot

        # Create a list of classes from which we select the N classes per batch
        self.iterations = sum(self.batches_per_class.values()) // self.N_way
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [
                i + p * self.num_classes for i, c in enumerate(self.classes) for p in range(self.batches_per_class[c])
            ]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        random.shuffle(self.class_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int) # default dict value is now 0
        for it in range(self.iterations):
            class_batch = self.class_list[it * self.N_way : (it + 1) * self.N_way]  # Select N classes for the batch
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.indices_per_class[c][start_index[c] : start_index[c] + self.K_shot])
                start_index[c] += self.K_shot
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
        self.batch_sampler = FewShotBatchSampler(dataset_targets, N_way, K_shot, include_query, shuffle)
        self.task_batch_size = 1
        self.local_batch_size = self.batch_sampler.batch_size
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.constant_size = constant_size
        self.pad_both_sides = pad_both_sides

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