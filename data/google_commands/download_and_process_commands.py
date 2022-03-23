import pandas as pd

import torchaudio

# dowloand all data
torchaudio.datasets.SPEECHCOMMANDS('./', download=True, url='speech_commands_v0.02')

# get all files
files=[]
words=[]
import os
rootdir = './SpeechCommands/speech_commands_v0.02/'
for it in os.scandir(rootdir):
    if it.is_dir():
        paths = [os.path.join(it.path.split('/')[-1], f).replace('\\', '/') for f in os.listdir(it.path)]
        files.extend(paths)
        words.extend([f.split('/')[0] for f in paths])

# split dataset into train and test
all_data = pd.DataFrame({'file': files, 'word': words})

# test words are according to the read me only said ones
test_words  = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
train_words = ['forward', 'off','left','go','no', 'down', 'one','zero','five','learn','backward','three','four','nine']
val_words = ['up', 'stop', 'follow', 'right','yes','seven','six','eight','visual','two']

training_data = all_data[all_data['word'].isin(train_words)]
val_data =  all_data[all_data['word'].isin(val_words)]
test_data = all_data[all_data['word'].isin(test_words)]

training_data.to_csv('./google_commands_word_splits_train.csv', index=False)
val_data.to_csv('./google_commands_word_splits_validation.csv', index=False)
test_data.to_csv('./google_commands_word_splits_test.csv', index=False)
