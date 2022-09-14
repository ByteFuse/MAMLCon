import os
from tqdm import tqdm

import pytorch_lightning as pl
import pandas as pd
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

tqdm.pandas()

import librosa

pl.seed_everything(42)

def load_and_cut_and_save(row):
    original_file = os.path.join('/Users/ruanvdmerwe/Desktop/flickr8k_multimodal_few_shot/data/flickr_audio/wavs',row['original_file'])
    start = row['start']
    duration = row['duration']
    new_loc = os.path.join('./wavs/', row['file'])
    
    if not os.path.exists(new_loc):
        audio, _ = librosa.load(original_file, sr=16000)
        
        start_index = int(start*16000)
        end_index = start_index + int(duration*16000)
        
        word = audio[start_index:end_index]
        
        wavfile.write(new_loc, 16000, word)


# load in meta data 
df = pd.read_csv(
    './flickr_8k.ctm.txt', 
    sep=' ', 
    names=['original_file','something','start','duration', 'word']
)
df = df[['original_file', 'start', 'duration', 'word']]
df.original_file = df.original_file.progress_apply(lambda x: x.replace('.jpg_#', '_')+'.wav')
df['file'] = df['word']+ '_' + +df['start'].progress_apply(lambda x: str(x).replace('.','_')) + '_' + df['original_file']

# get different versions of labels for stemming and lemmatising
wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

def get_stem_lem(words):
    stemed = {}
    for word in words:
        stemed[word] = porter_stemmer.stem(word)
    lematised = {}
    for word in words:
        lematised[word] = wordnet_lemmatizer.lemmatize(word)

    return stemed, lematised

stemed, lematised = get_stem_lem(df.word.unique())
df['stem'] = df.word.apply(lambda word: stemed[word])
df['lem'] = df.word.apply(lambda word: lematised[word])

# split into train test split
df = df[~df.word.isin(['<SPOKEN_NOISE>',"'S", 'A'])]
word_counts = df.groupby('stem').count().reset_index().rename(columns={'original_file':'count'})[['stem', 'word','count']]
word_counts['word_length'] = word_counts.stem.apply(lambda x: len(x))
word_counts = word_counts[(word_counts.word_length>1)&(word_counts['count']>=100)&(word_counts['count']<1000)]
chosen_words = word_counts.sort_values('word_length')

train, test = train_test_split(list(chosen_words.stem.values), test_size=.35, random_state=42)
df = df[df.stem.isin(chosen_words.stem.values)]
df['split'] = 'train'
df.loc[df.stem.isin(test), 'split'] = 'validation'

# save audio
df.progress_apply(load_and_cut_and_save, axis=1)

# save meta data
df[df.split=='train'].to_csv('./flickr8k_word_splits_train.csv', index=False)
df[df.split=='validation'].to_csv('./flickr8k_word_splits_validation.csv', index=False)