import os

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd 
import pandarallel
from tqdm import tqdm

from src.data.processing import raw_audio_to_melspectrogram, raw_audio_to_mfcc

def get_all_wav_files(dataset, audio_root):
    if dataset == 'flickr8k':
        wavs = os.listdir(audio_root)
        return wavs
    elif dataset=='google_commands':
        wavs = []
        for it in tqdm(os.scandir(audio_root)):
            if it.is_dir():
                paths = [os.path.join(it.path.split('/')[-1], f).replace('\\', '/') for f in os.listdir(it.path)]
                wavs.extend(paths)
        return wavs


def convert_audio_and_save(wav, conversion_parameters, audio_root):
    # import here to make windows also work
    import os
    import numpy as np
    from src.data.processing import raw_audio_to_melspectrogram, raw_audio_to_mfcc

    conversion_method = conversion_parameters['name']
    save_loc = os.path.join(audio_root, wav.replace('wav', 'npy'))

    if audio_root.split('/')[1] == 'google_commands':
        sub_folder = audio_root+wav.replace('\\','/').split('/')[0]
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)

    if conversion_method == 'melspec':
        converted_wav = raw_audio_to_melspectrogram(os.path.join(audio_root, wav), conversion_parameters)
    elif conversion_method == 'mfcc':
        converted_wav = raw_audio_to_mfcc(os.path.join(audio_root, wav), conversion_parameters)

    np.save(save_loc, converted_wav)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = {'name':'mfcc',
                'sample_rate':16000,
                'n_mfcc':13,
                'n_mels':128,
                'width':9,
                'hop_length':160,
                'win_length':400,
                'n_fft':400,
                'use_delta':True,
                'use_delta_delta':True,
                'normalize':True,
                'input_channels':39,
                'max_samples':101 ,
                'min_samples':95,
                'pad_both_sides':False,
                'constant_size':True,}

    pandarallel.initialize()
    conversion_parameters = config
    dataset = 'google_commands'
    audio_root = 'data/flickr/wavs/' if dataset=='flickr8k' else 'data/google_commands/SpeechCommands/speech_commands_v0.02'
    audio_root_out = f'data/flickr/{conversion_parameters["name"]}/' if dataset=='flickr8k' else f'data/google_commands/SpeechCommands/{conversion_parameters["name"]}/'
    wav_files = get_all_wav_files(dataset, audio_root)

    # start converting all files
    if not os.path.exists(audio_root_out):
        os.mkdir(audio_root_out)

    # quick and easy way to just process in parralell
    wavs = pd.DataFrame({'wavs':wav_files})
    wavs.wavs.parallel_apply(convert_audio_and_save, conversion_parameters=conversion_parameters, audio_root=audio_root, audio_root_out=audio_root_out)

    print("Done")


if __name__ == "__main__":
    main()
