import os
from multiprocessing import Pool

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd 
import swifter
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


def convert_audio_and_save(wav, conversion_parameters, audio_root, audio_root_out):
    conversion_method = conversion_parameters['name']
    save_loc = os.path.join(audio_root_out, wav.replace('wav', 'npy'))

    if audio_root_out.split('/')[1] == 'google_commands':
        sub_folder = audio_root_out+wav.replace('\\','/').split('/')[0]
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)

    if conversion_method == 'melspec':
        converted_wav = raw_audio_to_melspectrogram(os.path.join(audio_root, wav), conversion_parameters)
    elif conversion_method == 'mfcc':
        converted_wav = raw_audio_to_mfcc(os.path.join(audio_root, wav), conversion_parameters)

    np.save(save_loc, converted_wav)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    conversion_parameters = cfg.conversion_method
    dataset = cfg.dataset
    audio_root = 'data/flickr/wavs/' if dataset=='flickr8k' else 'data/google_commands/SpeechCommands/speech_commands_v0.02'
    audio_root_save = f'data/flickr/{conversion_parameters.name}/' if dataset=='flickr8k' else f'data/google_commands/SpeechCommands/{conversion_parameters.name}/'
    wav_files = get_all_wav_files(dataset, audio_root)

    # start converting all files
    if not os.path.exists(audio_root_save):
        os.mkdir(audio_root_save)

    # quick and easy way to just process in parralell
    wavs = pd.DataFrame({'wavs':wav_files})
    wavs.wavs.swifter.progress_bar(enable=True).apply(
        lambda wav: convert_audio_and_save(wav, conversion_parameters, audio_root, audio_root_save)
    )

    print("Done")


if __name__ == "__main__":
    main()







