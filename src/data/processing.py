import librosa
import numpy as np
import torch
import torch.nn.functional as F

def load_and_process_audio(audio_path, sample_rate=16000, config=None):
    if config:
        sample_rate=config['sample_rate']

    audio, original_rate = librosa.load(audio_path)

    if original_rate!=sample_rate:
        audio = librosa.resample(audio, original_rate, sample_rate)

    audio = torch.tensor(np.expand_dims(audio, axis=0))

    if audio.shape[0]>1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if isinstance(audio, np.ndarray):
        audio = librosa.effects.preemphasis(audio[0], coef=0.97)
    else:
        audio = librosa.effects.preemphasis(audio.numpy()[0], coef=0.97)

    return audio

def raw_audio_to_logspectrogram(audio_path,  config):
    audio = load_and_process_audio(audio_path,  config['sample_rate'])

    spec = librosa.stft(
            audio,
            n_fft=config['n_fft'], 
            win_length=config['win_length'],
            hop_length=config['hop_length'],
        )

    logspec = librosa.amplitude_to_db(spec)
    logspec = np.maximum(logspec, -80)
    logspec = torch.tensor(logspec/80)

    return logspec

def raw_audio_to_melspectrogram(audio_path, config):
    audio = load_and_process_audio(audio_path, config['sample_rate'])

    mel_spec = librosa.feature.melspectrogram(
        audio,
        sr=config['sample_rate'],
        n_mels=config['n_mels'],  
        n_fft=config['n_fft'], 
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        fmin=64,
        norm=1,
        power=1,
    )
    
    logmel = librosa.power_to_db(mel_spec)
    logmel = np.maximum(logmel, -80)
    logmel = torch.tensor(logmel/80)  
    return logmel

def raw_audio_to_mfcc(audio_path, config):
    audio = load_and_process_audio(audio_path, config['sample_rate'])
    
    mfcc = librosa.feature.mfcc(
                audio,
                sr=config['sample_rate'], 
                n_mfcc=config['n_mfcc'], 
                n_mels=config['n_mels'],  
                n_fft=config['n_fft'], 
                win_length=config['win_length'],
                hop_length=config['hop_length'],
                fmin=64, 
                fmax=8000
                )

    if config['use_delta']:
        try:
            mfcc_delta = librosa.feature.delta(mfcc, width=config['width'])
        except:
            # extend audio lenght to be able to calculate dela
            mfcc = torch.tensor(mfcc)
            mfcc = F.pad(mfcc, (0, int(101-mfcc.size(-1))), 'constant', 0).numpy()
            mfcc_delta = librosa.feature.delta(mfcc, width=config['width'])

        if config['use_delta_delta']:
            mfcc_delta_delta = librosa.feature.delta(mfcc, order=2, width=config['width'])
            mfcc = np.hstack([mfcc.T, mfcc_delta.T, mfcc_delta_delta.T])
        else:
             mfcc = np.hstack([mfcc.T, mfcc_delta.T])
             
        mfcc = mfcc.T

    if config['normalize']:
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()
    return mfcc