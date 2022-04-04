import hydra
from omegaconf import DictConfig
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import torch
import torch.nn as nn
from torchvision.transforms import Compose
import torchaudio

from src.models import WordClassificationAudio2DCnn, WordClassificationAudioCnnPool as WordClassificationAudioCnn, WordClassificationRnn
from src.losses import ClassificationLoss
from src.algorithms import ConMAML
from src.data.datasets import Flickr8kWordClassification, GoogleCommandsWordClassification
from src.data.samplers import SpokenWordTaskBatchSampler
from src.utils import flatten_dict

import warnings
warnings.filterwarnings("ignore")

class WordData(pl.LightningDataModule):
 
    def __init__(self, config):
        super().__init__()
        assert config['dataset'] in ['flickr8k', 'google_commands', 'fluent', 'google_commands_digit', 'google_commands_command'], 'Dataset not supported. Must be either flickr8k, google_commands, fluent'        
        self.cfg = config

    def setup(self, stage=None):
        if self.cfg['dataset'] == 'flickr8k':
            self.train_dataset = Flickr8kWordClassification(
                meta_path='../../../../../../../../data/flickr/flickr8k_word_splits_train.csv',
                audio_root='../../../../../../../../data/flickr/wavs/', 
                conversion_config=self.cfg.conversion_method,
                stemming=self.cfg.stemming, 
                lemmetise=self.cfg.lematise     
            )

            self.valiadation_dataset = Flickr8kWordClassification(
                meta_path='../../../../../../../../data/flickr/flickr8k_word_splits_validation.csv',
                audio_root='../../../../../../../../data/flickr/wavs/', 
                conversion_config=self.cfg.conversion_method,
                stemming=self.cfg.stemming, 
                lemmetise=self.cfg.lematise                
            )
        elif self.cfg['dataset'] == 'google_commands':
            self.train_dataset = GoogleCommandsWordClassification(
                meta_path='../../../../../../../../data/google_commands/google_commands_word_splits_train.csv',
                audio_root='../../../../../../../../data/google_commands/SpeechCommands/speech_commands_v0.02', 
                conversion_config=self.cfg.conversion_method,  
            )

            self.valiadation_dataset = GoogleCommandsWordClassification(
                meta_path='../../../../../../../../data/google_commands/google_commands_word_splits_validation.csv',
                audio_root='../../../../../../../../data/google_commands/SpeechCommands/speech_commands_v0.02', 
                conversion_config=self.cfg.conversion_method,             
            )    
        elif self.cfg['dataset'] == 'google_commands_digit':
            self.train_dataset = GoogleCommandsWordClassification(
                meta_path='../../../../../../../../data/google_commands/google_commands_word_splits_commands.csv',
                audio_root='../../../../../../../../data/google_commands/SpeechCommands/speech_commands_v0.02', 
                conversion_config=self.cfg.conversion_method,  
            )

            self.valiadation_dataset = GoogleCommandsWordClassification(
                meta_path='../../../../../../../../data/google_commands/google_commands_word_splits_digits.csv',
                audio_root='../../../../../../../../data/google_commands/SpeechCommands/speech_commands_v0.02', 
                conversion_config=self.cfg.conversion_method,             
            )         
        elif self.cfg['dataset'] == 'google_commands_commands':
            self.train_dataset = GoogleCommandsWordClassification(
                meta_path='../../../../../../../../data/google_commands/google_commands_word_splits_digits.csv',
                audio_root='../../../../../../../../data/google_commands/SpeechCommands/speech_commands_v0.02', 
                conversion_config=self.cfg.conversion_method,  
            )

            self.valiadation_dataset = GoogleCommandsWordClassification(
                meta_path='../../../../../../../../data/google_commands/google_commands_word_splits_commands.csv',
                audio_root='../../../../../../../../data/google_commands/SpeechCommands/speech_commands_v0.02', 
                conversion_config=self.cfg.conversion_method,             
            )
        elif self.cfg['dataset'] == 'fluent':
            raise NotImplementedError

        train_labels = torch.tensor(self.train_dataset.labels)
        validation_labels = torch.tensor(self.valiadation_dataset.labels)

        if self.cfg.noise_labels == 'noise':
            noise_labels = [-2]
        elif self.cfg.noise_labels == 'unknown':
            noise_labels = [-1]
        elif self.cfg.noise_labels == 'both':
            noise_labels = [-1,-2]
        else:
            noise_labels = None

        self.train_sampler = SpokenWordTaskBatchSampler(
            dataset_targets=train_labels, 
            N_way=self.cfg.n_way, 
            K_shot=self.cfg.k_shot, 
            conversion_cfg=self.cfg.conversion_method,
            noise_labels=noise_labels,
            min_samples=self.cfg.conversion_method.min_samples,
            max_samples=self.cfg.conversion_method.max_samples,
            include_query=True, 
            shuffle=True,
            constant_size = self.cfg.conversion_method.constant_size,
            pad_both_sides=self.cfg.conversion_method.pad_both_sides
        )
        self.valiadation_sampler = SpokenWordTaskBatchSampler(
            dataset_targets=validation_labels, 
            N_way=self.cfg.n_way, 
            K_shot=self.cfg.k_shot,
            conversion_cfg=self.cfg.conversion_method,
            noise_labels=noise_labels,
            min_samples=self.cfg.conversion_method.min_samples,
            max_samples=self.cfg.conversion_method.max_samples,
            include_query=True, 
            shuffle=False,
            constant_size = self.cfg.conversion_method.constant_size,
            pad_both_sides=self.cfg.conversion_method.pad_both_sides
        )
       
    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler, 
            collate_fn=self.train_sampler.get_collate_fn,
            num_workers=16,
            persistent_workers=True,
            pin_memory=True
        )

        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.valiadation_dataset,
            batch_sampler=self.valiadation_sampler, 
            collate_fn=self.valiadation_sampler.get_collate_fn, 
            num_workers=16,
            persistent_workers=True,
            pin_memory=True
        )

        return val_loader


class MetaModel(nn.Module):
    def __init__(self, encoder, embedding_dim, n_classes):
        super().__init__()

        self.encoder = encoder

        classification_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        self.classifiers = nn.ModuleList([classification_layer]*n_classes)

    def forward(self, audio, total_classes_present):
        features = self.encoder(audio)

        layer_logits = []
        for c_layer in range(total_classes_present):
            layer_logits.append(self.classifiers[c_layer](features))

        logits = torch.cat(layer_logits, dim=1)

        return {'logits':logits}

@hydra.main(config_path="config", config_name="config_cl")
def main(cfg: DictConfig):
    pl.utilities.seed.seed_everything(42)

    if cfg.encoder.name == '1d_cnn':	
        encoder = WordClassificationAudioCnn(cfg.embedding_dim, cfg.encoder.hidden_dim, input_channels=cfg.conversion_method.input_channels)
    elif cfg.encoder.name == '2d_cnn':
        encoder = WordClassificationAudio2DCnn(cfg.embedding_dim, cfg.encoder.hidden_dim, input_channels=cfg.conversion_method.input_channels)
    elif cfg.encoder.name == 'rnn':
        encoder = WordClassificationRnn(
            embedding_dim=cfg.embedding_dim, 
            hidden_dim=cfg.encoder.hidden_dim, 
            input_size=cfg.conversion_method.input_channels, 
            n_layers=cfg.encoder.n_layers, 
            learn_states=cfg.encoder.learn_states
          )

    loss_fn = ClassificationLoss()
    model = MetaModel(encoder, cfg.embedding_dim, cfg.n_way)
    data = WordData(cfg)
    data.setup()

    if cfg.method=='maml':
        algorithm = ConMAML(
            model=model, 
            training_steps=cfg.train_update_steps,
            n_classes_start=cfg.n_classes_start,
            n_class_additions=cfg.n_class_additions,
            loss_func=loss_fn,
            optim_config=cfg.optim,
            k_shot=cfg.k_shot,
            quick_adapt=cfg.quick_adapt
        )
    elif cfg.method=='reptile':
        raise NotImplementedError

    wandb.login(key=cfg.secrets.wandb_key)
    wandb_logger = WandbLogger(project='unimodal-isolated-few-shot-continual-learning', config=flatten_dict(cfg))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints', 
        filename='{epoch}-{validation_loss:.2f}', 
        save_top_k=5, 
        monitor='validation_loss',
        save_weights_only=False,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    earlystop_callback = EarlyStopping(monitor='validation_loss', patience=cfg.optim.scheduler_step, mode='min')

    callbacks = [checkpoint_callback, lr_monitor, earlystop_callback] if cfg.method=='maml' else [checkpoint_callback, earlystop_callback]

    trainer = pl.Trainer(
        logger=wandb_logger,    
        log_every_n_steps=2,   
        gpus=None if not torch.cuda.is_available() else -1,
        max_steps=cfg.max_steps,           
        deterministic=False, 
        precision=cfg.precision if cfg.method=='maml' else 32,
        profiler="simple",
        accumulate_grad_batches=cfg.batch_size,
        gradient_clip_val=cfg.optim.gradient_clip_val,
        limit_train_batches = cfg.epoch_n_tasks,
        limit_val_batches = cfg.epoch_n_tasks,
        callbacks=callbacks
    )

    trainer.fit(algorithm, data)
    wandb.finish()

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()