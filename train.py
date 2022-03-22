import hydra
import learn2learn as l2l
from omegaconf import DictConfig
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import torch
import torch.nn as nn
from torchvision.transforms import Compose
import torchaudio

from src.models import WordClassificationAudio2DCnn, WordClassificationAudioCnn, WordClassificationRnn
from src.losses import ClassificationLoss
from src.algorithms import VanillaMAML, Reptile
from src.data.datasets import Flickr8kWordClassification
from src.utils import flatten_dict


class WordData(pl.LightningDataModule):
 
    def __init__(self, config):
        super().__init__()
        assert config['dataset'] in ['flickr8k', 'google_commands', 'fluent'], 'Dataset not supported. Must be either flickr8k, google_commands, fluent'        
        self.cfg = config

    def setup(self, stage=None):
        if self.cfg['dataset'] == 'flickr8k':
            train_dataset = Flickr8kWordClassification(
                meta_path='../../../../../data/flickr/flickr8k_word_splits_train.csv',
                audio_root='../../../../../data/flickr/wavs/', 
                conversion_config=self.cfg.conversion_method,
                stemming=self.cfg.stemming, 
                lemmetise=self.cfg.lematise     
            )

            val_dataset = Flickr8kWordClassification(
                meta_path='./data/flickr/flickr8k_word_splits_validation.csv',
                audio_root='./data/flickr/wavs/', 
                conversion_config=self.cfg.conversion_method,
                stemming=self.cfg.stemming, 
                lemmetise=self.cfg.lematise                
            )
        else:
            pass #TODO: Add in other datasets

        train_dataset = l2l.data.MetaDataset(train_dataset, indices_to_labels=train_dataset.indices_to_labels, labels_to_indices=train_dataset.labels_to_indices)
        val_dataset = l2l.data.MetaDataset(val_dataset, indices_to_labels=val_dataset.indices_to_labels, labels_to_indices=val_dataset.labels_to_indices)

        train_transforms = [
            l2l.data.transforms.NWays(train_dataset, n=self.cfg['n_way']), 
            l2l.data.transforms.KShots(train_dataset,k=self.cfg['k_shot'], replacement=False), 
            l2l.data.transforms.LoadData(train_dataset)
        ]
        val_transforms = [
            l2l.data.transforms.NWays(val_dataset, n=self.cfg['n_way']), 
            l2l.data.transforms.KShots(val_dataset,k=self.cfg['k_shot'], replacement=False), 
            l2l.data.transforms.LoadData(val_dataset)
        ]

        self.train_dataset = l2l.data.TaskDataset(train_dataset, train_transforms, num_tasks=self.cfg['epoch_n_tasks'])
        self.valiadation_dataset = l2l.data.TaskDataset(val_dataset, val_transforms, num_tasks=self.cfg['epoch_n_tasks'])

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True
        )

        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.valiadation_dataset,  
            num_workers=8,
            persistent_workers=True,
            pin_memory=True
        )

        return val_loader


class MetaModal(nn.Module):
    def __init__(self, encoder, embedding_dim, n_classes, return_features=False):
        super().__init__()

        self.encoder = encoder
        self.classification_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embedding_dim, n_classes)
        )

        self.return_features = return_features

    def forward(self, audio):
        features = self.encoder(audio)
        logits = self.classification_layer(features)

        if self.return_features:
            return {"features":features, "logits":logits}

        return {'logits':logits}

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.utilities.seed.seed_everything(42)

    if cfg.augment == 'hard':
        audio_augmentation = Compose([
            torchaudio.transforms.TimeMasking(time_mask_param=5),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=2),
        ])
    elif cfg.augment == 'easy':
        audio_augmentation = Compose([
            torchaudio.transforms.TimeMasking(time_mask_param=5),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=1)
        ])
    elif cfg.augment == 'none':
        audio_augmentation = None

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
    model = MetaModal(encoder, cfg.embedding_dim, cfg.n_way, return_features=cfg.return_features)
    data = WordData(cfg)
    data.setup()

    if cfg.method=='maml':
        algorithm = VanillaMAML(
            model=model, 
            train_update_steps=cfg.optim.inner_steps,
            test_update_steps=cfg.optim.val_inner_steps,
            loss_func=loss_fn,
            optim_config=cfg.optim,
            k_shot=cfg.k_shot,
            first_order=cfg.first_order,
            augmentation=audio_augmentation
        )
    elif cfg.method=='reptile':
        algorithm = Reptile(
            model=model, 
            train_update_steps=cfg.optim.inner_steps,
            test_update_steps=cfg.optim.val_inner_steps,
            loss_func=loss_fn,
            optim_config=cfg.optim,
            k_shot=cfg.k_shot,
            augmentation=audio_augmentation
        )

    wandb.login(key=cfg.secrets.wandb_key)
    wandb_logger = WandbLogger(project='flickr8k-few-shot-learning', config=flatten_dict(cfg))
    
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

    trainer = pl.Trainer(
        logger=wandb_logger,    
        log_every_n_steps=2,   
        gpus=None if not torch.cuda.is_available() else -1,
        max_epochs=cfg.max_epochs,           
        deterministic=True, 
        precision=cfg.precision,
        profiler="simple",
        accumulate_grad_batches=cfg.batch_size,
        gradient_clip_val=cfg.optim.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor, earlystop_callback]
    )

    trainer.fit(algorithm, data)
    wandb.finish()