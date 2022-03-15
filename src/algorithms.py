from logging import raiseExceptions
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import pytorch_lightning as pl

import learn2learn as l2l

class GradientLearningBase(pl.LightningModule):

    """Inspired from the learn2lean PL vision example."""
    def __init__(self, model, train_update_steps, test_update_steps, loss_func, optim_config, audio_aug=None, image_aug=None, first_order=True):
        super().__init__()

        self.train_update_steps = train_update_steps
        self.test_update_steps = test_update_steps
        self.loss_func = loss_func
        self.optim_config = optim_config
        self.audio_aug = audio_aug
        self.image_aug = image_aug

        self.image_val_aug = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        ])

        self.model = l2l.algorithms.MAML(model, lr=self.adaptation_lr, first_order=first_order, allow_nograd=True)

    def training_step(self, batch, batch_idx):
        query_loss, query_accuracy = self.meta_learn(batch, self.train_update_steps)

        self.log(f"train_accuracy_3",query_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"train_loss", query_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return query_loss

    def validation_step(self, batch, batch_idx):
        query_loss, query_accuracy = self.meta_learn(batch, self.test_update_steps)

        self.log(f"validation_accuracy",query_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"validation_loss", query_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_config['outer_learning_rate'])

        if self.optim_config['scheduler']:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.optim_config['scheduler_step'],
                gamma=self.optim_config['scheduler_decay'],
            )
            return [optimizer], [lr_scheduler]
        
        return optimizer

    def meta_learn(self):
        raise NotImplementedError('User must impliment this method')


    def split_batch_into_support_query(self, _input, label):
        
        support_input, query_input = _input.chunk(2, dim=0)
        support_labels, query_labels = label.chunk(2, dim=0)
        return support_input, query_input, support_labels, query_labels


    def calculate_accuracy(output):
        pass


class VanillaMAML(GradientLearningBase):
    
    """Based on examples in learn2learn"""

    def __init__(self, train_update_steps, test_update_steps, loss_func, optim_config, audio_aug=None, image_aug=None):
        super().__init__(train_update_steps, test_update_steps, loss_func, optim_config, audio_aug, image_aug)

    def meta_learn(self, batch):
        # to accumulate tasks change accumaluate gradients of trainer
        _inputs, labels = batch
        support_input, query_input, support_labels, query_labels = self.split_batch_into_support_query(_inputs, labels)
        
        learner = self.model.clone()
        learner.train()
            
        # fast train
        for step in range(self.train_update_steps):
            output = learner(support_input)
            output['labels'] = support_labels
            support_error = self.loss_func(output)
            learner.adapt(support_error) 

        output = learner(query_input)
        output['labels'] = query_labels
        query_error = self.loss_func(output)
        query_accuracy = self.calculate_accuracy(output)

        return query_error, query_accuracy
        

class VanillaMAML(GradientLearningBase):
    
    """Based on examples in learn2learn"""

    def __init__(self, train_update_steps, test_update_steps, loss_func, optim_config, audio_aug=None, image_aug=None):
        super().__init__(train_update_steps, test_update_steps, loss_func, optim_config, audio_aug, image_aug)

    def meta_learn(self, batch):
        # to accumulate tasks change accumaluate gradients of trainer
        _inputs, labels = batch
        support_input, query_input, support_labels, query_labels = self.split_batch_into_support_query(_inputs, labels)
        
        learner = self.model.clone()
        learner.train()
            
        # fast train
        for step in range(self.train_update_steps):
            output = learner(support_input)
            output['labels'] = support_labels
            support_error = self.loss_func(output)
            learner.adapt(support_error) 

        output = learner(query_input)
        output['labels'] = query_labels
        query_error = self.loss_func(output)
        query_accuracy = self.calculate_accuracy(output)

        return query_error, query_accuracy


class Reptile(GradientLearningBase):
    
    """Based on examples in learn2learn"""

    def __init__(self, train_update_steps, test_update_steps, loss_func, optim_config, audio_aug=None, image_aug=None):
        super().__init__(train_update_steps, test_update_steps, loss_func, optim_config, audio_aug, image_aug)

    def meta_learn(self, batch):
        pass



            







