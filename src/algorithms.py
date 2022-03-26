from copy import deepcopy

import learn2learn as l2l
import pytorch_lightning as pl

import torch
import torch.nn.functional as F

class GradientLearningBase(pl.LightningModule):

    """Inspired from the learn2lean PL vision example."""
    def __init__(self, train_update_steps, test_update_steps, loss_func, optim_config, k_shot):
        super().__init__()

        self.train_update_steps = train_update_steps
        self.test_update_steps = test_update_steps
        self.loss_func = loss_func
        self.optim_config = optim_config
        self.k_shot = k_shot

    def training_step(self, batch, batch_idx):
        self.training = True
        query_loss, query_accuracy = self.meta_learn(batch)

        self.log(f"train_accuracy",query_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"train_loss", query_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return query_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.training = False
        query_loss, query_accuracy = self.meta_learn(batch)

        self.log(f"validation_accuracy",query_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"validation_loss", query_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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

    def split_batch(self, _input, labels):
        support_input, query_input = _input.chunk(2, dim=0)
        support_labels, query_labels = labels.chunk(2, dim=0)
        return support_input, support_labels, query_input, query_labels

    def meta_learn(self):
        raise NotImplementedError('User must impliment this method')

    def calculate_accuracy(self, output):
        logits, labels = output['logits'], output['labels']
        predicted_probs = F.softmax(logits, dim=1).detach().cpu()
        labels = labels.long().detach().cpu()
        predicted_label = torch.argmax(predicted_probs, dim=1)
        accuracy = (predicted_label==labels).sum().item()/len(labels)

        return accuracy

class VanillaMAML(GradientLearningBase):
    
    """Based on examples in learn2learn"""

    def __init__(self, model, train_update_steps, test_update_steps, loss_func, optim_config, k_shot, first_order=True, augmentation=None):
        super().__init__(train_update_steps, test_update_steps, loss_func, optim_config, k_shot)
        self.model = l2l.algorithms.MAML(model, lr=optim_config['inner_learning_rate'], first_order=first_order, allow_nograd=True)
        self.augmentation = augmentation
        self.configure_optimizers()


    def meta_learn(self, batch):
        # to accumulate tasks change accumaluate gradients of trainer
        _inputs, labels = batch
        support_input, support_labels, query_input, query_labels = self.split_batch(_inputs, labels)

        learner = self.model.clone()
        learner.train()
            
        # fast train
        for step in range(self.train_update_steps):
            if self.training:
                output = learner(self.augmentation(support_input)) if self.augmentation else learner(support_input)
            else:
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
    
    """Based on code from orginal blog https://openai.com/blog/reptile/"""
    def __init__(self, model, train_update_steps, test_update_steps, loss_func, optim_config, k_shot,  augmentation=None):
        super().__init__(train_update_steps, test_update_steps, loss_func, optim_config, k_shot)
        self.model = model
        self.automatic_optimization = False
        self.outer_steps = 0
        self.initial_lr = optim_config['outer_learning_rate']
        self.augmentation = augmentation
        self.configure_optimizers()

    def meta_learn(self, batch):
        # to accumulate tasks change accumaluate gradients of trainer
        _inputs, labels = batch
        support_input, support_labels, query_input, query_labels = self.split_batch(_inputs, labels)
        
        old_weights = deepcopy(self.model.state_dict()) 
        opt = self.optimizers()

        # fast train
        for step in range(self.train_update_steps):
            opt.zero_grad()
            if self.training:
                output = self.model(self.augmentation(support_input)) if self.augmentation else self.model(support_input)
            else:
                output = self.model(support_input)
            output['labels'] = support_labels
            support_error = self.loss_func(output)
            support_error.backward()
            opt.step() 

        output = self.model(query_input)
        output['labels'] = query_labels
        query_error = self.loss_func(output)
        query_accuracy = self.calculate_accuracy(output)

        if self.training:
            new_weights = self.model.state_dict()
            current_lr = self.initial_lr * (1 - self.outer_steps / self.trainer.max_steps) # linear schedule
            self.log('learning_rate',current_lr, on_step=True, on_epoch=True, prog_bar=False, logger=False)
            self.model.load_state_dict({name : old_weights[name] + (new_weights[name] - old_weights[name]) * current_lr for name in old_weights})
            self.outer_steps += 1
        else:
            self.model.load_state_dict(old_weights)

        return query_error, query_accuracy

    def configure_optimizers(self):
        # this will be used only in the inner step of the meta training for reptile
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_config['inner_learning_rate'])
       
        return optimizer



