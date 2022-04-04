from copy import deepcopy
from tkinter import N

import learn2learn as l2l
import pytorch_lightning as pl

import torch
import torch.nn.functional as F

class GradientLearningBase(pl.LightningModule):

    """Inspired from the learn2lean PL vision example."""
    def __init__(self, train_update_steps=None, test_update_steps=None, loss_func=None, optim_config=None, k_shot=None):
        super().__init__()

        self.train_update_steps = train_update_steps
        self.test_update_steps = test_update_steps
        self.loss_func = loss_func
        self.optim_config = optim_config
        self.k_shot = k_shot

    def training_step(self, batch, batch_idx):
        self.training = True
        output = self.meta_learn(batch)

        self.log(output, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return output['query_error']

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.training = False
        output = self.meta_learn(batch)

        self.log(output)

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

        return {'query_error':query_error, 'query_accuracy':query_accuracy}

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

        return {'query_error':query_error, 'query_accuracy':query_accuracy}

    def configure_optimizers(self):
        # this will be used only in the inner step of the meta training for reptile
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optim_config['inner_learning_rate'])
       
        return optimizer

class ConMAML(GradientLearningBase):
    
    """Based on examples in learn2learn"""

    def __init__(self, model, n_classes_start, n_class_additions, training_steps, loss_func, optim_config, k_shot):
        super().__init__(None, None, None, optim_config, k_shot)

        self.n_classes_start = n_classes_start
        self.n_class_additions = n_class_additions
        self.training_steps = training_steps
        self.loss_func = loss_func
        self.model = l2l.algorithms.MAML(model, lr=optim_config['inner_learning_rate'], first_order=True, allow_nograd=True)
        self.configure_optimizers()

    def return_label_batches(self, batch_labels):
        unique_classes = torch.unique(batch_labels, return_inverse=False)

        classes = [unique_classes[:self.n_classes_start]]
        for i in range((len(unique_classes)-self.n_classes_start)//self.n_class_additions):
            classes.append(unique_classes[self.n_classes_start+self.n_class_additions*(i):self.n_classes_start+self.n_class_additions*(i+1)])
        
        return classes

    @staticmethod
    def return_indexes(batch_labels, classes):
        batch_indexes = []
        for i in range(len(classes)):
            batch_indexes.append(torch.where(batch_labels==classes[i])[0])
        return torch.concat(batch_indexes)

    @staticmethod
    def return_random_labels(batch_labels):
        random_labels = batch_labels[torch.randperm(batch_labels.size(0))]
        converter = {i:random_labels[i] for i in range(random_labels.size(0))}

        for i in range(len(batch_labels)):
            batch_labels[i] = converter[int(batch_labels[i])]

        return batch_labels

    def return_adaption_and_query(self, labels):
        indexes = ConMAML.return_indexes(labels, torch.unique(labels, return_inverse=False))
        first_occurence = indexes[::self.k_shot]
        other_occurences = torch.tensor([i for i in range(len(indexes)) if i%self.k_shot!=0])
        return first_occurence, other_occurences

    def meta_learn(self, batch):
        # used to measure metrics
        logging = {}

        # to accumulate tasks change accumaluate gradients of trainer
        _inputs, labels = batch
        labels = ConMAML.return_random_labels(labels) # ensuring words are not always assigned as first class

        # initiliase model that will be trained
        learner = self.model.clone()
        learner.train()

        # create new continual learning tasks
        class_batches = self.return_label_batches(labels)

        # final measure of accuracy
        query_inputs, query_labels = [], []

        # train on first iteration of classes
        iteration_indexes = ConMAML.return_indexes(labels, class_batches[0])
        iteration_inputs, iteration_labels = _inputs[iteration_indexes], labels[iteration_indexes]
        iteration_support_input, iteration_support_labels, iteration_query_input, iteration_query_labels = self.split_batch(iteration_inputs, iteration_labels)
        query_inputs.append(iteration_query_input)
        query_labels.append(iteration_query_labels)

        # train initial model with intial classes
        for step in range(self.training_steps):
            output = learner(iteration_support_input)
            output['labels'] = iteration_support_labels
            support_error = self.loss_func(output)
            learner.adapt(support_error) 
        logging['step_0_inner_accuracy'] = self.calculate_accuracy(output)

        # train inner loops continually learn models
        for i in range(len(class_batches[1:])):
            iteration_indexes = self.return_indexes(labels, class_batches[i])
            iteration_inputs, iteration_labels = _inputs[iteration_indexes], labels[iteration_indexes]
            iteration_support_input, iteration_support_labels, iteration_query_input, iteration_query_labels = self.split_batch(iteration_inputs, iteration_labels)
            query_inputs.append(iteration_query_input)
            query_labels.append(iteration_query_labels)

            # train additional classes
            for step in range(self.training_steps):
                output = learner(iteration_support_input)
                output['labels'] = iteration_support_labels
                support_error = self.loss_func(output)
                learner.adapt(support_error) 
            logging[f'step_{i+1}_inner_accuracy'] = self.calculate_accuracy(output)

        # train final model with all classes with ONE example
        query_inputs, query_labels = torch.cat(query_inputs), torch.cat(query_labels)
        train_indexes, test_indexes = self.return_adaption_and_query(query_labels)
        output = learner(query_inputs[train_indexes])
        output['labels'] = query_labels[train_indexes]
        quick_update_error = self.loss_func(output)
        learner.adapt(quick_update_error)
        quick_update_accuracy = self.calculate_accuracy(output)
        logging[f'quik_update_inner_accuracy'] = quick_update_accuracy    

        # measure performance over all classes in history
        output = learner(query_inputs[test_indexes])
        output['labels'] = query_labels[test_indexes]
        query_error = self.loss_func(output)
        query_accuracy = self.calculate_accuracy(output)

        return {'query_error':query_error, 'query_accuracy':query_accuracy}