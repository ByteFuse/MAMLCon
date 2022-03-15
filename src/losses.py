import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, output):
        logits, labels = output['logits'], output['labels']
        loss  = self.loss_fn(logits, labels.long().to(logits.device))

        return loss   
