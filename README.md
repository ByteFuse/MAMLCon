# Model Agnostic Meta-Learning for Continual Learning: MAMLCon

This repository contains all of the code required to replicate results for the paper: Mitigating Catastrophic Forgetting for Few-Shot Spoken Word Classification Through Meta-Learning (link to be provided soon). 

We consider the problem of few-shot spoken word classification in a setting where a model is incrementally introduced to new word classes. This would occur in a user-defined keyword system where new words can be added as the system is used. In such a continual learning scenario, a model might start to misclassify earlier words as newer classes are added (i.e. catastrophic forgetting). To address this, we propose an extension to model-agnostic meta learning (MAML): each inner learning loop—where a model “learns how to learn” new classes—ends with a single gradient update using stored templates from all the classes that the model has already seen (one template per class). We compare this method to OML (another extension of MAML) in few-shot isolated word classification experiments on Google Commands and FACC. Our method consistently outperforms OML in experiments where the number of shots and the final number of classes are varied.

# Steps to replicate

Will be supplied in later stage. 
