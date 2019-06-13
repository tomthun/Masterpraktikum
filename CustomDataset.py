# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:06:12 2019

@author: Thomas
"""
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, featuresdata, labels):
        self.features = featuresdata
        self.labels = labels
        
    def __len__(self):
        return len(self.features)   
    
    def __getitem__(self,idx):
        X = torch.tensor(self.features[idx]).float().view(1024,70)
        y = torch.tensor(self.labels[idx]).long().view(1,70)
        return X,y
    