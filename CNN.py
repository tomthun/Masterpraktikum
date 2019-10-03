# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:34:48 2019

@author: Thomas
"""

import torch.nn as nn
from torchcrf import CRF
class SimpleCNN(nn.Module):
    
    # ToDo adapt CNN
    def __init__(self,num_tags):
        super(SimpleCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 254, kernel_size=(5,1),  padding = (2,0)), nn.ReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(254)) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(254, 64, kernel_size=(3,1),  padding=(1,0)), nn.ReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=(1,1),  padding=(0,0)), nn.ReLU(), nn.Dropout2d(0.65), nn.BatchNorm2d(4))
        self.crf = CRF(num_tags)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out
