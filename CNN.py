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
            nn.Conv2d(1024, 4, kernel_size=(5,1),  padding = (2,0))) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(100, 4, kernel_size=(3,1),  padding=(1,0)))
        self.fc1 = nn.Linear(1, 350)
        self.fc2 = nn.Linear(350, 6)
        self.crf = CRF(num_tags)
    
    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        #out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        #out = self.fc1(out)
        #out = self.fc2(out)
        return out
