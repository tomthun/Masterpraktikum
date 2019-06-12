# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 23:34:48 2019

@author: Thomas
"""

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

class SimpleCNN(nn.Module):
    
    # ToDo adapt CNN
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=5, stride = 1, padding = 2),
            nn.ReLU()) 
#        self.layer2 = nn.Sequential(
#            nn.Conv2d(32, 64, kernel_size=(5,1024), stride=1, padding=2),
#            nn.ReLU(),
#            nn.MaxPool2d(kernel_size=2, stride=2))
#       self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(1, 350)
        self.fc2 = nn.Linear(350, 6)
    
    def forward(self, x):
        out = self.layer1(x)
#        out = self.layer2(out)
        #out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
