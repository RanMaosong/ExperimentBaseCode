import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basemodel import BaseModel

class MinistClassifier(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.initialize()

    def forward(self, x):
        x = torch.reshape(x, (-1, 28 * 28))
        out = self.fc1(x)
        out = F.relu(out)
        logit = self.fc2(out)
        return logit

