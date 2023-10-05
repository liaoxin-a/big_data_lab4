import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import configparser
config = configparser.ConfigParser()
config_path=os.path.join('config','config.ini')
config.read(config_path)
FEATURES_DIM=config['default'].getint('features_dim')
NUM_CLASSES=config['default'].getint('num_classes')

class SwishActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(x) * x

class ClassifierModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        width = FEATURES_DIM

        for num_neurons in [2765, 1662]:
            layers.append(nn.Linear(width, num_neurons))
            width = num_neurons

            layers.append(nn.BatchNorm1d(width))
            layers.append(SwishActivation())

        layers.append(nn.Linear(width, NUM_CLASSES))
        self.layers = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.layers(x)
        return x


