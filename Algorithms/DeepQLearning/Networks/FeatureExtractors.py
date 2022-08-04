import torch
from torch import nn
from gym.spaces import Box
import numpy as np


class FeatureExtractor(nn.Module):
    """ Convolutional Feature Extractor for input images """

    def __init__(self, obs_space_shape, num_of_features):
        super(FeatureExtractor, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=obs_space_shape[0], out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.cnn_out_size = np.prod(self.cnn(torch.zeros(size=(1,
                                                               obs_space_shape[0],
                                                               obs_space_shape[1],
                                                               obs_space_shape[2]))).shape)

        self.linear = nn.Linear(int(self.cnn_out_size), num_of_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))
