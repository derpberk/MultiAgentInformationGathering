from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn


class FullVisualQModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        """ Create a squential model with convolutional neural network with 3 convolutional layers """
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=obs_space.shape[0], out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten())

        """ Compute the output size of the CNN """
        cnn_output_size = self.cnn(torch.zeros(1, *obs_space.shape)).shape[1]

        """ Create a sequential model with 3 fully connected layers and ReLu activation function """
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(cnn_output_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_outputs))

        self.v_vals = None

        self.v_fc = torch.nn.Linear(num_outputs, 1)

    def forward(self, input_dict, state, seq_lens):
        """ Get the state from input_dict and forward in this order: 1) cnn, 2) fc, 3) q_fc, 4) v_fc """
        x = input_dict["obs"]
        x = self.cnn(x)
        q_vals = self.fc(x)
        self.v_vals = self.v_fc(q_vals)  # Save value function for later use

        return q_vals, state


    def value_function(self):
        return self.v_vals