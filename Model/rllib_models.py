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

class DictVisualQModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        """ Create a squential model with convolutional neural network with 3 convolutional layers """
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=obs_space['visual_observation'].shape[0], out_channels=32, kernel_size=7, stride=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Flatten())

        """ Compute the output size of the CNN """
        cnn_output_size = self.cnn(torch.zeros(1, *obs_space['visual_observation'].shape)).shape[1]

        """ Create a final layer for the feature extractor """

        n_features = 256 + obs_space['vector_observation'].shape[0]
        self.feature_extractor_final_fc = torch.nn.Linear(cnn_output_size, 256)

        """ The fully sequential final layer. The vector observation is appended to the features """
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_outputs))

        self.v_vals = None

        self.v_fc = torch.nn.Linear(num_outputs, 1)

    def forward(self, input_dict, state, seq_lens):
        """ Get the state from input_dict and forward in this order: 1) cnn, 2) fc, 3) q_fc, 4) v_fc """
        x = input_dict["obs"]["visual_observation"]
        x = self.cnn(x)
        x = self.feature_extractor_final_fc(x)
        # Append the output of the feature extractor to the vector observation
        x = torch.cat((x, input_dict["obs"]["vector_observation"]), dim=1)

        q_vals = self.fc(x)
        self.v_vals = self.v_fc(q_vals)  # Save value function for later use

        return q_vals, state


    def value_function(self):
        return self.v_vals

if __name__ == '__main__':
    import gym
    import numpy as np


    action_space = gym.spaces.Discrete(8)
    observation_space = gym.spaces.Dict({'visual_observation': gym.spaces.Box(low=-1.0, high=1.0, shape=(3, 50, 50)),
                                         'vector_observation': gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))})

    model = DictVisualQModel(observation_space, action_space, 64, {}, 'my_model')
    v_input = torch.FloatTensor()

    v_input = torch.FloatTensor(np.random.rand(5, 3, 50, 50))
    o_input = torch.FloatTensor(np.random.rand(5,2))
    input = {'visual_observation': v_input, 'vector_observation': o_input}
    input_dict = {'obs': input}

    output = model(input_dict)
    print(output[0].shape)