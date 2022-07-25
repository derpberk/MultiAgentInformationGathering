from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch


class FullVisualQModel(TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.num_outputs = action_space.n

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
            torch.nn.Linear(256, 256))

        """ Create a sequential model with 3 fully connected layers with an output size of self.num_outputs for the q values """
        self.q_fc = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.num_outputs))

        """ Create a sequential model with 3 fully connected layers with an output size of self.num_outputs for the value function """
        self.v_fc = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

        self.v_vals = None

    def forward(self, input_dict, state, seq_lens):
        """ Get the state from input_dict and forward in this order: 1) cnn, 2) fc, 3) q_fc, 4) v_fc """
        x = input_dict["obs"]
        x = self.cnn(x)
        x = self.fc(x)
        q_vals = self.q_fc(x)
        self.v_vals = self.v_fc(x)  # Save value function for later use
        return q_vals


    def value_function(self):
        return self.v_vals