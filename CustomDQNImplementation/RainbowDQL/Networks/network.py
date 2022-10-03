from ..NoisyLayers.layers import NoisyLinear
import torch
from torch import nn
import torch.nn.functional as F
from ..Networks.FeatureExtractors import FeatureExtractor


class Network(nn.Module):
	def __init__(
			self,
			in_dim: int,
			out_dim: int,
			atom_size: int,
			support: torch.Tensor
	):
		"""Initialization."""
		super(Network, self).__init__()

		self.support = support
		self.out_dim = out_dim
		self.atom_size = atom_size

		# set common feature layer
		self.feature_layer = nn.Sequential(
			nn.Linear(in_dim, 128),
			nn.ReLU(),
		)

		# set advantage layer
		self.advantage_hidden_layer = NoisyLinear(128, 128)
		self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

		# set value layer
		self.value_hidden_layer = NoisyLinear(128, 128)
		self.value_layer = NoisyLinear(128, atom_size)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation."""
		dist = self.dist(x)
		q = torch.sum(dist * self.support, dim=2)

		return q

	def dist(self, x: torch.Tensor) -> torch.Tensor:
		"""Get distribution for atoms."""
		feature = self.feature_layer(x)
		adv_hid = F.relu(self.advantage_hidden_layer(feature))
		val_hid = F.relu(self.value_hidden_layer(feature))

		advantage = self.advantage_layer(adv_hid).view(
			-1, self.out_dim, self.atom_size
		)
		value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
		q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

		dist = F.softmax(q_atoms, dim=-1)
		dist = dist.clamp(min=1e-3)  # for avoiding nans

		return dist

	def reset_noise(self):
		"""Reset all noisy_layers layers."""
		self.advantage_hidden_layer.reset_noise()
		self.advantage_layer.reset_noise()
		self.value_hidden_layer.reset_noise()
		self.value_layer.reset_noise()

class DuelingVisualNetwork(nn.Module):

	def __init__(
			self,
			in_dim: tuple,
			out_dim: int,
			number_of_features: int,
	):
		"""Initialization."""
		super(DuelingVisualNetwork, self).__init__()

		self.out_dim = out_dim

		# set common feature layer
		self.feature_layer = nn.Sequential(
			FeatureExtractor(in_dim, number_of_features),
			nn.Linear(number_of_features, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
		)

		# set advantage layer
		self.advantage_hidden_layer = nn.Linear(256, 64)
		self.advantage_layer = nn.Linear(64, out_dim)

		# set value layer
		self.value_hidden_layer = nn.Linear(256, 64)
		self.value_layer = nn.Linear(64, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation."""
		feature = self.feature_layer(x)

		adv_hid = F.relu(self.advantage_hidden_layer(feature))
		val_hid = F.relu(self.value_hidden_layer(feature))

		value = self.value_layer(val_hid)
		advantage = self.advantage_layer(adv_hid)

		q = value + advantage - advantage.mean(dim=-1, keepdim=True)

		return q

class NoisyDuelingVisualNetwork(nn.Module):

	def __init__(
			self,
			in_dim: tuple,
			out_dim: int,
			number_of_features: int,
	):
		"""Initialization."""
		super(NoisyDuelingVisualNetwork, self).__init__()

		self.out_dim = out_dim

		# set common feature layer
		self.feature_layer = nn.Sequential(
			FeatureExtractor(in_dim, number_of_features))

		self.common_layer_1 = NoisyLinear(number_of_features, 256)
		self.common_layer_2 = NoisyLinear(256, 256)
		self.common_layer_3 = NoisyLinear(256, 256)

		# set advantage layer
		self.advantage_hidden_layer = NoisyLinear(256, 64)
		self.advantage_layer = NoisyLinear(64, out_dim)

		# set value layer
		self.value_hidden_layer = NoisyLinear(256, 64)
		self.value_layer = NoisyLinear(64, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward method implementation."""
		feature = self.feature_layer(x)
		feature = F.relu(self.common_layer_1(feature))
		feature = F.relu(self.common_layer_2(feature))
		feature = F.relu(self.common_layer_3(feature))

		adv_hid = F.relu(self.advantage_hidden_layer(feature))
		val_hid = F.relu(self.value_hidden_layer(feature))

		value = self.value_layer(val_hid)
		advantage = self.advantage_layer(adv_hid)

		q = value + advantage - advantage.mean(dim=-1, keepdim=True)

		return q

	def reset_noise(self):

		self.common_layer_1.reset_noise()
		self.common_layer_2.reset_noise()
		self.common_layer_3.reset_noise()

		self.advantage_hidden_layer.reset_noise()
		self.advantage_layer.reset_noise()

		self.value_hidden_layer.reset_noise()
		self.value_layer.reset_noise()









