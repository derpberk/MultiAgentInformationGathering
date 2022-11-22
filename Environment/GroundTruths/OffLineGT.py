""" Offline GT environment. It records all ground truths in an off-line fashion for speeding up
the training process """

from groundtruth import GroundTruth
import numpy as np
from tqdm import trange

def record_env(gt: GroundTruth, path: str, number_of_epochs: int = 1, starting_time: int = 0):
	""" Reset the env, update to time t0 and record the gt field. """

	ground_truths = []

	for run in trange(number_of_epochs):
		# Reset the environment # 
		gt.reset()
		# Update to time t0:
		gt.update_to_time(starting_time)
		# Get the environment field #
		ground_truths.append(gt.ground_truth_field.astype(np.float16))

	# Binarize the resulting data 

	np_ground_truths = np.asarray(ground_truths)
	np.save(path, np_ground_truths)


class OfflineGT(GroundTruth):
	""" Offline bag of gt experiences """

	def __init__(self, sim_config: dict, path: str):
		super().__init__(sim_config)

		# Load the paths #
		self.gts = np.random.shuffle(np.load(path))
		self.ground_truth_field = None
		self.t_pointer = -1

	def reset(self, random_gt: bool = False):
		
		if random_gt:
			self.t_pointer += 1
			self.t_pointer = 0 if self.t_pointer >= len(self.gts) else self.t_pointer
		
		self.ground_truth_field = self.gts[self.t_pointer]
		
	def read(self, position=None):

		if position is None:
			return self.ground_truth_field
		else:
			return self.ground_truth_field[int(position[0]), int(position[1])]

	def step(self):
		return None

	def update_to_time(self, t):
		return None


if __name__ == '__main__':

	import matplotlib.pyplot as plt

	# EXAMPLE FOR RECORDING WILD FIRE SIMULATOR #

	from FireFront import WildfireSimulator

	config = WildfireSimulator.sim_config_template
	ground_truth = WildfireSimulator(config)

	record_env(gt = ground_truth, path = './Environment/WFgts.npy', number_of_epochs=100, starting_time=30)


	gt = OfflineGT(OfflineGT.sim_config_template, "./Environment/WFgts.npy")

	gt.reset(True)

	for i in range(50):

		plt.imshow(gt.read(), cmap='hot')
		plt.pause(0.5)
		gt.reset(True)

	




