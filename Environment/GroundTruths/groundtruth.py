
class GroundTruth:
	""" This is a template for a ground truth. Every ground truth must inherit and implement all of its methods and variables. """

	sim_config_template = {
		"seed": 0,
	}

	def __init__(self, sim_config: dict):

		self.seed = sim_config['seed']
		self.ground_truth_field = None

	def step(self):
		raise NotImplementedError('This method has not being implemented yet.')

	def reset(self, random_gt: bool = False):
		""" Reset ground Truth """
		raise NotImplementedError('This method has not being implemented yet.')

	def read(self, position=None):
		""" Read the complete ground truth or a certain position """
		raise NotImplementedError('This method has not being implemented yet.')

	def update_to_time(self, t):
		""" Update the environment a number of steps """
		raise NotImplementedError('This method has not being implemented yet.')



