import ray
from ray import tune
from ray.rllib.algorithms.dqn.dqn import DQN
from TrainingConfigurations.ShekelConfiguration import create_configuration


""" Initialize Ray """
ray.init()

""" Environment configuration"""
configuration = create_configuration()

""" Stop criterion """
stop_criterion = {
	"episodes_total": 1,
}

""" Build the alorithm """

# Create our RLlib Trainer.
tune.run(DQN,
		 config=configuration.to_dict(),
		 stop=stop_criterion,
		 local_dir='./runs',
		 checkpoint_at_end=True,
		 checkpoint_freq=100)


