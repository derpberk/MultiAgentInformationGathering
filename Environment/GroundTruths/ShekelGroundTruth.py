import sys
import numpy as np
from deap import benchmarks
from collections import defaultdict
from Environment.GroundTruths.groundtruth import GroundTruth


class Shekel(GroundTruth):

    """ Ground Truth generator class.
        It creates a ground truth within the specified navigation map.
        The ground truth is generated randomly following some realistic rules of the enviornment
        and using a Shekel function.
    """

    sim_config_template = {
        "navigation_map": np.ones((100, 100)),
        "seed": 9875123,
        "max_number_of_peaks": None,
        "normalize": True,
    }

    def __init__(self, sim_config: dict):
        super().__init__(sim_config)

        self.ground_truth_field = None
        self.max_number_of_peaks = 6 if sim_config['max_number_of_peaks'] is None else sim_config['max_number_of_peaks']
        self.seed = sim_config['seed']

        np.random.seed(self.seed)

        """ random map features creation """
        self.grid = 1 - sim_config['navigation_map']

        self.xy_size = np.array([self.grid.shape[1]/self.grid.shape[0]*10, 10])
        self.normalize_gt = sim_config['normalize']

        # Peaks positions bounded from 1 to 9 in every axis
        self.number_of_peaks = np.random.randint(1, self.max_number_of_peaks+1)
        self.A = np.random.rand(self.number_of_peaks, 2) * self.xy_size * 0.8 + self.xy_size*0.2
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = np.random.rand(self.number_of_peaks, 1) * 4 + 0

        """ Creation of the map field """
        self._x = np.arange(0, self.grid.shape[1], 1)
        self._y = np.arange(0, self.grid.shape[0], 1)

        self._x, self._y = np.meshgrid(self._x, self._y)

        self._z, self.meanz, self.stdz, self.normalized_z = None, None, None, None # To instantiate attr after assigning in __init__
        self.create_field()  # This method creates the normalized_z values

    def shekel_arg0(self, sol):

        return np.nan if self.grid[sol[1]][sol[0]] == 1 else \
            benchmarks.shekel(sol[:2]/np.array(self.grid.shape)*10, self.A, self.C)[0]

    def create_field(self):

        """ Creation of the normalized z field """
        self._z = np.fromiter(map(self.shekel_arg0, zip(self._x.flat, self._y.flat)), dtype=np.float32,
                              count=self._x.shape[0] * self._x.shape[1]).reshape(self._x.shape)

        self.meanz = np.nanmean(self._z)
        self.stdz = np.nanstd(self._z)

        if self.stdz > 0.001:
            self.normalized_z = (self._z - self.meanz) / self.stdz
        else:
            self.normalized_z = self._z

        if self.normalize_gt:
            self.normalized_z = np.nan_to_num(self.normalized_z, nan=np.nanmin(self.normalized_z))
            self.normalized_z = (self.normalized_z - np.min(self.normalized_z))/(np.max(self.normalized_z) - np.min(self.normalized_z))
            
        self.ground_truth_field = self.normalized_z

    def reset(self, random_benchmark=True):


        if random_benchmark:
            self.seed += 1

        """ Reset ground Truth """
        # Peaks positions bounded from 1 to 9 in every axis
        self.number_of_peaks = np.random.RandomState(self.seed).randint(1, self.max_number_of_peaks+1)
        self.A = np.random.RandomState(self.seed).rand(self.number_of_peaks, 2) * self.xy_size * 0.9 + self.xy_size*0.1
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = np.random.RandomState(self.seed).rand(self.number_of_peaks, 1) * 4 + 1
        # Reconstruct the field #
        self.create_field()

    def read(self, position=None):

        """ Read the complete ground truth or a certain position """

        if position is None:
            return self.ground_truth_field
        else:
            position = position.astype(int)
            return self.ground_truth_field[position[0]][position[1]]

    def render(self):

        """ Show the ground truth """
        grid = self.read()
        grid[np.where(self.grid == 1)] = np.nan
        plt.imshow(grid, cmap='coolwarm', interpolation='none')
        #cs = plt.contour(self.read(), colors='royalblue', alpha=1, linewidths=1)
        #plt.clabel(cs, inline=1, fontsize=7)
        #plt.title("Nº of peaks: {}".format(gt.number_of_peaks), color='black', fontsize=10)
        #im = plt.plot(self.A[:, 0]*self.grid.shape[0]/10,
        #              self.A[:, 1]*self.grid.shape[1]/10, 'hk', )
        plt.show()

    def step(self):
        # This ground truth is static #
        pass

    def update_to_time(self, t):
        pass


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    nav_map = np.genfromtxt('Environment/wesslinger_map.txt')

    gt_config = Shekel.sim_config_template
    gt_config['navigation_map'] = nav_map
    gt = Shekel(gt_config)

    for i in range(10):
        gt.reset(random_benchmark=True)
        gt.render()
        plt.pause(0.1)






