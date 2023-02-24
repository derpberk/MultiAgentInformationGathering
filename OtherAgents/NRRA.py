import numpy as np

class WanderingAgent:

    def __init__(self, world: np.ndarray, movement_length: float, number_of_actions: int, consecutive_movements = None):
        
        self.world = world
        self.move_length = movement_length
        self.number_of_actions = number_of_actions
        self.consecutive_movements = consecutive_movements
        self.t = 0
        self.action = None
    
    def move(self, actual_position):

        if self.action is None:
            self.action = self.select_action_without_collision(actual_position)
        
        # Compute if there is an obstacle or reached the border #
        OBS = self.check_collision(self.action, actual_position)

        if OBS:
            self.action = self.select_action_without_collision(actual_position)

        if self.consecutive_movements is not None:
            if self.t == self.consecutive_movements:
                self.action = self.select_action_without_collision(actual_position)
                self.t = 0

        self.t += 1
        return self.action
    
    
    def action_to_vector(self, action):
        """ Transform an action to a vector """

        vectors = np.array([[np.cos(2*np.pi*i/self.number_of_actions), np.sin(2*np.pi*i/self.number_of_actions)] for i in range(self.number_of_actions)])

        return np.round(vectors[action]).astype(int)
    
    def opposite_action(self, action):
        """ Compute the opposite action """
        return (action + self.number_of_actions//2) % self.number_of_actions
    
    def check_collision(self, action, actual_position):
        """ Check if the agent collides with an obstacle """
        new_position = actual_position + self.action_to_vector(action) * self.move_length
        OBS = (new_position[0] < 0) or (new_position[0] >= self.world.shape[0]) or (new_position[1] < 0) or (new_position[1] >= self.world.shape[1])
        if not OBS:
            OBS = self.world[new_position[0], new_position[1]] == 0

        return OBS

    def select_action_without_collision(self, actual_position):
        """ Select an action without collision """
        action_caused_collision = [self.check_collision(action, actual_position) for action in range(self.number_of_actions)]

        # Select a random action without collision and that is not the oppositve previous action #
        if self.action is not None:
            opposite_action = self.opposite_action(self.action)
            action_caused_collision[opposite_action] = True
        action = np.random.choice(np.where(np.logical_not(action_caused_collision))[0])

        return action