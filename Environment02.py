# 2 players n actions

import numpy as np


class Environment02:
    def __init__(self, Nact, reward1s, reward2s):
        self.Nact = Nact
        self.dimSA = (1, Nact)
        self.actions = np.arange(Nact)
        
        self.reward1s = reward1s
        self.reward2s = reward2s
    
    def getRewards(self, actions):
        return self.reward1s[actions[0], actions[1]], self.reward2s[actions[0], actions[1]]




