
# SARSA(inf), Boltzmann
# switch/stay, PD game
# knowing opponent past 1 action

import numpy as np

from Environment02 import Environment02

class Agent36b:
    def __init__(self, env, lr = 0.1, gamma = 1, tau = 1, name='IL-bolt'):
        self.name = name
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.Npast = 1

        self.Q = {}
        self.Q[0] = np.zeros(2, dtype=np.float32)  # (link? 0:N; 1:Y) | opponent C
        self.Q[1] = np.zeros(2, dtype=np.float32)  # (link? 0:N; 1:Y) | opponent D
        self.Q[100] = np.zeros(env.Nact, dtype=np.float32)  # (break? 0:N; 1:Y) | opponent C
        self.Q[101] = np.zeros(env.Nact, dtype=np.float32)  # (break? 0:N; 1:Y) | opponent C
        self.Q[200] = np.zeros(env.Nact, dtype=np.float32)  # (PD? 0:C; 1:D) | opponent C
        self.Q[201] = np.zeros(env.Nact, dtype=np.float32)  # (PD? 0:C; 1:D) | opponent D
        # self.Q[200] = np.zeros(env.Nact, dtype=np.float32)  # (PD? 0:C; 1:D) | opponent C
        # self.Q[201] = np.zeros(env.Nact, dtype=np.float32)  # (PD? 0:C; 1:D) | opponent D
        
        self.mStates1 = []
        self.mActions1 = []
        self.mRewards1 = []

    def storeMemory(self, s, a, r):
        self.mStates1.append(s)
        self.mActions1.append(a)
        self.mRewards1.append(r)

    def getQall(self):
        Qall = np.array([], dtype=np.float32)
        for i in range(self.Npast + 1):
            Qall = np.concatenate((Qall, self.Q[i]))
        for i in range(self.Npast + 1):
            Qall = np.concatenate((Qall, self.Q[100+i]))
        for i in range(self.Npast + 1):
            Qall = np.concatenate((Qall, self.Q[200+i]))
        return Qall

    def getQsa(self, s, a):
        return self.Q[s][a]

    def getQ(self, s):
        return self.Q[s]

    def getPolicyAll(self):
        Pall = np.array([], dtype=np.float32)
        for i in range(self.Npast + 1):
            Pall = np.concatenate((Pall, self.getPolicy(i)))
        for i in range(self.Npast + 1):
            Pall = np.concatenate((Pall, self.getPolicy(100+i)))
        for i in range(self.Npast + 1):
            Pall = np.concatenate((Pall, self.getPolicy(200+i)))
            
        return Pall

    def getPolicy(self, s):
        return self.softmax(self.tau*self.Q[s])

    def softmax(self, x):
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator
        return softmax

    def getAction(self, s): # all actions are legal
        policyS = self.getPolicy(s)
        action = np.random.choice(len(policyS), p=policyS)
        return action

    def train(self):
        discountedRewards1 = self.calDiscountedRewards()

        mStates = self.mStates1
        mActions = self.mActions1
        discountedRewards = discountedRewards1

        if self.name[-2:] == '_0' and False:
            print('%s; %s; %s'%(mStates, mActions, discountedRewards))

        for s, a, sample in zip(mStates, mActions, discountedRewards):
            self.Q[s][a] = (1-self.lr)*self.Q[s][a] + self.lr*sample

        self.mStates1 = []
        self.mActions1 = []
        self.mRewards1 = []

    def calDiscountedRewards(self):
        mRewards = self.mRewards1
        discountedRewards = []
        runningSum = 0
        for reward in mRewards[::-1]:
            runningSum = reward + self.gamma*runningSum
            discountedRewards.append(runningSum)
        discountedRewards.reverse()
        discountedRewards = np.array(discountedRewards)
        return discountedRewards

if __name__ == '__main__':
    env = Environment02(2, np.zeros((2,2)), np.zeros((2,2)))
    agent = Agent36b(env)
    print(agent.getQall())
    print(agent.getPolicyAll())
    pass
