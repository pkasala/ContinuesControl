import gym
import math
from unityagents import UnityEnvironment
import numpy as np



class EnvironmentWrapper:
    def __init__(self):
        self.action_dim = -math.inf
        self.state_dim = -math.inf
        self.action_space_min = 0
        self.action_space_max = 0
        self.env = None
        self.brain_name= None
        self.train_mode = True

    def __createEnv(self,name):
        return UnityEnvironment(file_name=name)

    def initEnviroment(self,name):
        self.env = self.__createEnv(name)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        # reset the environment
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]

        # number of agents
        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        # size of each action
        action_size  = brain.vector_action_space_size
        print('Size of each action:', action_size)

        # examine the state space
        states = env_info.vector_observations
        state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
        print('The state for the first agent looks like:', states[0])
        self.state_dim = state_size
        self.action_dim = action_size
        self.action_space_min = -1
        self.action_space_max = 1

    #reset the environment into init state
    def reset(self):
       env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
       return env_info.vector_observations

    #perform one step
    def step(self,action):
         env_info =  self.env.step(action)[self.brain_name]
         next_state = env_info.vector_observations[0]  # get next state (for each agent)
         reward = env_info.rewards[0]  # get reward (for each agent)
         done = env_info.local_done[0]  # see if episode finished
         return (next_state,reward,done)