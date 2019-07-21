import gym
import numpy as np
from collections import deque

from Config import Config
from EnvironmentWrapper import EnvironmentWrapper
from Agent import Agent
from Memory import ReplayBuffer
from NeuralNetwork import *
import Util as util

gym.logger.set_level(40) # suppress warnings (please remove if gives error)
UNITY_ENVIROMENT = '../../reacher/Reacher_Windows_x86_64/Reacher.exe'
MODEL = "TD3"

class ExperienceManager:

    def __init__(self):
        self.config =None
        self.agent = None
        self.model_file_name = None
        self.plot_file_name = None

    def initConfig(self):
        self.config = Config()

    # initialize enviroment and set the state space size and  action space size
    def initEnviroment(self):
        print('Initialize env')
        # update to your
        env = EnvironmentWrapper()
        env.initEnviroment(UNITY_ENVIROMENT)
        print('observation space:', env.state_dim)
        print('action space:', env.action_dim)
        # get the default brain
        print('Env init done')
        self.config.model_name = MODEL
        self.config.env = env

    def initNetwork(self):
        #alias
        env = self.config.env
        config = self.config
        #init Actor local and target
        config.actor_local = Actor(env.state_dim, env.action_dim)
        config.actor_target = Actor(env.state_dim, env.action_dim)
        #copy init weights and bias
        config.actor_target.load_state_dict(config.actor_local.state_dict())
        config.actor_optimizer = torch.optim.Adam(config.actor_local.parameters())
        #init critic
        config.critic_local = Critic(env.state_dim, env.action_dim)
        config.critic_target = Critic(env.state_dim, env.action_dim)
        # copy init weights and bias
        config.critic_target.load_state_dict(config.critic_local.state_dict())
        config.critic_optimizer = torch.optim.Adam(config.critic_local.parameters())

    def initAgent(self):
        self.agent = Agent(self.config)

    def initMemory(self):
        self.config.memory =  ReplayBuffer(self.config.memory_buffer_size,self.config.memory_batch_size,self.config.memory_seed)

    def initMiscellaneous(self):
        self.model_file_name = util.model_file_name(self.config)
        self.plot_file_name = util.plot_file_name(self.config)

    def runEpisode(self):
        done = False
        score = 0
        #alias
        conf = self.config
        agent = self.agent
        while done==False:
            reward,done = agent.step()
            score +=reward
        return score

    def runExperiment(self):
        i_episode = 0
        #over all scores
        scores = []
        #last 100 scores
        scores_window = deque(maxlen=100)  # last 100 scores
        for i in range(self.config.max_episode):
            #run the one episode
            ep_score = self.runEpisode()
            scores.append(ep_score)
            scores_window.append(ep_score)
            i_episode += 1
            # print progress
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            # keep progress of last 100 episodes
            if i_episode % self.config.print_each_steps == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if i_episode % self.config.store_each_episodes ==0:
                util.save_plot_scores(scores,self.plot_file_name)
                self.agent.save(self.model_file_name)


experiment = ExperienceManager()
experiment.initConfig()
experiment.initEnviroment()
experiment.initNetwork()
experiment.initMemory()
experiment.initAgent()
experiment.initMiscellaneous()

experiment.runExperiment()