import torch

class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):

        self.num_agents = 1
        self.model_name = ""
        self.env= None
        self.warm_up = 5000

        self.memory_buffer_size = 100000
        self.memory_batch_size = 100
        self.memory_seed = None
        self.memory = None

        self.actor_local = None
        self.actor_target = None
        self.actor_optimizer = None
        self.critic_local = None
        self.critic_target = None
        self.critic_optimizer = None

        self.explore_noise = 0.08

        self.policy_freg_update = 3
        self.polyak_tau = 0.008
        self.learn_every = 5

        self.discount = 0.99

        self.print_each_steps = 10
        self.store_each_episodes = 10

        self.max_episode = 1000

        self.local_explore_step = 0.005
        self.local_explore_unti_steps = 40000