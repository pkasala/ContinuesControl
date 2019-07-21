import numpy as np
import torch
import torch.nn.functional as F

class Agent():
    def __init__(self,config):
        self.config = config
        self.env = config.env
        self.memory = config.memory
       # self.random_process = config.random_process_fn()
        self.actor_local = config.actor_local
        self.actor_target = config.actor_target
        self.actor_optim = config.actor_optimizer

        self.critic_local = config.critic_local
        self.critic_target = config.critic_target
        self.critic_optim = config.critic_optimizer

        self.total_steps = 0
        self.state = None

    def get_action(self,state):
        #fully random action until warm up reached
        if self.total_steps < self.config.warm_up:
            actions = np.random.randn(self.config.num_agents,
                                      self.config.env.action_dim)  # select an action (for each agent)
            actions = np.clip(actions,self.config.env.action_space_min , self.config.env.action_space_max)  # all actions between -1 and 1
            return actions
        #action from local actor network
        else:
            state = torch.Tensor(state.reshape(1, -1)).to(self.config.DEVICE)
            return self.actor_local(state).cpu().data.numpy().flatten()

    def step(self):
        #alias
        config = self.config
        env = config.env
        #if first step, reset the environment
        if self.state is None:
            self.state = config.env.reset()
        #obtain the action, if warmup, this returns sample action values
        action = self.get_action(self.state)
        # If the explore_noise parameter is not 0, we add noise to the action and we clip it
        if config.explore_noise != 0:
          action = (action + np.random.normal(0, config.explore_noise , size= config.env.action_dim)).clip(config.env.action_space_min , config.env.action_space_max)
        # perform one step action        
        next_state, reward, done = env.step(action)
        #store action to memory
        config.memory.add(self.state, action, reward, next_state, done)
        #start the local exploration, which means reward other than 0 at the begining
        if reward !=0 and self.total_steps<config.local_explore_unti_steps:
            self.small_interaction(action,reward)
        #update the total number of steps
        self.total_steps += 1
        #set the current state
        self.state  = next_state
        # learn
        if self.total_steps>config.warm_up and (self.total_steps) % self.config.learn_every == 0:
            # If enough samples are available in memory start learning
            if self.memory.size() >= config.memory.batch_size :
                self.learn()
        return reward,done

    def step_without_learn(self):
        # alias
        config = self.config
        env = config.env
        # if first step, reset the environment
        if self.state is None:
            self.state = config.env.reset()
        # obtain the action, if warmup, this returns sample action values
        action = self.get_action(self.state)
        # perform one step action
        next_state, reward, done = env.step(action)
        return reward, done

    def small_interaction(self,action,reward):
        config = self.config
        # perform one step action
        step_size = self.config.local_explore_step
        max_steps = 20
        done = False
        step_num = 0
        #perform small random change to actions until gets reward or max_steps reached
        while reward != 0 and done==False and step_num<max_steps:
            step_num +=1
            action = (action + np.random.normal(0, step_size*step_num, size=config.env.action_dim)).clip(config.env.action_space_min, config.env.action_space_max)
            next_state, reward, done = self.config.env.step(action)
            config.memory.add(self.state, action, reward, next_state, done)
        print("\rlocal explore num {}".format(step_num))

    def learn(self):
        #Step 1 - get sample
        states, actions, rewards, next_states, dones = self.config.memory.sample()
        #Step2 - from actor target obtain the next_action
        next_actions = self.actor_target(next_states)
        #Step3 - from two critics target network get the Q' values = prediction of next_state and next_action
        Q1_target, Q2_target = self.critic_target(next_states,next_actions)
        # step4 - get the minimum of Q'
        Q_target_min = torch.min(Q1_target,Q2_target)
        #step5 - calculate the future Q
        Q_target = rewards + ((1-dones)* self.config.discount * Q_target_min).detach()
        #step6 - from two critic target the Q value = prediction of state and action
        Q1_local, Q2_local  = self.critic_local(states,actions)

        #step7 - calculate loos as sum of MSE betwee
        critic_loss = F.mse_loss(Q1_local,Q_target) + F.mse_loss(Q2_local,Q_target)
        #step 8 update the critic local networks.
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.total_steps % self.config.policy_freg_update == 0:
            #step 9 - update the local actor network
            actor_actions = self.actor_local(states)
            #gradient ascent so the
            actor_loss = -self.critic_local.Q1(states,actor_actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            #step 10 - perform the polyak update of all target networks. (First and second critic network is build in NeuralNetwork class)
            self.polyak_update(self.actor_local,self.actor_target)
            self.polyak_update(self.critic_local, self.critic_target)

    #small update from local to target network
    def polyak_update(self,source,target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.config.polyak_tau * param.data + (1 - self.config.polyak_tau) * target_param.data)

    #save the save of network
    def save(self, filename):
        torch.save(self.actor_local.state_dict(), '%sactor.pth' % (filename))
        torch.save(self.critic_local.state_dict(), '%scritic.pth' % (filename))