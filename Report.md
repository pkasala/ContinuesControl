# Continues Control - Moving Arm - Agent Learning process
## Article goal
The aim of this article is to describe how the agent learns. 
And describe the TD3 algorithm which is based on Actor-Critic model.

## High level description
The full learning process is based on off-policy method. This means learning process is not directly on each action, but instead using some kind of experience memory.
Each experience is representing by tuple State, Action, Reward, NextState.
The agent starts with getting the experiences by execution fully random actions.
This phase is called warm-up and helps to randomize the action space.
Each agent experience is than stored in memory. 
After warm-up the agent starts to learn.
First takes the random mini-batch of experiences from memory - this step is called experience sampling.
From each experience perform the TD3 algorithm steps (describe later).
After certain learning steps, the current knowleadge is moved to target network copy to stabilize the learning process, by applying polyak update

## TD3 Algorithm
The algorithm is based on Actor-Critic algorithm with target Neural Network. This means each Neural Network has a target copy. And during the process the knowleadge of local neural newtork is copied (weight and bias values of neurons are copied over) to target network.
Basic Actor-Critic contains two NeuralNetwork. Where Actor NN (Policy NN) learns the optimal action for each state and Critic NN learns the Q-Value for state+action. Becasue this basis learning process happend in same time, this is not an optimal solution, and lead to the problem with aproximation error. 
TD3 algorithm extend the basic Actor-Critic and trying ot avoid the approximation error problem by adding second Critic model. 
### Algorithm steps:
1.  From experience take the NextState as an input to the Target Actor NN and predict the NextAction
2.  Than both Target Critic NN learns the two Next Q-Values based on NextState+predicted NextAction
3.  Take the minimum of two Next Q-Values, which actually represents the axpromition of value of next state.
    Taking minimum improves the stability, because the Q-Value is not over-estimated.
4.  As we just calculate the Next Q-Value, we can than apply this as a requisted Q-Value target to local Critic NN. 
    Simply by applying eq: requisted Q-Value = Reward + DiscountFactor*MIN(Next Q-Value)
5.  Get the prediciton of the two local Critic NN model for current state+current action
6.  Than we can calculate the loss over prediction and requisted Q-Value target
7.  Apply backpropagation of local critic loss
8.  Than as suggested by author of TD3, we perform the update of actor model not every learning iteration, but in slower manner.
    For example each third learning step.Applying the gradient chain rule we can apply the result of local Critic model to Actor model.
    This update the Actor weights into gradient direction of increasing the expected Q-Value. We have to applying the gradient ascent.
9.  Becasue we just update the local NNs, we need to either update the target NNs. 
    Update all target NN. Actor Target NN, and both Critic target NN by applying the polyak update rules. 
 
### Neural Network description
#### Actor Network
 This network define the policy. Means the best action at current state. The actions are defined by continues space. This require the regresion activation at output level of Neural Newtork. The action space is defined from -1 to 1 value. This negative and positive space prefered the Tahn output activation function. 
  The newtork is linear NN of 2 hidden layers with 400 and 300 neurons respectively. Layers are connected with RELU activation function.
#### Critic Network
  This network define the Q-Value function of any action taken in any states. The network model is similar to actor model. The diferent is on output layer activation function which is RELU function, because the reward is a continues positive value. 
  


The whole process is describe by this picture:

![Model](https://github.com/pkasala/ContinuesControl/blob/master/model_explain.png)
@Copyright by Authors of course - Deep Reinforcement Learning 2.0

The information has been taken from https://arxiv.org/pdf/1802.09477.pdf and Course Deep Reinforcement Learning 2.0 https://www.udemy.com/deep-reinforcement-learning/



        



