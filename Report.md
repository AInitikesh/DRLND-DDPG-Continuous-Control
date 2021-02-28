# Report
---

## Video 

[![](http://img.youtube.com/vi/rAa5RplGmeA/0.jpg)](http://www.youtube.com/watch?v=rAa5RplGmeA "")

## Learning algorithm

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is continuous, and in order to solve the environment, agent must get an average score of +30 over 100 consecutive episodes. 
Training algorithm is `In [5]: ddpg` inside [Continuous_Control.ipynb](https://github.com/AInitikesh/DRLND-DDPG-Continuous-Control/blob/master/Continuous_Control.ipynb). This function iterates over `n_episodes=200` to train the ddpg agent model. Max length of episode can be `max_t=1000`. Maximum time steps value should be equal to Agent replay buffer. I have used environment version 2 where 20 parallel agents are simulated. After 200 episodes model was not learning much and average score was constant so it doesn't makes sense to train the Agent after 200 steps. 

### DDPG Agent Hyper Parameters

- BUFFER_SIZE (int): replay buffer size
- BATCH_SIZE (int): minibatch size
- GAMMA (float): discount factor
- TAU (float): for soft update of target parameters
- LR_ACTOR (float): learning rate of the actor 
- LR_CRITIC (float): learning rate of the critic
- WEIGHT_DECAY (float): L2 weight decay

Where 
`BUFFER_SIZE = int(1e5)`, `BATCH_SIZE = 128`, `GAMMA = 0.99`, `TAU = 1e-3`, `LR_ACTOR = 1e-3`, `LR_CRITIC = 1e-3` and `WEIGHT_DECAY = 0`   

### Neural Network

DDPG is an actor-critic method which uses 2 neural networks. One is Actor network that learns to predicts best action for given state and Critic network that learns to estimate the Q value for a given state action pair. Critic network is used to estimate the loss for Actor network.

1) [Actor model](https://github.com/AInitikesh/DRLND-DDPG-Continuous-Control/blob/master/model.py#L12) - Consist of an input layer of state size(33), two fully connected hidden layers of size 400 and 300 having relu activation and output fully connected layer size of action_size(4) and tanh activation function.

1) [Critic model](https://github.com/AInitikesh/DRLND-DDPG-Continuous-Control/blob/master/model.py#L44) - Consist of two input layers. First  input of state size(33) followed by fully connected hidden layer of size 400 and relu activation. We then concat the output of first hidden layer with second input ie action size(4) followed by one more hidden layer of size 300 and relu activation. Final output layer predicts a single Q value.

![DDPG algorithm](https://github.com/AInitikesh/DRLND-DDPG-Continuous-Control/blob/master/ddpg-algo.png)

Referenced from original paper [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT
LEARNING](https://arxiv.org/pdf/1509.02971v6.pdf)

## Plot of Rewards

### Reward Plot QNetwork

![Reward Plot DDPG Network](https://github.com/AInitikesh/DRLND-DDPG-Continuous-Control/blob/master/score-card.png)

```
Episode 100	Average Score: 23.49
Episode 200	Average Score: 36.84

Environment solved in 100 episodes!	Average Score: 36.84
```

## Ideas for Future Work

Implement other methods like Trust Region Policy Optimization (TRPO) and Truncated Natural Policy Gradient (TNPG), Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) to check the performance. 

Also tuning the hyper parameters of neural network architecture could help.

