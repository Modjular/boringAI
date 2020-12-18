---
layout: default
title:  Final Report
---
## Video

## Project Summary
In Minecraft, certain tools mine through blocks faster than others (i.e. pickaxes destroy stone-blocks faster than shovels). We aim to train a Minecraft agent using deep reinforcement learning to dig to the end of a tunnel as fast as possible, by learning what tools are best for each material it encounters. Ideally, our AI will learn to use the proper tool to destroy the block, for maximum tunneling speed. For our prototype, we concentrated on the switching to the right tool. To do so, we limited our malmo environment to discrete movements and used dense rewards. For our finished project, we used sparse rewards based on time, and added durability values for each tool. Tools made out of diamond would have lower durability than tools made out of iron. Our goal is for the AI to learn to save the diamond tools for cases where it would mine faster than iron tools.  

## Approaches
In this section, we’ll explain our initial prototype. Problems we encountered. Why we switched to RLLib. Why continuous vs discrete?
<img src="assets/agent.png">


### Phase 1: Dense rewards and Discrete Movements (Prototype) 
For each mission, our agent is spawned with 9 blocks lined in front of him as seen in the picture above. To complete the mission, he has to reach the finish line by destroying the blocks in front of him, using one of the tools in the hotbar ( shovel or pickaxe).  

To train our agent, we are implementing a deep reinforcement learning algorithm, DQN from the PyTorch library. Since our AI is bulldozing through in a straight line, our action space is limited to choosing between a diamond pickaxe and diamond shovel. When the agent encounters a block in front of itself, it chooses a tool. Afterwards, the agent follows a command sequence of breaking the block and moving forward. The agent repeats this cycle until it has reached the end of the tunnel, which is denoted by the coal block. Our observation state includes the 3x3 block space surrounding the agent. 

In our prototype, we have 2 different tools (pickaxe and shovel) with 2 different blocks (dirt and stone). For our prototype’s rewarding function, we’re using dense rewards. We have direct rewards at each step (+10 for using the right tool, -10 for not), to make sure that the agent functions correctly. For future implementations, we will try to use sparse rewards to train the agent. 
```
if get_block_front(world_state) == 'dirt':
                if action_idx == 0: # switching to pickaxe
                    reward += -10
                else:
                    reward += 10
elif get_block_front(world_state) == 'stone':
                if action_idx == 1: #switching to shovel
                    reward += 10
                else:
                    reward += -10
```

### Phase 2: Sparse rewards and Continuous Movement 
For each mission, our agent is spawned with 9 blocks (3 blocks each of dirt, stone, and oak). The tools in the agent’s action space consists of a diamond pickaxe, diamond axe, and diamond shovel. 

We decided to switch to sparse rewards to increase the complexity of our problem, and promote learning for our AI. We switched to continuous movement in order to make better use of time in the calculation of the reward. With discrete movement, the AI would break a block within one hit regardless of its tool. However, with continuous movement, it would take multiple hits even with the optimal tool to break a block. This allows us to reward our AI based on how fast it digs through the tunnel.  

We used two different models to train our AI. For one model, we used PPOTrainer from rlLib and for the second model, we used DQNTrainer from rlLib. 

#### Proximal Policy Optimization (PPO)
PPO uses gradient descent to optimize a policy that aims to maximize the reward. Below is the pseudocode for the ppo algorithm. 
<img src="assets/ppo_algorithm.png" >  
image from https://spinningup.openai.com/en/latest/algorithms/ppo.html 

We chose PPO since it was easy to implement and performed fairly well. PPO starts off by randomly choosing actions in the action space. As it trains longer, it exploits rewards that it has already discovered. An example of this in our AI is when the agent learns that using a pickaxe to mine a stone block takes less time than using either an axe or shovel. This experience motivates the AI to choose this tool in its future encounters with a stone block, so that its reward is maximized.

#### Deep Q Network (DQN)
DQN combines Q-Learning with Deep Neural Networks to optimize a policy that aims to maximize the reward. Below is the pseudocode for the dqn algorithm. 
<img src="assets/dqn_algorithm.png" width="80%">  
image from https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

We chose DQN for our secondary model since we wanted to compare how well PPO performed. Since our DQN came from the same library as PPO, it was easy to implement. 

## Evaluation

### Quantitative
<img src="assets/BPM_Equation.png"> 


PPO vs DQN Returns \
<img src="assets/returns_PP0_21.png" width="45%"> <img src="assets/returns_rllib_dqn.png" width="45%">  

PPO vs DQN Tool Usage \
<img src="assets/toolstats_PPO_21.png" width="45%"> <img src="assets/toolstats_rllib_dqn.png" width="45%">  

PPO Durability Returns and Tool Usage \
<img src="assets/returns_durability.png" width="45%"> <img src="assets/toolstats_durability.png" width="45%">  

### Qualitative
Although our metrics are straightforward, our sanity check is changing tools in response to its environment. Qualitatively, we can check that it’s using the right tool (shovel) to dig through dirt. Because of the nature of our state, there are not many things we can qualitatively measure. But perhaps we will discover qualitative metrics as we progress and attempt to add more difficult states.

After incorporating time, our metric will be based on the time it takes for the agent to dig a 20 block straight tunnel through a mountain made up of different types of blocks. 

<img src="assets/steve2.png">

## References
RLlib - algorithms\
PyTorch - algorithms\
Tqdm - testing\
Matplotlib - graphing\
iMovie - video editing

Links\
https://spinningup.openai.com/en/latest/algorithms/ppo.html \
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
