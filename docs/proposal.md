---
layout: default
title:  Proposal
---

## Summary of the Project 
Our project is to build an AI whose goal is to dig a straight tunnel made up of different types of blocks in the fastest manner. Since certain tools mine through blocks faster than others, our AI will learn to use the proper tool for maximum efficiency. For example shovels are more efficient for dirt and pickaxes are good for stone. The actions that the AI can perform are moving forward, digging, and accessing tools. The AI is rewarded when it breaks a block and punished for each second it takes to break per block. This reward system will encourage the AI to learn different tools to help break blocks faster.   

## AI/ML Algorithms 
Some algorithms we will be enforcing and exploring more in depth are reinforced learning, Qlearning, prioritized experience replay, and distributed DQN.

## Evaluation Plan 
### Quantitative
Our metric is how long it takes for the agent to dig a 20 block straight tunnel through a mountain made up of different types of blocks. Since the digging speed is controlled by: the type of block, the item currently wielded, and other mining penalities (underwater). Our baseline will be reaching the end of the mountain faster than how long it takes to dig by hand under constant mining penalties. For example shovels are more efficient for dirt and pickaxes are good for stone. 

### Qualitative
Our sanity check would be seeing that the agent’s time to complete the mission is improving and it’s changing tools. Qualitatively, we can check that it’s using the right tool (shovel) to dig through dirt. Our moon shot would be adding a dimension so that the agent has a choice to circle around tough blocks instead of bulldozing through and avoiding lava blocks.


## Appointment with the Instructor 
Time: October 22, 9:30 AM

## Internal Meetings
Minimum: Thursday 8pm - 11pm (weekly)
