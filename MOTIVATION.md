# Multi Agent Reinforcement Learning for Dense Path Planning

This project was built to conduct research on Navigation among multiple cars (agents) on a NxN square grid using 
Reinforcement Learning.

The environment is a block which is a NxN grid. Numbering for the boxes is done in top down fashion. For example, in a 3x3
environment, visually, the numbering for the boxes would be
                                
                                1 4 7
                                2 5 8
                                3 6 9
                                
This problem statement involves solving the Navigation of multiple agents (agents are cars in the environment) having 
different destinations all contained within the environment. This problem is unique in itself as it involves 
coordination among the agents so that all of them reach their goal using the most optimized path available, in a 2d grid.

In this experiment, we have used two Q-value based approaches to solve the above problem. 

1 - Q-learning

2 - DQN

A few problems that occurred while running the experiments were :

Deadlocks --> A Deadlock is said to occur when 2 or more cars get in each others optimal paths. This results in 
untrained states which are unsolvable with the current reward function.

Generalization --> 
   


