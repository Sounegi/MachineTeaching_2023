# MachineTeaching_2023
Special Topic Implementation with Professor Shan-Hung Wu

# Problem Definition
Implementing contextual Multi-Armed Bandit (cMAB) using Upper-Confidence Bound(UCB) in Recsim ()

My understanding

User - Have hidden feature + hidden preference(undecide), and learn whatever recommender gives in each time step.
Document - Static(most of the time) similar to the content of the exam that has a predecided scope.
Recommender Policy - UCB algorithm, with the difficult value change up to the individual user.



# How RecSIm work

RecSim mainly consists of 2 part
1. Environment
2. Agent

# Agent in RecSim

the main job of an Agent in RecSim is divided into 2 step
1. Observe the User and Document of Available choice(arm for Bandit) and user context
2. Calculate using the observed features to give the user a K-size slate from available choices for user to response

# Set Up
First, have to initiate the environment to illustrate what they produce and consume, and how they handle with agent.
-> Then the agent reset the environment at the start of each session (trigger the resampling of the user) 

# Observation
  A RecSim observation is a dictionary with 3 keys:
  1. 'user' ~ User observable feature
  2. 'doc' ~ Document observable feature -> determine by environment
  3. 'response' ~ User response

  ?Note that this environment does not implement user observable features, so that field would be empty at all times.

