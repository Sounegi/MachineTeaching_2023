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

# Slate
  then Agent will represent the slate (list of K indices) from 'doc' to user

# Hierarchical Agent Layers

![image](https://github.com/Sounegi/MachineTeaching_2023/assets/67320090/29fe2188-ee1b-4d3d-9127-8890a513e5e7)
Can recursive the hierarchical agent with several base agents, the work of a hierarchical agent is to preprocess the observation data, and postprocessing the abstract output from base agents.
this can be used to design more complex agents that calculate several observed factors.

* for agent we can have kwarg
* like
   """TabularQAgent init.

    Args:
  - observation_space: a gym.spaces object specifying the format of observations.
  - action_space: a gym.spaces object that specifies the format of actions.
  - eval_mode: Boolean indicating whether the agent is in training or eval mode.
  - ignore_response: Boolean indicating whether the agent should ignore the response part of the observation.
  - discretization_bounds: pair of real numbers indicating the min and max value for continuous attributes discretization. Values below the min will all be grouped in the first bin, while values above the max will all be grouped in the last bin. See the documentation of numpy.digitize for further details.
  -  number_bins: positive integer number of bins used to discretize continuous attributes.
  -  exploration_policy: either one of ['epsilon_greedy', 'min_count'] or a custom function. TODO(mmladenov): formalize requirements of this function.
  -  exploration_temperature: a real number passed as parameter to the exploration policy.
  -  learning_rate: a real number between 0 and 1 indicating how much to update Q-values, i.e. Q_t+1(s,a) = (1 - learning_rate) * Q_t(s, a) + learning_rate * (R(s,a) + ...).
  -  gamma: real value between 0 and 1 indicating the discount factor of the MDP.
  -  ordinal_slates: boolean indicating whether slate ordering matters, e.g. whether the slates (1, 2) and (2, 1) should be considered different actions. Using ordinal slates increases complexity factorially.


      **kwargs: additional arguments like eval_mode.
    """
# Formula for UCB and out flashcard teaching
Retention Rate <br>
Arm selecting formula
