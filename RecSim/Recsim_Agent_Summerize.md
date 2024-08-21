# Summerize of Abstract class, method, and agent that Recsim provides
## Base Class for Agent
+ **AbstractRecommenderAgent** => base class for the basic agent <br>
  Inherited Class
  - **AbstractEpisodicRecommenderAgent** => for episodic recommending task
    - **AbstractMultiUserEpisodicRecommenderAgent** => episodic for multiuser
  - **AbstractHirerachicalAgentLayer** => for recursive agent building <br>
    Helper Class
    - **AbstractClickBanditLayer** => use to mix base layer
    - **SufficientStatisticLayer** => use to log user response & agent recommend on different cluster of doc.
      - **FixedLenghtHistoryLayer** => record data with fixed length
      - **ClusterClickStatsLayer** => collect cluster click(which pass to agent), and impression count(how many time this cluster getting recommended)
+ **MABAlgorithm** => base class for Multi-Armed N=Bandit agent <br>
    Inherited Class
    - **UCB1**
    - **KLUCB** => Kullback-Leibler Upper Confidence Bounds (KL-UCB) algorithm.
    - **ThompsonSampling**
+ **GLMAlgorithm** => base class for Generalized Linear Model Bandit algorithm agent <br>
    Inherited Class
    - **GLM_UCB**
    - **GLM_TS**
+ **DQNAgentRecsim** => Recsim-specify Dopamine DQN <br>
more about [Dopamine DQN](https://github.com/google/dopamine) <br>
special method and class for DQN
  - **DQNNetworkType** => call for q_value of DQN
  - **ResponseAdapter** => custom flattening of response for DQN
  - **ObservationAdapter** => convert doc. for DQN
  - recsim_dqn_network => get q_value from DQN Network
  - wrapped_replay_buffer => buffering
## Method & Function for Agent
+ **GymSpaceWalker** => use to flatten doc that come in gym_space, usually used by recursive agent
+ **epsilon-greedyexploration** => compute epsilon greedy explore
+ **min_count** => find slate with least being selected
## Pre-made Agent (Ready to Use)
+ **ClusterBanditAgent**(AstractClickBanditLayer) <br>
  recommend item with the highest UCBs of topic affinity (don't know user interest), pick best Topic -> pick best doc. in that cluster
+ **GreedyClusterAgent**(AbstractEpisodicRecommenderAgent) <br>
  sorting all doc. of _**A Topic**_ by doc.'s quality
+ **FullSlateQAgent**(DQN + AbstractEpisodicRecommenderAgent) <br>
  apply standard Q-learning. more about [Q-Learning](https://towardsdatascience.com/a-beginners-guide-to-q-learning-c3e2a30a653c) or [Q-learning_alt](https://hackmd.io/@shaoeChen/Bywb8YLKS/https%3A%2F%2Fhackmd.io%2F%40shaoeChen%2FSyqVopoYr)
+ **SlateDecompQAgent**(DQN + AbstractEpisodicRecommenderAgent) <br>
  apply Slate-Q with the decomposition of slate. [paper](https://arxiv.org/abs/1905.12767), [SlateQ](https://medium.com/analytics-vidhya/slateq-a-scalable-algorithm-for-slate-recommendation-problems-735a1c24458c)
+ **GreedyPCTRAgent**(AbstractEpisodicRecommenderAgent) <br>
  recommend slate with the highest pCTR
+ **RandomAgent**(AbstractEpisodicRecommenderAgent) <br>
  recommend random slate
+ **TabularQAgent**(AbstractEpisodicRecommenderAgent) <br>
  apply tabular Q-learning. [Medium](https://medium.com/analytics-vidhya/slateq-a-scalable-algorithm-for-slate-recommendation-problems-735a1c24458c), [LevelupCoding](https://levelup.gitconnected.com/tabular-q-learning-a-prominent-reinforcement-learning-rl-algorithm-db364fe2d474?gi=21a4351e098a)

## Choice Model
*Abstract classes that encode a user's state and dynamics. <br>
