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
      - **FixedLenghtHistoryLayer** => record data with fixed lenght
      - **ClusterClickStatsLayer** => collect cluster click(which pass to agent), and impression count(how many time this cluster getting recommended)
+ **MABAlgorithm** => base class for Multi-Armed N=Bandit agent <br>
    Inherited Class
    - **UCB1**
    - **KLUCB**
    - **ThompsonSampling**
+ **GLMAlgorithm** => base class for Generalized Linear Model Bandit algorithm agent <br>
    Inherited Class
    - **GLM_UCB**
    - **GLM_TS**
+ **DQNAgentRecsim** => Recsim-specify Dopamine DQN <br>
more about [Dopamine DQN](https://github.com/google/dopamine)
## Method & Function for Agent
+ **GymSpaceWalker** => use to flatten doc that come in gym_space, usually used by recursive agent
+ **epsilon-greedyexploration** => compute epsilon greedy explore
+ **min_count** => find slate with least being selected

