# Summerize of Abstract class, method, and agent that Recsim provides
+ **AbstractRecommenderAgent** => base class for the basic agent <br>
  Method in class
  1. set_state
  2. update
  Inherited Class
  - **AbstractEpisodicRecommenderAgent** => for episodic recommending task
    - **AbstractMultiUserEpisodicRecommenderAgent** => episodic for multiuser
  - **AbstractHirerachicalAgentLayer** => for recursive agent building
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
