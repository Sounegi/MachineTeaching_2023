import functools
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# RecSim imports
from recsim import agent
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym
from recsim.simulator import runner_lib

class RandomAgent(agent.AbstractEpisodicRecommenderAgent):
  """Parent class for stackable agent layers."""

  def __init__(self, action_space, random_seed):
    super(RandomAgent, self).__init__(action_space)
    self._rng = np.random.RandomState(random_seed)

  def step(self,  reward, observation):
    del reward
    doc_obs = observation['doc']
    # Simulate a random slate
    num_documents = len(doc_obs)
    doc_ids = list(range(len(doc_obs)))
    self._rng.shuffle(doc_ids)
    slate = doc_ids[:self._slate_size]
    print(f"Number of available documents: {len(doc_ids)}")
    print('Recommended slate: %s' % slate)
    return slate
  
  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        pass

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        pass
