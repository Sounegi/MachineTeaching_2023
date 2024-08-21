import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

#@title RecSim imports
from recsim import document
from recsim import user
from recsim import agent
from recsim.choice_model import MultinomialLogitChoiceModel
from recsim.simulator import environment
from recsim.simulator import recsim_gym

# diasble eager execution to avoid error
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from agents import flashcard_agents
import flashcard_environment

from recsim.simulator import runner_lib

#Set up Environment
slate_size = 1
num_candidates = 10

def reward(responses):
  reward = 0.0
  for response in responses:
    reward += int(response._recall)
  return reward

def update_metrics(responses, metrics, info):
  # print("responses: ", responses)
  prs = []
  for response in responses:
    prs.append(response['pr'])
  if type(metrics) != list:
    metrics = [prs]
  else:
    metrics.append(prs)
  # print(metrics)
  return metrics

ltsenv = environment.Environment(
  FlashcardUserModel(slate_size),
  FlashcardDocumentSampler(),
  num_candidates,
  slate_size,
  resample_documents=False)

lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, reward, update_metrics)
lts_gym_env.reset()

#Set up Agent
def create_greedy_agent(sess, environment, eval_mode, summary_writer=None):
  return GreedyGainAgent(environment.action_space, 60)


#Run Experiment
tmp_base_dir = '/tmp/recsim/'
runner = runner_lib.TrainRunner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_greedy_agent,
    env=lts_gym_env,
    episode_log_file="",
    max_training_steps=5,
    num_iterations=2
)

runner.run_experiment()