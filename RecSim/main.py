import tensorflow as tf
from agents.flashcard_agents import *
# from recsim.simulator import environment
from environment import environment
from user import FlashcardUserModel
from document import FlashcardDocumentSampler
from simulator.flashcard_runner import FlashcardRunner
from recsim.simulator import recsim_gym
from recsim.agents import full_slate_q_agent
from recsim.simulator import runner_lib
from agents import *
from util import reward, update_metrics

########################################################

#set parameter
slate_size = 2
num_candidates = 10
time_budget = 20
eval_delay_time = 10

#necessary to make runner run normally
tf.compat.v1.disable_eager_execution()

#initialize agent for flashcard recommendation
flashcard_agent = create_agent_helper(GreedyMarginGainAgent, 
  deadline = time_budget+eval_delay_time)


#initialize environment to simulate flashcard learning
ltsenv = environment.DocAccessibleEnvironment(
  FlashcardUserModel(num_candidates, time_budget, 
    slate_size, eval_delay_time=eval_delay_time, seed=0, sample_seed=0),
  FlashcardDocumentSampler(seed=0),
  num_candidates,
  slate_size,
  resample_documents=False)

#Set environment for simulator & reset to resample cards
lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, reward, update_metrics)
lts_gym_env.reset()

#set directory for saving a simulation log and result
tmp_base_dir = './recsim/'

#put everything into simulation runner
runner = FlashcardRunner(
    base_dir=tmp_base_dir,
    create_agent_fn=flashcard_agent,
    env=lts_gym_env,
    max_training_steps=10, #to make sure runner simulation round = num_iteration
    num_iterations=1,
)

#run simulation
runner.run_experiment()
