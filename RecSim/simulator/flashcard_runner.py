import os

from recsim.simulator import runner_lib

#New runner without checkpoint
class FlashcardRunner(runner_lib.Runner):
  """Object that handles running the training.

  See main.py for a simple example to train an agent.
  """

  def __init__(self, max_training_steps=250000, num_iterations=100, **kwargs):
    '''tf.logging.info(
        'max_training_steps = %s, number_iterations = %s,'
        'checkpoint frequency = %s iterations.', max_training_steps,
        num_iterations, checkpoint_frequency)'''

    super(FlashcardRunner, self).__init__(**kwargs)
    self._max_training_steps = max_training_steps
    self._num_iterations = num_iterations

    self._output_dir = os.path.join(self._base_dir, 'train')

    self._set_up(eval_mode=False)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    #tf.logging.info('Beginning training...')
    start_iter = 0
    total_steps = 0

    for iteration in range(start_iter, self._num_iterations):
      #tf.logging.info('Starting iteration %d', iteration)
      total_steps = self._run_train_phase(total_steps)
      '''if iteration % self._checkpoint_frequency == 0:
        self._checkpoint_experiment(iteration, total_steps)'''

  def _run_train_phase(self, total_steps):
    """Runs training phase and updates total_steps."""

    self._initialize_metrics()

    num_steps = 0

    while num_steps < self._max_training_steps:
      episode_length, _ = self._run_one_episode()
      num_steps += episode_length

    total_steps += num_steps
    self._write_metrics(total_steps, suffix='train')
    return total_steps