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
import tensorflow as tf
import numpy as np

class GreedyGainAgent(agent.AbstractEpisodicRecommenderAgent):
    """Agent for flashcard teaching, recommend falshcard that give max retention rate gain // fully exploit"""
    def __init__(self, 
                 sess, observation_space, action_space, eval_mode, summary_writer=None, 
                 deadline = 1):
        """Initialize greedy agent that select argmax(gain(i))

            Arg:
            deadline: time after the last review for user
        """
        super(GreedyGainAgent, self).__init__(action_space)
        self._deadline = deadline
    def step(self, reward, observation):
        """calculate gain of each flascard and select maximum one"""
        #calculate gain of each flashcard
        #difficulty = observation['doc'][];
        doc_gain = [] #keep gain of each flashcard
        for i in range(len(observation['doc'])):
            doc_gain.append(self.find_doc_gain(observation['user']['time'], 
                                          observation['user']['history'][i], 
                                          observation['doc'][str(i)])) 
        #find the best
        best_flashcard = doc_gain.index(max(doc_gain))
        #return best_gain
        
        return [best_flashcard]
    def find_doc_gain(self, current_time, user_history, card_difficulty):
        #calculate gain(i) of flashcard
        current_RR = self.find_retention_rate(self._deadline - current_time, 
                                         user_history[0], 
                                         user_history[1], 
                                         user_history[2], 
                                         card_difficulty)
        next_pos_RR = self.find_retention_rate(self._deadline - current_time, 
                                         user_history[0]+1, 
                                         user_history[1]+1, 
                                         user_history[2], 
                                         card_difficulty)
        next_neg_RR = self.find_retention_rate(self._deadline - current_time, 
                                         user_history[0]+1, 
                                         user_history[1], 
                                         user_history[2]+1, 
                                         card_difficulty)
        
        return 1/2*(next_pos_RR+next_neg_RR)-current_RR
    def find_retention_rate(self, delta_t, n_sum, n_pos, n_neg, difficulty):
        #calculate retention rate
        return np.exp(-(delta_t)/ np.exp(difficulty[0]*n_sum + difficulty[1]*n_pos + difficulty[2]*n_neg))

class GreedyMarginGainAgent(agent.AbstractEpisodicRecommenderAgent):
    """Agent for flashcard teaching, recommend falshcard that give max retention rate gain // fully exploit"""
    def __init__(self, 
                 sess, observation_space, action_space, eval_mode, summary_writer=None, 
                 deadline = 1):
        """Initialize greedy agent that select argmax(gain(i))

            Arg:
            deadline: time after the last review for user
        """
        super(GreedyMarginGainAgent, self).__init__(action_space)
        self._num_candidates = int(action_space.nvec[0])
        self._deadline = deadline
        np.random.seed(1)
        self._W = np.zeros((self._num_candidates,3))
        self._W[:, 0] = 1
        self._W[:, 1] = np.random.uniform(1.5, 5, self._num_candidates)
        self._W[:, 2] = np.random.uniform(0.75, 2.5, self._num_candidates)

        
    def step(self, reward, observation):
        """calculate gain of each flascard and select maximum one"""
        #calculate gain of each flashcard
        #difficulty = observation['doc'][];
        del reward
        #doc_gain = [] #keep gain of each flashcard
        #set random difficulty
        #self._W = observation['doc']
        doc_marg_gain = []

        for i in range(len(observation['doc'])):
            #doc_gain.append(self.find_doc_gain(observation['user']['time'], observation['user']['history'][i], observation['doc'][str(i)])) 
            doc_marg_gain.append(self.find_marginal_gain(len(observation['doc']),
                                                         i,
                                                         observation['user']['time'],
                                                         observation['user']['history'],
                                                         self._W))

        #find the best
        best_flashcard = doc_marg_gain.index(max(doc_marg_gain))
        #doc_gain.index(max(doc_gain))
        #return best_gain
        
        return [best_flashcard]
    
    def find_marginal_gain(self, num_cards, card_i, current_time, user_history, card_difficulty):
        current_gain = 0
        card_i_gain = 0
        for n in range(num_cards):
            for t in range(self._deadline - current_time):
                current_gain += self.find_retention_rate(t+1, 
                                                         user_history[n][0], 
                                                         user_history[n][1],
                                                         user_history[n][2],
                                                         card_difficulty[n])
                if(n != card_i):
                    card_i_gain += self.find_retention_rate(t+1, 
                                                         user_history[n][0], 
                                                         user_history[n][1],
                                                         user_history[n][2],
                                                         card_difficulty[n])
                else:
                    card_i_gain += self.find_doc_gain(t+1, 
                                                      user_history[card_i], 
                                                      card_difficulty[card_i])
        
        current_gain = (1/(num_cards*(self._deadline - current_time+1)))*current_gain
        card_i_gain = (1/(num_cards*(self._deadline - current_time+1)))*card_i_gain

        return card_i_gain-current_gain
    
    def find_doc_gain(self, current_time, user_history, card_difficulty):
        #calculate gain(i) of flashcard
        '''
        current_RR = self.find_retention_rate(current_time, 
                                         user_history[0], 
                                         user_history[1], 
                                         user_history[2], 
                                         card_difficulty)
        '''
        next_pos_RR = self.find_retention_rate(current_time, 
                                         user_history[0]+1, 
                                         user_history[1]+1, 
                                         user_history[2], 
                                         card_difficulty)
        next_neg_RR = self.find_retention_rate(current_time, 
                                         user_history[0]+1, 
                                         user_history[1], 
                                         user_history[2]+1, 
                                         card_difficulty)
        
        return 1/2*(next_pos_RR+next_neg_RR)
    def find_retention_rate(self, delta_t, n_sum, n_pos, n_neg, difficulty):
        #calculate retention rate
        return np.exp(-(delta_t)/ np.exp(difficulty[0]*n_sum + difficulty[1]*n_pos + difficulty[2]*n_neg))
    def begin_episode(self, observation=None):
        docs = observation['doc']
        user = observation['user']

        self._deadline = user['time_budget']


        self._episode_num += 1
        return self.step(0, observation)

class RandomAgent(agent.AbstractEpisodicRecommenderAgent):
  """Parent class for stackable agent layers."""

  def __init__(self, sess, observation_space, action_space, eval_mode, summary_writer=None, random_seed = 0):
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

  
class UCBAgent(agent.AbstractEpisodicRecommenderAgent):
  def __init__(self, sess, observation_space, action_space, eval_mode, 
      eval_delay_time=0, alpha=1.0, learning_rate=0.001, summary_writer=None):
    super(UCBAgent, self).__init__(action_space, summary_writer)
    self._num_candidates = int(action_space.nvec[0])
    np.random.seed(1)
    initial_W = np.zeros((self._num_candidates,3))
    initial_W[:, 0] = 1
    initial_W[:, 1] = np.random.uniform(1.5, 5, self._num_candidates)
    initial_W[:, 2] = np.random.uniform(0.75, 2.5, self._num_candidates)
    #self._W = tf.Variable(np.random.uniform(0, 5, size=(self._num_candidates, 3)), name='W')
    self._W = tf.Variable(initial_W, name= 'W')
    self._sess = sess
    self._return_idx = None
    self._prev_pred_pr = None
    self._opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    self._alpha = alpha
    self._deadline = None
    self._eval_delay_time = eval_delay_time # eval at T + s

    assert self._slate_size == 1
  def begin_episode(self, observation=None):
    docs = observation['doc']
    user = observation['user']

    self._deadline = user['time_budget']
    '''
    if 'W' in user:
      assign = self._W.assign(user['W'])
      self._sess.run(assign)
    else:
      w = []
      for doc_id in docs:
        w.append(docs[doc_id])
      w = np.array(w).reshape((-1, 3))
      #print("observe from docs")
      assign = self._W.assign(w)
      self._sess.run(assign)
      #print(self._W.eval(session=self._sess))
    '''
    self._episode_num += 1
    return self.step(0, observation)
  
  def step(self, reward, observation):
    docs = observation['doc']
    user = observation['user']
    response = observation['response']

    if self._return_idx != None and response != None:
      # update w
      y_true = [response[0]['recall']]
      y_pred = self._prev_pred_pr
      loss = tf.losses.binary_crossentropy(y_true, y_pred)
      self._sess.run(self._opt.minimize(loss))
    base_pr = self.calc_prs(user['time'], user['last_review'], user['history'], self._W)

    time = user['time'] + 1
    history_pos = user['history'].copy()
    history_pos[:, [0, 1]] += 1 # add n, n+ by 1
    history_neg = user['history'].copy()
    history_neg[:, [0, 2]] += 1 # add n, n- by 1
    last_review_now = np.repeat(user['time'], len(user['last_review']))

    # always evaluate at deadline + eval_delay_time
    eval_time = self._deadline + self._eval_delay_time
    pr_pos = self.calc_prs(eval_time, last_review_now, history_pos, self._W)
    pr_neg = self.calc_prs(eval_time, last_review_now, history_neg, self._W)

    gain = (pr_pos + pr_neg) / 2 - base_pr
    time_since_last_review = user['time'] - user['last_review']
    uncertainty = self._alpha * tf.math.sqrt(tf.math.log(time_since_last_review) / user['history'][:, 0])
    # print(gain.eval(session=self._sess))
    # print(time_since_last_review)
    # print(uncertainty.eval(session=self._sess))
    ucb_score = gain + uncertainty
    #print("       gain:", gain.eval(session=self._sess))
    #print("uncertainty:", uncertainty.eval(session=self._sess))
    best_idx = tf.argmax(ucb_score)

    self._return_idx = self._sess.run(best_idx)
    self._prev_pred_pr = base_pr[self._return_idx]
    return [self._return_idx]

    
  def calc_prs(self, train_time, last_review, history, W):
    last_review = train_time - last_review
    mem_param = tf.math.exp(tf.reduce_sum(history * W, axis=1))
    pr = tf.math.exp(-last_review / mem_param)
    return pr

class UCBGainAgent(GreedyGainAgent): #not complete
    """Agent for flashcard teaching, recommend flashcard from gain(i) and have epsilon-probability to explore"""
    def __init__(self, action_space, deadline = 1):
        """Initialize greedy agent that select argmax(gain(i))

            Arg:
            deadline: time after the last review for user
        """
        super(UCBGainAgent, self).__init__(action_space)
        self._alpha = 1
        self._beta = 1
        self._eta = 1
        self._difficulty = []
    def step(self, reward, observation):
        """calculate gain of each flascard and select maximum one"""
        #calculate gain of each flashcard
        difficulty = np.ones(3)
        doc_gain = [] #keep gain of each flashcard
        #update difficulty
        if(observation['response'] == None):
             for i in range(len(observation['doc'])): 
                 self._difficulty.append(observation['doc'][str(i)][0])
        
        for i in range(len(observation['doc'])): 
            #apply gradient descent on card difficulty
            if(observation['response'] != None):
                self.difficulty_grad_descent(observation['user']['last_review'][i], 
                                             observation['user']['history'][i], 
                                             self._difficulty[i])
        
            doc_gain.append(self.find_doc_gain(observation['user']['time'], 
                                          observation['user']['history'][i], 
                                          self._difficulty[i])
                           + self.add_epsilon_explore(observation['user']['last_review'][i],
                                                observation['user']['history'][i],
                                                self._alpha))
        #find the best
        best_flashcard = doc_gain.index(max(doc_gain))
        #return best_gain
        return [best_flashcard]
    def add_epsilon_explore(self, last_t, card_history, alpha):
        return self._alpha*np.sqrt(np.log(last_t)/card_history)

    def difficulty_grad_descent(self, last_review, user_history, prev_diff):
        diff_loss = []
        new_diff = prev_diff
        #update difficulty of flashcard (i)
        '''
        if(last respone == this card == 1)
            loss = - s.t.
        '''
        '''
        if(observation['response'][-1]):
            for i in range(len(observation['doc'])):
                diff_loss.append()
        else:
            for i in range(len(observation['doc'])):
                diff_loss.append()
        for j in range(len(observation['doc'])):
            new_diff[j] = difficulty[j] - self._eta*(self._beta*difficulty[i] + diff_loss[i])
        difficulty = new_diff
        '''
        return
#todo: apply grad, calculate loss

class NewAgentTemplate(agent.AbstractEpisodicRecommenderAgent):
    def __init__(self, 
                 sess, observation_space, action_space, eval_mode, summary_writer=None, 
                 deadline = 1):
        """Initialize greedy agent that select argmax(gain(i))

            Arg:
            deadline: time after the last review for user
        """
        super(NewAgentTemplate, self).__init__(action_space)
        self._num_candidates = int(action_space.nvec[0])
        self._deadline = deadline #deadline = time when user be evaluate
        #initialize your own parameter here
        

    def step(self,  reward, observation):
        #select the recommend flashcard and return here
        return
  
