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

class GreedyGainAgent(agent.AbstractEpisodicRecommenderAgent):
    """Agent for flashcard teaching, recommend falshcard that give max retention rate gain // fully exploit"""
    def __init__(self, 
                 action_space, 
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
        del reward
        #doc_gain = [] #keep gain of each flashcard
        doc_marg_gain = []
        for i in range(len(observation['doc'])):
            #doc_gain.append(self.find_doc_gain(observation['user']['time'], observation['user']['history'][i], observation['doc'][str(i)])) 
            doc_marg_gain.append(self.find_marginal_gain(len(observation['doc']),
                                                         i,
                                                         observation['user']['time'],
                                                         observation['user']['history'],
                                                         observation['doc']))

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
                                                         card_difficulty[str(n)][0])
                if(n != card_i):
                    card_i_gain += self.find_retention_rate(t+1, 
                                                         user_history[n][0], 
                                                         user_history[n][1],
                                                         user_history[n][2],
                                                         card_difficulty[str(n)][0])
                else:
                    card_i_gain += self.find_doc_gain(t+1, 
                                                      user_history[card_i], 
                                                      card_difficulty[str(card_i)])
        
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

#don't use anymore
""" 
class UCBGainAgent(GreedyGainAgent):
    #Agent for flashcard teaching, recommend flashcard from gain(i) and have epsilon-probability to explore
    def __init__(self, action_space, deadline = 1):
        Initialize greedy agent that select argmax(gain(i))

            Arg:
            deadline: time after the last review for user
        
        super(UCBGainAgent, self).__init__(action_space)
        self._alpha = 1
        self._beta = 1
        self._eta = 1
        self._difficulty = []
    def step(self, reward, observation):
        #calculate gain of each flascard and select maximum one
        #calculate gain of each flashcard
        difficulty = np.ones(3)
        doc_gain = [] #keep gain of each flashcard
        #update difficulty
        if(observation['response'] == None):
             for i in range(len(observation['doc'])): 
                 self._difficulty.append(observation['doc'][str(i)][0])
        
        for i in range(len(observation['doc'])): 
            if(observation['response'] != None):
                self.difficulty_grad_descent(observation, self._difficulty)
        
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

    def difficulty_grad_descent(self, observation, prev_diff):
        diff_loss = []
        new_diff = prev_diff
        #update every flashcard difficulty
        if(observation['response'][-1]):
            for i in range(len(observation['doc'])):
                diff_loss.append()
        else:
            for i in range(len(observation['doc'])):
                diff_loss.append()
        for j in range(len(observation['doc'])):
            new_diff[j] = difficulty[j] - self._eta*(self._beta*difficulty[i] + diff_loss[i])
        difficulty = new_diff
        return
#todo: add gradient descent of difficulty(make agent learn), determine alpha exploration
"""