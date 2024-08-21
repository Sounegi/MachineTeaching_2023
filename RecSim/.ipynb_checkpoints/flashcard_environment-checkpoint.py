#@title Generic imports
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

'''
Flashcard
'''
class FlashcardDocument(document.AbstractDocument):
  def __init__(self, doc_id, difficulty):
    self.base_difficulty = difficulty
    # doc_id is an integer representing the unique ID of this document
    super(FlashcardDocument, self).__init__(doc_id)

  def create_observation(self):
    return np.array(self.base_difficulty)

  @staticmethod
  def observation_space():
    return spaces.Box(shape=(1,3), dtype=np.float32, low=0.0, high=1.0)

  def __str__(self):
    return "Flashcard {} with difficulty {}.".format(self._doc_id, self.base_difficulty)

class FlashcardDocumentSampler(document.AbstractDocumentSampler):
  def __init__(self, doc_ctor=FlashcardDocument, **kwargs):
    super(FlashcardDocumentSampler, self).__init__(doc_ctor, **kwargs)
    self._doc_count = 0

  def sample_document(self):
    doc_features = {}
    doc_features['doc_id'] = self._doc_count
    doc_features['difficulty'] = self._rng.random_sample((1, 3))
    self._doc_count += 1
    return self._doc_ctor(**doc_features)

'''
User
'''
class UserState(user.AbstractUserState):
  def __init__(self, num_candidates, time_budget):
    self._cards = num_candidates
    self._history = np.zeros((num_candidates, 3))
    self._last_review = np.zeros((num_candidates,))
    self._time_budget = time_budget
    self._time = 0
    self._W = np.zeros((num_candidates, 3))
    super(UserState, self).__init__()
  def create_observation(self):
    return {'history': self._history, 'last_review': self._last_review, 'time': self._time, 'time_budget': self._time_budget}

  @staticmethod
  def observation_space():
    return spaces.Dict({
        'history': spaces.Box(shape=(num_candidates, 3), low=0, high=np.inf, dtype=int),
        'last_review': spaces.Box(shape=(num_candidates,), low=0, high=np.inf, dtype=int),
        'time': spaces.Box(shape=(1,), low=0, high=np.inf, dtype=int),
        'time_budget': spaces.Box(shape=(1,), low=0, high=np.inf, dtype=int),
    })

  def score_document(self, doc_obs):
    return 1

class UserSampler(user.AbstractUserSampler):
  _state_parameters = {'num_candidates': num_candidates, 'time_budget': 60}
  def __init__(self,
               user_ctor=UserState,
               **kwargs):
    # self._state_parameters = {'num_candidates': num_candidates}
    super(UserSampler, self).__init__(user_ctor, **kwargs)


  def sample_user(self):
    return self._user_ctor(**self._state_parameters)

'''
Response
'''
class UserResponse(user.AbstractResponse):
  def __init__(self, recall=False, pr=0):
    self._recall = recall
    self._pr = pr

  def create_observation(self):
    return {'recall': int(self._recall), 'pr': self._pr}

  @classmethod
  def response_space(cls):
    # return spaces.Discrete(2)
    return spaces.Dict({'recall': spaces.Discrete(2), 'pr': spaces.Box(low=0.0, high=1.0)})

'''
Reward
'''
from datetime import datetime
def eval_result(train_time, last_review, history, W):
  with open(f"{datetime.now()}.txt", "w") as f:
    print(train_time, file=f)
    print(last_review, file=f)
    print(history, file=f)
    print(W, file=f)
    # np.einsum('ij,ij->i', a, b)
    last_review = train_time - last_review
    mem_param = np.exp(np.einsum('ij,ij->i', history, W))
    pr = np.exp(-last_review / mem_param)
    print(pr, file=f)
    print(pr)
    print("score:", np.sum(pr) / pr.shape[0], file=f)
    print("score:", np.sum(pr) / pr.shape[0])

class FlashcardUserModel(user.AbstractUserModel):
  def __init__(self, slate_size, seed=0):
    super(FlashcardUserModel, self).__init__(
        UserResponse, UserSampler(
            UserState, seed=seed
        ), slate_size)
    self.choice_model = MultinomialLogitChoiceModel({})

  def is_terminal(self):
    terminated = self._user_state._time > self._user_state._time_budget
    if terminated: # run evaluation process
      eval_result(self._user_state._time,
                  self._user_state._last_review.copy(),
                  self._user_state._history.copy(),
                  self._user_state._W.copy())
    return terminated

  def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
      doc_id = doc._doc_id
      self._user_state._history[doc_id][0] += 1
      if response._recall:
        self._user_state._history[doc_id][1] += 1
      else:
        self._user_state._history[doc_id][2] += 1
      self._user_state._last_review[doc_id] = self._user_state._time
    self._user_state._time += 1

  def simulate_response(self, slate_documents):
    responses = [self._response_model_ctor() for _ in slate_documents]
    # Get click from of choice model.
    self.choice_model.score_documents(
      self._user_state, [doc.create_observation() for doc in slate_documents])
    scores = self.choice_model.scores
    selected_index = self.choice_model.choose_item()
    # Populate clicked item.
    self._generate_response(slate_documents[selected_index],
                            responses[selected_index])
    return responses

  def _generate_response(self, doc, response):
    # W = np.array([1,1,1])
    doc_id = doc._doc_id
    W = self._user_state._W[doc_id]
    if not W.any(): # uninitialzed
      self._user_state._W[doc_id] = W = doc.base_difficulty + np.random.uniform(-1, 1, (1, 3)) # a uniform error for each user
      print(W)
    # use exponential function to simulate whether the user recalls
    last_review = self._user_state._time - self._user_state._last_review[doc_id]
    x = self._user_state._history[doc_id]

    pr = np.exp(-last_review / np.exp(np.dot(W, x))).squeeze()
    print(f"time: {self._user_state._time}, reviewing flashcard {doc_id}, recall rate = {pr}")
    if np.random.rand() < pr: # remembered
      response._recall = True
    response._pr = pr
      