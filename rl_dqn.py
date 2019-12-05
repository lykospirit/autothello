import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import collections
import sys
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.initializers import RandomNormal, RandomUniform, Zeros
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from run_game import *
from ai import AI

class ExperienceBuffer:
  def __init__(self, replay_size):
    self.buf = collections.deque([])
    self.max_size = replay_size
    self.buf_size = 0

  def sample(self, batch_size):
    inds = np.random.randint(len(self.buf), size=batch_size)
    return [self.buf[i] for i in inds]

  def append(self, tup):
    if self.buf_size >= self.max_size:
      self.buf.popleft()
    else:
      self.buf_size += 1
    self.buf.append(tup)

class DeepQ:
  def __init__(self, **kwargs):
    # Possible kwargs:
    # - lr:               learning rate
    # - batch_size:       batch size
    # - episodes:         # episodes for training
    # - hidden:           # of units in hidden layer
    # - discount_factor:  rate of reward decay over each step
    # - replay_size:      size of experience replay buffer
    # - sample_size:      # of samples from experience replay
    # - epsilon:          prob of random action in epsilon-greedy policy
    self.player = kwargs['player'] if 'player' in kwargs else 1
    self.lr = kwargs['lr'] if 'lr' in kwargs else 1e-4
    self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 32
    self.episodes = kwargs['episodes'] if 'episodes' in kwargs else 5000
    self.hidden_size = kwargs['hidden_size'] if 'hidden_size' in kwargs else 64
    self.disc_factor = kwargs['disc_factor'] if 'disc_factor' in kwargs else 0.95
    self.replay_init = kwargs['replay_init'] if 'replay_init' in kwargs else 25000
    self.replay_size = kwargs['replay_size'] if 'replay_size' in kwargs else 250000
    self.sample_size = kwargs['sample_size'] if 'sample_size' in kwargs else 32
    self.clone_steps = kwargs['clone_steps'] if 'clone_steps' in kwargs else 5000
    self.max_epsilon = kwargs['max_epsilon'] if 'max_epsilon' in kwargs else 1.0
    self.min_epsilon = kwargs['min_epsilon'] if 'min_epsilon' in kwargs else 0.1
    self.clone_count = 0
    self.expBuf = ExperienceBuffer(self.replay_size)

  def setup(self):
    # Create models
    kernel_init = RandomUniform(minval=-0.2165, maxval=0.2165)#RandomNormal(mean=0.0, stddev=1.0)
    bias_init = Zeros()#RandomNormal(mean=0.0, stddev=1.0)

    train_input = Input(shape=(64,))
    hidden1 = Dense(self.hidden_size, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='hidden1')(train_input)
    hidden2 = Dense(self.hidden_size, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='hidden2')(hidden1)
    # output  = Dense(64, activation='linear', kernel_initializer=kernel_init, bias_initializer=bias_init, name='output')(hidden2)
    output  = Dense(64, activation='tanh', kernel_initializer=kernel_init, bias_initializer=bias_init, name='output')(hidden2)

    target_input = Input(shape=(64,))
    target_hidden1 = Dense(self.hidden_size, activation='relu', name='target_hidden1')(target_input)
    target_hidden2 = Dense(self.hidden_size, activation='relu', name='target_hidden2')(target_hidden1)
    # target_output  = Dense(64, activation='linear', name='output')(target_hidden2)
    target_output  = Dense(64, activation='tanh', name='output')(target_hidden2)

    self.model = Model(inputs=train_input, outputs=output)
    self.target_model = Model(inputs=target_input, outputs=target_output)
    self.model.compile(optimizer=Adam(lr=self.lr), loss='mse')
    self.target_model.compile(optimizer=Adam(lr=self.lr), loss='mse')
    self.target_model.set_weights(self.model.get_weights())

    # Initialize experience buffer
    board = init_board()
    for sample in range(self.replay_init):
      coords = get_valid_coords(board, 1)
      action = coords[np.random.randint(len(coords))]
      next_board = board.copy()
      make_move(next_board, 1, action)
      reward, ended = 0, False
      while True: # run game until game is over or player 1 has moves
        coords = get_valid_coords(next_board, 2)
        if len(coords) > 0: # player 2 has moves: make a random one
          make_move(next_board, 2, coords[np.random.randint(len(coords))])
          coords = get_valid_coords(next_board, 1)
          if len(coords) > 0: break
        else: # player 2 has no moves
          coords = get_valid_coords(next_board, 1)
          if len(coords) == 0:  # neither player has moves: the game is over
            ended = True
            [blank, p1, p2] = evaluate_board(next_board)
            if p1 > p2: reward = 0.999999
            elif p1 < p2: reward = -0.999999
          break

      self.expBuf.append((board, action, reward, next_board))
      if (ended): board = init_board()
      else: board = next_board.copy()

  def get_epsilon_action(self, board, eps=None): # returns a coord
    if eps is None: epsilon = 0.05
    else: epsilon = self.min_epsilon + (self.max_epsilon-self.min_epsilon) * max(0.0,(1.0-(float(eps)/float(self.episodes//10))))
    # else: epsilon = self.min_epsilon + (self.max_epsilon-self.min_epsilon) * (1.0-(float(eps)/float(self.episodes//10))) # deepQ_p1_{1-5}
    r = np.random.uniform(0,1)
    coords = get_valid_coords(board, self.player)
    if r < epsilon:
      return coords[np.random.randint(len(coords))]
    else:
      ids = list(map(lambda c: c[0]*8+c[1], coords))
      q_values = self.model.predict(np.array([board.flatten()]))[0]
      opt_id = ids[np.argmax(q_values[ids])]
      return (opt_id//8, opt_id%8)

  def train(self, filename):
    print("Training {}...".format(filename))
    for eps in range(self.episodes):
      if eps % 100 == 99: print("Episode {}...".format(eps+1))
      self.board = init_board()
      done = False
      while not done:
        # Execute one step using epsilon action and store transition in buffer
        action = self.get_epsilon_action(self.board, eps)
        next_board = self.board.copy()
        make_move(next_board, 1, action)
        reward = 0
        while True: # run game until game is over or player 1 has moves
          coords = get_valid_coords(next_board, 2)
          if len(coords) > 0: # player 2 has moves: make a random one
            make_move(next_board, 2, coords[np.random.randint(len(coords))])
            coords = get_valid_coords(next_board, 1)
            if len(coords) > 0: break
          else: # player 2 has no moves
            coords = get_valid_coords(next_board, 1)
            if len(coords) == 0:  # neither player has moves: the game is over
              done = True
              [blank, p1, p2] = evaluate_board(next_board)
              if p1 > p2: reward = 0.999999
              elif p1 < p2: reward = -0.999999
            break
        self.expBuf.append((self.board, action, reward, next_board))
        self.board = next_board.copy()

        # SGD using samples from experience buffer
        experience = self.expBuf.sample(self.sample_size)
        states, targets = [], []
        for (b1,a,r,b2) in experience:
          ended = (len(get_valid_coords(b2,1)) == 0)
          if ended: target = r
          else:
            q_values = self.target_model.predict(np.array([b2.flatten()]))[0]
            target = r + self.disc_factor * np.max(q_values)
          states.append(b1.flatten())
          target_qs = self.model.predict(np.array([b1.flatten()]))[0]
          id = a[0]*8+a[1]
          target_qs[id] = np.clip(target, target_qs[id]-1.0, target_qs[id]+1.0)
          targets.append(target_qs)
        self.model.fit(np.array(states), np.array(targets), batch_size=self.batch_size, epochs=1, verbose=0)

        # Copy weights from model to target after self.clone_steps steps
        self.clone_count += 1
        if self.clone_count >= self.clone_steps:
          self.clone_count = 0
          self.target_model.set_weights(self.model.get_weights())

      if eps % 500 == 499:
        tmp = "models/{}_{}.h5".format(filename, eps+1)
        tmp_target = "models/{}_{}_target.h5".format(filename, eps+1)
        self.model.save_weights(tmp)
        self.target_model.save_weights(tmp_target)
        self.test()

  def test(self, filename=None, eps=None):
    if filename is not None:
      assert(eps is not None)
      print("Running {}_{}...".format(filename, eps))
      self.model.load_weights("models/{}_{}.h5".format(filename, eps))
      self.target_model.load_weights("models/{}_{}_target.h5".format(filename, eps))

    wins, losses, draws = 0, 0, 0
    for i in range(500):
      board = init_board()
      moves = []
      player = 1
      while True:
        if player == 1: move = self.get_epsilon_action(board)
        else:
          valid_coords = get_valid_coords(board, 2)
          move = valid_coords[np.random.choice(len(valid_coords), 1)[0]]
        x, y = move
        moves.append(move)
        make_move(board, player, move)
        player = 1 if player == 2 else 2
        if get_valid_coords(board, player) == []:
          player = 1 if player == 2 else 2
        if get_valid_coords(board, player) == []:
          break
      state = evaluate_board(board)
      if state[1] > state[2]: wins += 1
      elif state[1] < state[2]: losses += 1
      else: draws += 1
    print("DQN - Wins: {}, Losses: {}, Draws: {}".format(wins, losses, draws))

if __name__ == "__main__":
  # kwargs = {} # deepQ_p1_1
  # kwargs = {'clone_steps': 10000, 'lr': 2e-5, 'hidden_size': 128} # deepQ_p1_2
  # kwargs = {'clone_steps': 10000, 'lr': 1e-4, 'hidden_size': 256, 'disc_factor': 0.99} # deepQ_p1_3
  kwargs = {'clone_steps': 10000, 'lr': 0.00025, 'hidden_size': 256, 'disc_factor': 0.99} # deepQ_p1_4
  # kwargs = {'clone_steps': 10000, 'lr': 0.00025, 'hidden_size': 512, 'disc_factor': 0.99, 'replay_init': 50000, 'replay_size': 1000000} # deepQ_p1_5
  # kwargs = {'clone_steps': 10000, 'lr': 0.0005, 'hidden_size': 512, 'disc_factor': 0.99, 'replay_init': 25000, 'replay_size': 250000, 'episodes': 10000} # deepQ_p1_6
  # kwargs = {'clone_steps': 10000, 'lr': 0.0001, 'hidden_size': 512, 'disc_factor': 0.99, 'replay_init': 25000, 'replay_size': 250000, 'episodes': 10000} # deepQ_p1_7
  # kwargs = {'clone_steps': 10000, 'lr': 0.00005, 'hidden_size': 512, 'disc_factor': 0.99, 'replay_init': 25000, 'replay_size': 250000, 'episodes': 10000} # deepQ_p1_8
  # kwargs = {'clone_steps': 8000, 'lr': 0.0001, 'hidden_size': 512, 'disc_factor': 0.99, 'replay_init': 20000, 'replay_size': 200000, 'episodes': 10000} # deepQ_p1_9
  # kwargs = {'clone_steps': 10000, 'lr': 0.00025, 'hidden_size': 512, 'disc_factor': 0.99, 'replay_init': 50000, 'replay_size': 1000000} # deepQ_p1_10
  # kwargs = {'clone_steps': 10000, 'lr': 1e-4, 'hidden_size': 256, 'disc_factor': 0.99} # deepQ_p1_11
  # kwargs = {'clone_steps': 10000, 'lr': 5e-4, 'hidden_size': 256, 'disc_factor': 0.99} # deepQ_p1_12
  model = DeepQ(**kwargs)
  model.setup()
  # model.train("deepQ_p1_1")
  # model.train("deepQ_p1_2")
  # model.train("deepQ_p1_3")
  model.train("deepQ_p1_4")
  # model.train("deepQ_p1_5")
  # model.train("deepQ_p1_6")
  # model.train("deepQ_p1_7")
  # model.train("deepQ_p1_8")
  # model.train("deepQ_p1_9")
  # model.train("deepQ_p1_10")
  # model.train("deepQ_p1_11")
  # model.train("deepQ_p1_12")
  print("done")
