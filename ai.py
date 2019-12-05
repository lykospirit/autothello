from run_game import *
from util import *
from keras.models import load_model
import keras.backend as K
import sys
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.initializers import RandomNormal, RandomUniform
from keras.optimizers import Adam
from keras.losses import mean_squared_error#, kullback_leibler_divergence
import numpy as np

class AI:
  def __init__(self, player):
    self.player = player

  def move(self, board, moves): # given board, makes a valid move
    valid_coords = get_valid_coords(board, self.player)
    assert(len(valid_coords) > 0)
    return valid_coords[np.random.choice(len(valid_coords), 1)[0]]

class VAE_AI(AI):
  def __init__(self, player, filename):
    super().__init__(player)
    # load model weights
    decoder_input  = Input(shape=(64,))
    decoder_hidden = Dense(64, activation='relu', name='hidden_dec')(decoder_input)
    decoder_output = Dense(60, activation='sigmoid', name='output')(decoder_hidden)
    self.model     = Model(inputs=decoder_input, outputs=decoder_output)
    self.model.load_weights('models/{}.h5'.format(filename), by_name=True)
    print("VAE_AI initialized with weights {}".format(filename))

  def move(self, board, coords):
    # run forward pass on 2500 random samples
    sample_size, vote_size = 2500, 250
    samples = np.random.normal(0.0, 1.0, (sample_size,64))
    decodes = self.model.predict(samples)
    assert(decodes.shape == (sample_size,60))

    # compute similarity to coords so far
    dists = []
    indices = coords_to_indices(coords)
    valid_coords = get_valid_coords(board, self.player)
    assert(len(valid_coords) > 0)
    for i in range(sample_size):
      dists.append(compute_mse(indices, decodes[i][:len(indices)]))            # for vae_1
      # dists.append(compute_cat_xentropy(indices, decodes[i][:len(indices)]))   # for vae_{2,3}

    # get 250 best samples and vote for move
    coords = np.zeros(64, dtype=np.int32)
    for i in range(vote_size):
      argmin = np.argmin(dists)
      x, y = indices_to_coords(np.array([decodes[argmin][len(indices)]]))[0]
      dists[argmin] = np.max(dists)
      if (x,y) in valid_coords:
        coords[x*8+y] += 1

    argmax = np.argmax(coords)
    if coords[argmax] != 0: # it was voted for at some point
      coord = (argmax//8, argmax%8)
      assert(coord in valid_coords)
      return coord

    # failsafe: return random valid coord
    return valid_coords[np.random.choice(len(valid_coords), 1)[0]]

class DQN_AI(AI):
  def __init__(self, player, filename):
    super().__init__(player)
    train_input = Input(shape=(64,))
    hidden1 = Dense(256, activation='relu', name='hidden1')(train_input)
    hidden2 = Dense(256, activation='relu', name='hidden2')(hidden1)
    output  = Dense(64, activation='tanh', name='output')(hidden2)
    self.model = Model(inputs=train_input, outputs=output)
    self.model.load_weights('models/{}.h5'.format(filename), by_name=True)
    print("DQN_AI initialized with weights {}".format(filename))

  def move(self, board, coords):
    epsilon = 0.05
    r = np.random.uniform(0,1)
    coords = get_valid_coords(board, self.player)
    if r < epsilon:
      return coords[np.random.randint(len(coords))]
    else:
      ids = list(map(lambda c: c[0]*8+c[1], coords))
      q_values = self.model.predict(np.array([board.flatten()]))[0]
      opt_id = ids[np.argmax(q_values[ids])]
      return (opt_id//8, opt_id%8)

if __name__ == "__main__":
  pass
