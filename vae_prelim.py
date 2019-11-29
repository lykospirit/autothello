import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import sys
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.initializers import RandomNormal, RandomUniform
from keras.optimizers import Adam
from keras.losses import mean_squared_error, categorical_crossentropy#, kullback_leibler_divergence
from run_game import *

sys.setrecursionlimit(10000)

class VAE:
  def __init__(self, **kwargs):
    # Possible kwargs:
    # - lr:         learning rate
    # - batch_size: batch size
    # - epochs:     # epochs for training
    # - hidden:     # of units in hidden layer
    # - latent:     # of units in latent representation
    self.lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
    self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 16
    self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
    self.hidden = kwargs['hidden'] if 'hidden' in kwargs else 64
    self.latent = kwargs['latent'] if 'latent' in kwargs else 64

  def create_network(self):
    kernel_init = RandomNormal(mean=0.0, stddev=1.0)
    bias_init = RandomNormal(mean=0.0, stddev=1.0)

    train_input = Input(shape=(60,))
    hidden_enc  = Dense(self.hidden, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='hidden_enc')(train_input)
    latent_mean = Dense(self.latent, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='latent_mean')(hidden_enc)
    latent_cov  = Dense(self.latent, activation='sigmoid', kernel_initializer=kernel_init, bias_initializer=bias_init, name='latent_cov')(hidden_enc)

    def get_sample(args):
      latent_mean, latent_cov = args
      epsilon = K.random_normal(shape=(self.latent,), mean=0.0, stddev=1.0)
      return epsilon * K.exp(latent_cov) + latent_mean

    sample     = Lambda(get_sample, output_shape=(self.latent,))([latent_mean, latent_cov])
    hidden_dec = Dense(self.hidden, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='hidden_dec')(sample)
    output     = Dense(60, activation='sigmoid', kernel_initializer=kernel_init, bias_initializer=bias_init, name='output')(hidden_dec)

    def vae_loss(train_input, output):
      pred_loss = categorical_crossentropy(train_input, output) # mean squared error
      # KL divergence between N(0,I) and latent params (eqn 7 in (Doersch, 2016))
      kl_diverg = 0.5 * (K.sum(K.exp(latent_cov)) + K.sum(K.square(latent_mean)) - self.latent - K.sum(latent_cov))
      return pred_loss + kl_diverg

    self.model = Model(inputs=train_input, outputs=output)
    self.model.compile(optimizer=Adam(lr=self.lr), loss=vae_loss)

  def load_data(self):
    self.data = {"train_X": [], "valid_X": [], "test_X": []}
    with open("othello_database/player1.txt", "rb") as f:
      for i,line in enumerate(f): # player1.txt has 53207 lines
        indices = [float(i.decode('UTF-8')) for i in line.strip().split()]
        assert(len(indices) == 60)
        if (i<40000): self.data["train_X"].append(indices)
        elif (i<45000): self.data["valid_X"].append(indices)
        else: self.data["test_X"].append(indices)

  def train(self, filename):
    print("Running {}...".format(filename))
    train_X = np.array(self.data["train_X"])
    valid_X = np.array(self.data["valid_X"])

    history = self.model.fit(train_X, train_X, epochs=self.epochs, batch_size=self.batch_size, validation_data=(valid_X, valid_X))

    trainLosses_filename = "results/{}_trainLosses.txt".format(filename)
    validLosses_filename = "results/{}_validLosses.txt".format(filename)
    np.savetxt(trainLosses_filename, history.history['loss'], delimiter=',')
    np.savetxt(validLosses_filename, history.history['val_loss'], delimiter=',')

    self.model.save_weights("models/{}.h5".format(filename))

if __name__ == "__main__":
  # kwargs = {'lr': 1e-3, 'batch_size': 16, 'epochs': 100}     ###     vae_1: loss - MSE
  # kwargs = {'lr': 1e-3, 'batch_size': 16, 'epochs': 100}     ###     vae_2: loss - Cat x-entropy
  kwargs = {'lr': 1e-3, 'batch_size': 16, 'epochs': 100}     ###     vae_3: loss - Cat x-entropy + KL divg
  model = VAE(**kwargs)
  model.create_network()
  model.load_data()
  model.train("vae_3")
  print("done")
