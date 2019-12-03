from keras.models import Model
from keras.layers import Input, Dense, LSTM
import keras.backend as K
import numpy as np 
import os.path



def get_seq2seq_model_input(indices):
    encoder_in = np.zeros((self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
    decoder_in = np.zeros((self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
    decoder_in[0, 65]=1
    for t, index in enumerate(indices):
        encoder_in[t, int(index*64)] = 1
    return encoder_in, decoder_in

class Seq2Seq:
    def __init__(self, **kwargs):
        self.lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 16
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        self.hidden = kwargs['hidden'] if 'hidden' in kwargs else 64
        self.latent = kwargs['latent'] if 'latent' in kwargs else 64
        self.num_encoder_tokens = 64
        #decoder has 66 tokens where token 64 is end of game and 65 is SOS
        self.num_decoder_tokens = 66
        self.max_encoder_seq_length = 60
        self.max_decoder_seq_length = 1

    def create_network(self):
        encoder_inputs = Input((None, self.num_encoder_tokens))
        encoder = LSTM(self.latent, return_state = True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_LSTM = LSTM(self.latent, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_LSTM(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])


    def load_data(self):
        if not os.path.isfile("train_X_seq2seq"):
            num_train = 10000
            num_valid = 5000
            num_test = 40000+5000+8207-10000-5000 
            self.data = dict()
            train_X = np.zeros((num_train*64, self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
            valid_X = np.zeros((num_valid*64, self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
            test_X = np.zeros((num_test*64, self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
            train_Y = np.zeros((num_train*64, self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
            valid_Y = np.zeros((num_valid*64, self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
            test_Y = np.zeros((num_test*64, self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
            train_decoder_in = np.zeros((num_train*64, self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
            valid_decoder_in = np.zeros((num_valid*64, self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
            test_decoder_in = np.zeros((num_test*64, self.max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
            with open("othello_database/player1.txt", "rb") as f:
                for i,line in enumerate(f): # player1.txt has 53207 lines
                    indices = [float(i.decode('UTF-8')) for i in line.strip().split()]
                    for t, index in enumerate(indices):
                        if (i<num_train):
                            for n in range(64-t):
                                train_X[i+n, t, int(index*64)-1] = 1
                                train_decoder_in[i+n, 0, 65] = 1
                            if(t+1<60):
                                train_Y[i+t, 0, int(indices[t+1]*64)-1] = 1
                            else:
                                train_Y[i+t, 0, 64] = 1
                        elif (i<num_train+num_valid):
                            for n in range(64-t):
                                valid_X[i+n-40000, t, int(index*64)-1] = 1
                                valid_decoder_in[i+n-40000, 0, 65] = 1
                            if(t+1<60):
                                valid_Y[i+t-40000, 0, int(indices[t+1]*64)-1] = 1
                            else:
                                valid_Y[i+t-40000, 0, 64] = 1
                        else:
                            for n in range(64-t):
                                test_X[i+n-45000, t, int(index*64)-1] = 1
                                test_decoder_in[i+n-45000, 0, 65] = 1
                            if(t+1<60):
                                test_Y[i+t-45000, 0, int(indices[t+1]*64)-1] = 1
                            else:
                                test_Y[i, 0, 64] = 1
            # np.save("train_X_seq2seq", train_X, allow_pickle=True)
            # np.save("test_X_seq2seq", test_X, allow_pickle=True)
            # np.save("valid_X_seq2seq", valid_X, allow_pickle=True)
            # np.save("train_Y_seq2seq", train_Y, allow_pickle=True)
            # np.save("test_Y_seq2seq", test_Y, allow_pickle=True)
            # np.save("valid_Y_seq2seq", valid_Y, allow_pickle=True)
        else:
            train_X = np.load("train_X_seq2seq", allow_pickle=True)
            test_X = np.load("test_X_seq2seq", allow_pickle=True)
            valid_X = np.load("valid_X_seq2seq", allow_pickle=True)
            train_Y = np.load("train_Y_seq2seq", allow_pickle=True)
            test_Y = np.load("test_Y_seq2seq", allow_pickle=True)
            valid_Y = np.load("valid_Y_seq2seq", allow_pickle=True)
        self.data["test_X"] = test_X
        self.data["test_Y"] = test_Y
        self.data["train_X"] = train_X
        self.data["train_Y"] = train_Y
        self.data["valid_X"] = valid_X
        self.data["valid_Y"] = valid_Y
        self.data["train_decoder_in"] = train_decoder_in
        self.data["test_decoder_in"] = test_decoder_in
        self.data["valid_decoder_in"] = valid_decoder_in

    def train(self, filename):
        print("Running {}...".format(filename))
        train_X = self.data["train_X"]
        valid_X = self.data["valid_X"]
        train_decoder_in = self.data["train_decoder_in"]
        valid_decoder_in = self.data["valid_decoder_in"]
        train_Y = self.data["train_Y"]
        valid_Y = self.data["valid_Y"]

        history = self.model.fit([train_X, train_decoder_in], train_Y, epochs=self.epochs, verbose=1, batch_size=self.batch_size, validation_data=([valid_X, valid_decoder_in], valid_Y))

        trainLosses_filename = "results/{}_trainLosses.txt".format(filename)
        validLosses_filename = "results/{}_validLosses.txt".format(filename)
        np.savetxt(trainLosses_filename, history.history['loss'], delimiter=',')
        np.savetxt(validLosses_filename, history.history['val_loss'], delimiter=',')

        self.model.save_weights("models/{}.h5".format(filename))



if __name__ == "__main__":
    kwargs = {'lr': 1e-5, 'batch_size': 16, 'epochs': 5} 
    model = Seq2Seq(**kwargs)
    model.create_network()
    model.load_data()
    model.train("seq2seq")
    #print(model.data["train_X"])