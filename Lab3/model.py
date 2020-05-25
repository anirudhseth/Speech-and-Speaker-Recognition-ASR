import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Input, InputLayer, Dropout
from utils import load_data_npz, plot_train_val_curves
from lab3_proto import normalise_over_train
from evaluate_model import *
import numpy as np
import os

class PhonemeRecogniser:
    def __init__(self, data_dir):
        self.stateList = np.load(data_dir+'/StateList.npy')
        self.n_classes = len(self.stateList)

    def load_data(self, hparams):
        data_dir = hparams['data_dir']
        dynamic = hparams['dynamic']
        features = hparams['features']

        print('Using dynamic features:', dynamic)
        print('Using features', features)

        normalisation_strategy = hparams['normalisation']

        if not dynamic:
            x_train, y_train = load_data_npz(data_dir+'/train.npz', features, dynamic=dynamic)
            x_val, y_val = load_data_npz(data_dir+'/val.npz', features, dynamic=dynamic)
            x_test, y_test = load_data_npz(data_dir+'/test.npz', features, dynamic=dynamic)
        else:
            x_train = np.load(f'{data_dir}/train_{features}_dynamic.npy', allow_pickle=True)
            x_val = np.load(f'{data_dir}/val_{features}_dynamic.npy', allow_pickle=True)
            x_test = np.load(f'{data_dir}/test_{features}_dynamic.npy', allow_pickle=True)
            y_train = np.load(f'{data_dir}/train_target_dynamic.npy', allow_pickle=True)
            y_val = np.load(f'{data_dir}/val_target_dynamic.npy', allow_pickle=True)
            y_test = np.load(f'{data_dir}/test_target_dynamic.npy', allow_pickle=True)



        if normalisation_strategy == 0:
            x_train, x_val, x_test, y_train, y_val, y_test = normalise_over_train(x_train, x_val, x_test, y_train, y_val, y_test)
        else:
            print('Not implemented this normalisation strategy yet')
            raise NotImplementedError

        # convert feature arrays to 32 bits floating point format because of the hardware limitation in most GPUs
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')

        # one hot encode y's
        y_train = np_utils.to_categorical(np.array(y_train), self.n_classes)
        y_val = np_utils.to_categorical(y_val, self.n_classes)
        y_test = np_utils.to_categorical(y_test, self.n_classes)


        self.input_dim = x_train.shape[1]
        return x_train, x_val, x_test, y_train, y_val, y_test


    def build_model(self, hparams):
        n_hidden_layers = hparams['n_hidden_layers']
        n_hidden_nodes = hparams['n_hidden_nodes']
        dropout_rate = hparams['dropout_rate']

        if n_hidden_layers != len(n_hidden_nodes):
            print('Your number of hidden layers', n_hidden_layers, 'does not match the size of list n_hidden_nodes:',
                  len(n_hidden_nodes))
            exit()

        print('Building network with', n_hidden_layers, 'hidden layers')

        if hparams['activation'] == 'relu':
            activation = tf.nn.relu
        elif hparams['activation'] == 'sigmoid':
            activation = tf.nn.sigmoid
        else:
            print('Activation function', hparams['activation'], 'not implemented yet')
            raise NotImplementedError

        print('Using activation function on hidden layers', hparams['activation'])

        model = tf.keras.models.Sequential()
        model.add(Input(self.input_dim))

        for i in range(n_hidden_layers):
            model.add(Dense(n_hidden_nodes[i], activation=activation))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        model.add(Dense(self.n_classes, activation=tf.nn.softmax))

        optimiser = hparams['optimiser']
        print('using optimiser', optimiser)
        if optimiser == 'adam':
            opt = Adam(learning_rate=hparams['lr'])
        elif optimiser == 'sgd':
            opt = SGD(learning_rate=hparams['lr'])
        else:
            print('Optimiser', optimiser, 'not implemented yet')
            raise NotImplementedError

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        print(model.summary())

        self.model = model
        self.model_name = hparams['model_name']
        print('model name', self.model_name)

        # create dir for model results
        os.mkdir('results/' + self.model_name)

        self.n_epochs = hparams['n_epochs']
        self.batch_size = hparams['batch_size']
        return model


    def train_model(self, x_train, y_train, x_val, y_val):
        print('Training for', self.n_epochs, 'with batch size', self.batch_size)
        history = self.model.fit(x_train, y_train, epochs=self.n_epochs, batch_size=self.batch_size, validation_data=(x_val, y_val))
        return history


    def evaluate(self, x_test, y_test, history, compute_edit_distances=False):
        predictions = self.model.predict(x_test)
        print('Saving predictions')
        np.save('results/' + self.model_name + '/predictions.npy', predictions)
        print('Done')
        predictions_idx = np.argmax(predictions, axis=1)
        true_idx = np.argmax(y_test, axis=1)

        # frame-by-frame at state level, compute confusion matrix as well
        evaluate_by_frame_state_level(predictions_idx, true_idx, self.stateList, self.model_name)
        # frame-by-frame at the phoneme level, compute confusion matrix as well
        evaluate_by_frame_phoneme_level(predictions_idx, true_idx, self.stateList, self.model_name)
        if compute_edit_distances:
            # edit distance at state level
            edit_distance_state_level(predictions_idx, true_idx, self.stateList)
            # edit distance at phoneme level
            edit_distance_phoneme_level(predictions_idx, true_idx, self.stateList)

        print('Plotting graphs...')
        plot_train_val_curves(history, self.model_name)

