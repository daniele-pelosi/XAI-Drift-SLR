import tensorflow as tf
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from tensorflow.keras.layers import Input, Dense
from time import perf_counter
from sklearn.model_selection import train_test_split
import os
import random

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

class MLP():
    def __init__(self, X, y, log_name, strategy, seed):
        self.X = X
        self.y = y
        self.seed = seed
        self.log_name = log_name
        self.strategy = strategy
        self.best_model = None
        self.best_numparameters = 0
        self.best_score = np.inf
        self.best_time = 0
        self.space = {#'w_factor': hp.choice('w_factor', [2]),
                      'cellsize': hp.choice('cellsize', [16, 32, 64, 128]),#scope.int(hp.loguniform('cellsize', np.log(16), np.log(64))),
                      'dropout': hp.uniform("dropout", 0, 0.5),
                      'batch_size': hp.choice('batch_size', [8, 9, 10]),
                      #'act_func': hp.choice('act_func', ['relu', 'tanh', 'linear']),
                      'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
                      'n_layers': hp.choice('n_layers', [
             {'n_layers': 1},
             {'n_layers': 2, 'cellsize22': scope.int(hp.loguniform('cellsize22', np.log(10), np.log(150)))},
             {'n_layers': 3, 'cellsize32': scope.int(hp.loguniform('cellsize32', np.log(10), np.log(150))),
              'cellsize33': scope.int(hp.loguniform('cellsize33', np.log(10), np.log(150)))}
         ])
                      }
        set_seeds(self.seed)

    def get_model(self, params):
        n_layers = int(params["n_layers"]["n_layers"])

        input_act = Input(shape=(self.X.shape[1]), dtype='float32', name='input_act')

        x = (tf.keras.layers.Dense(int(params["cellsize"]),
                                  kernel_initializer='glorot_uniform',
                                  ))(input_act)

        x = tf.keras.layers.Dropout(params["dropout"])(x)
        for i in range(2, n_layers + 1):
            x = (tf.keras.layers.Dense(int(params["n_layers"]["cellsize%s%s" % (n_layers, i)]),
                                      kernel_initializer='glorot_uniform',
                                      ))(x)

            x = tf.keras.layers.Dropout(params["dropout"])(x)

        out_a = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform', name='output_a')(x)
        model = Model(inputs=input_act, outputs=out_a)

        opt = Adam(learning_rate=params["learning_rate"])

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

        return model

    def train_and_evaluate_model(self, params):
        start_time = perf_counter()
        model = self.get_model(params)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42, stratify=self.y, shuffle=True)

        dist = (np.unique(y_train, return_counts=True))

        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            verbose=0,
                            callbacks=[early_stopping, lr_reducer],
                            batch_size=2 ** params['batch_size'], epochs=200,
                            class_weight={0: ((len(X_train)) / (2 * dist[1][0])),
                                          1: ((len(X_train)) / (2* dist[1][1]))})

        scores = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
        score = min(scores)
        end_time = perf_counter()

        if self.best_score > score:
            self.best_score = score
            self.best_model = model
            self.best_numparameters = model.count_params()
            self.best_time = end_time - start_time

        return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(history.history['loss']),
                'n_params': model.count_params(), 'time': end_time - start_time}

    def start_opt(self):
        log_file = open(self.log_name + '_' + self.strategy + '.log', 'w')
        log_file.write('Starting model selection...')
        trials = Trials()
        best = fmin(self.train_and_evaluate_model, self.space, algo=tpe.suggest, max_evals=20, trials=trials,
                    rstate=np.random.RandomState(42))
        self.best_params = hyperopt.space_eval(self.space, best)

        log_file.write("\nHyperopt trials")
        log_file.write("\ntid,loss,learning_rate,n_modules,batch_size,w_factor,time,n_epochs,n_params,perf_time")
        for trial in trials.trials:
            log_file.write("\n%d,%f,%f,%d,%d,%s,%d,%d,%f" % (trial['tid'],
                                                               trial['result']['loss'],
                                                               trial['misc']['vals']['learning_rate'][0],
                                                               int(trial['misc']['vals']['n_layers'][0] + 1),
                                                               trial['misc']['vals']['batch_size'][0] + 8,
                                                               (trial['refresh_time'] - trial['book_time']).total_seconds(),
                                                               trial['result']['n_epochs'],
                                                               trial['result']['n_params'],
                                                               trial['result']['time']))

        log_file.write("\n\nBest parameters:")
        log_file.write("\nModel parameters: %d" % self.best_numparameters)
        log_file.write('\nBest Time taken: %f' % self.best_time)
        return self.best_model, self.best_params