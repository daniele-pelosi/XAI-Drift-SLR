import os
import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import f1_score
from src.classification.PickleLoader import PickleLoader
from src.classification.online.nn import MLP
from river.drift import ADWIN
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, History
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)
import time
import argparse

class Main:
    STREAMFOLDERPATH = ''
    MODELSFOLDERPATH = 'serialized_models/online'


    def __init__(self, start: str, end: str, serialized: str):
        self.__main(start, end, serialized)

    def __main(self, start: str, end: str, serialized: str):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        parser.add_argument('-dataset', type=str, help="Dataset name")
        name = args.dataset
        if name =='brazilian':
            train_percentage = 93
        else:
            train_percentage = 7
        self.STREAMFOLDERPATH = args.dataset
        files = os.listdir(self.STREAMFOLDERPATH)
        files = sorted(files)
        strategy = 'D'
        stream_loader = PickleLoader(self.STREAMFOLDERPATH, files, start=files[0], end=files[-1])
        scaler = MinMaxScaler()
        number_of_model = 1
        list_t = []
        j = 0
        best_params = None
        dict_ens = {}

        for df in stream_loader:
            list_t.append(df)
            if j == train_percentage-1:
                init_df = pd.concat(list_t)
                y = init_df.iloc[:, -1].to_numpy().astype(int)
                X = init_df.iloc[:, 0:-1].to_numpy()
                X = scaler.fit_transform(X)
                diversity_list = []
                fscore_list = []
                for i in range(number_of_model):
                    if strategy == 'D':
                        start_time_init = time.time()
                        model, best_params = MLP(X, y, 'opt_'+name+'_'+str(i), 'D', i).start_opt()
                        x_test = model.predict(X, verbose=0)
                        y_predicted1 = (x_test > 0.5).astype(int)
                        cut_point = f1_score(y, y_predicted1, average='macro')
                        model.save('model_init_'+str(i)+'.h5')
                    else:
                        model = load_model('model_init_' + str(i) + '.h5')
                        x_test = model.predict(X, verbose=0)
                        y_predicted1 = (x_test > 0.5).astype(int)
                        cut_point = f1_score(y, y_predicted1, average='macro')

                    fscore_list.append(cut_point)
                    diversity_list.append(y_predicted1.flatten())
                    dict_ens[i] = [load_model('model_init_'+str(i)+'.h5'), best_params, ADWIN(delta=0.2), open('TSUNAMI_'+name+str(i)+'_'+strategy+'.log','w'), cut_point, []]
                    dict_ens[i][3].write('real,predicted\n')
                    index_div = [0]

                for g in list_t:
                    y =  g.iloc[:, -1].to_numpy().astype(int)
                    X = g.iloc[:, 0:-1].to_numpy()
                    X = scaler.transform(X)
                    for i in index_div:
                        dict_ens[i][5].append(g)
                        x_test = dict_ens[i][0].predict(X, verbose=0)
                        y_predicted1 = (x_test > 0.5).astype(int)
                        fscore = f1_score(y, y_predicted1, average='macro')
                        if fscore>=dict_ens[i][4]:
                            score=1
                        else:
                            score=0

                        in_drift, in_warning = dict_ens[i][2].update(score)

            elif j > train_percentage-1:
                y = df.iloc[:, -1].to_numpy().astype(int)
                X = df.iloc[:, 0:-1].to_numpy()
                X = scaler.transform(X)
                for i in index_div:
                    dict_ens[i][5].append(df)
                    x_test = dict_ens[i][0].predict(X, verbose=0)
                    y_predicted1 = (x_test > 0.5).astype(int)
                    fscore = f1_score(y, y_predicted1, average='macro')

                    if fscore >= dict_ens[i][4]:
                        score = 1
                    else:
                        score = 0

                    zz = 0
                    while zz < len(y):
                        dict_ens[i][3].write(str(y[zz]) + ',' + str(x_test[zz][0]) + '\n')
                        zz = zz + 1

                    if strategy == 'D':
                        in_drift, in_warning = dict_ens[i][2].update(score)
                        if in_drift:
                            print('DRIFT detected model->', i, 'index->', (j-train_percentage+1))
                            dict_ens[i][5] = dict_ens[i][5][len(dict_ens[i][5])-int(dict_ens[i][2].width):]
                            list_p = []

                            for p in dict_ens[i][5]:
                                list_p.append(p)
                            past_df = pd.concat(list_p)
                            X_past = past_df.iloc[:, 0:-1].to_numpy()
                            X_past = scaler.transform(X_past)
                            y_past = past_df.iloc[:, -1].to_numpy()

                            X_train, X_val, y_train, y_val = train_test_split(X_past, y_past, test_size=0.2, random_state=42, stratify=y_past)
                            dist = (np.unique(y_past, return_counts=True))
                            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
                            dict_ens[i][0].fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping, lr_reducer], batch_size=2 ** dict_ens[i][1]['batch_size'], epochs=200, class_weight = {0:((len(X_train))/(2*dist[1][0])), 1:((len(X_train))/(2*dist[1][1]))})
                            dict_ens[i][0].save('model_' + str(i) + '_' + str(j-train_percentage+1) + '.h5')
                            dict_ens[i][0] = load_model('model_' + str(i) + '_' + str(j-train_percentage+1) + '.h5')
            j = j + 1


parser = argparse.ArgumentParser()
parser.add_argument('--start',
                    help='Data di partenza in formato: AAAA-MM-DD, OPZIONALE: di default la prima della cartella.',
                    default=None)
parser.add_argument('--end',
                    help='Data di fine in formato: AAAA-MM-DD, OPZIONALE: di default l\'ultima della cartella.',
                    default=None)
parser.add_argument('--serialized',
                    help='Nome del file da caricare che contiene il modello precedentemente addestrato e serializzato',
                    default=None)
args = parser.parse_args()
try:
    Main(args.start, args.end, args.serialized)
except ValueError as err:
    print('\033[91m' + str(err))
