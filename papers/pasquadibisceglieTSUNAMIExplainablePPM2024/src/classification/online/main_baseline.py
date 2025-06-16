import os
import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import f1_score
from src.classification.PickleLoader import PickleLoader
from river.drift import ADWIN
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)
from src.classification.online.rf_opt import RF
from src.classification.online.lr_opt import LR
from src.classification.online.xgb_opt import XGB
import argparse

class Main:
    # Costante dove salviamo il percorso della cartella dove viene serializzato lo stream
    STREAMFOLDERPATH = ''
    MODELSFOLDERPATH = 'serialized_models/online'

    """
        Metodo costruttore. Richiama il metodo main di Main.
    """

    def __init__(self, start: str, end: str, serialized: str):
        self.__main(start, end, serialized)

    def __main(self, start: str, end: str, serialized: str):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        parser.add_argument('-dataset', type=str, help="Dataset name")
        parser.add_argument('-classifier', type=str, help="Classifier: XGB, LR or RF")

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
        method = args.classifier
        for df in stream_loader:
            list_t.append(df)
            if j == train_percentage-1:
                init_df = pd.concat(list_t)
                y = init_df.iloc[:, -1].to_numpy().astype(int)  # g[10].to_numpy().astype(int)
                X = init_df.iloc[:, 0:-1].to_numpy()
                X = scaler.fit_transform(X)
                diversity_list = []
                fscore_list = []
                for i in range(number_of_model):
                    dist = (np.unique(y, return_counts=True))
                    class_weight = {0: ((len(X)) / (2 * dist[1][0])), 1: ((len(X)) / (2 * dist[1][1]))}
                    if method == 'RF':
                        rf = RF(X, y)
                        best = rf.find_best()
                        model = RandomForestClassifier(random_state=42, class_weight=class_weight, n_estimators=500, max_features=best['max_features'])

                    elif method == 'LR':
                        lr = LR(X, y)
                        best = lr.find_best()
                        model = LogisticRegression(random_state=42, class_weight=class_weight, C = 2**best['C'])
                    elif method == 'XGB':
                        xgb = XGB(X, y)
                        best = xgb.find_best()
                        model = XGBClassifier(random_state = 42, scale_pos_weight=sum(y == 0) / sum(y == 1),
                                              objective='binary:logistic',
                                              n_estimators=500,
                                              learning_rate= best['learning_rate'],
                                              subsample=best['subsample'],
                                              max_depth=int(best['max_depth']),
                                              colsample_bytree=best['colsample_bytree'],
                                              min_child_weight=int(best['min_child_weight']))
                    model.fit(X,y)
                    y_predicted1 = model.predict(X)

                    cut_point = f1_score(y, y_predicted1, average='macro')
                    fscore_list.append(cut_point)
                    diversity_list.append(y_predicted1.flatten())
                    dict_ens[i] = [model, best_params, ADWIN(delta=0.2), open(method+'_'+name+str(i)+'_'+strategy+'.log','w'), cut_point, []]
                    dict_ens[i][3].write('real,predicted\n')
                    index_div = [0]
                for g in list_t:
                    y =  g.iloc[:, -1].to_numpy().astype(int)
                    X = g.iloc[:, 0:-1].to_numpy()
                    X = scaler.transform(X)
                    for i in index_div:
                        dict_ens[i][5].append(g)
                        x_test = dict_ens[i][0].predict(X)
                        y_predicted1 = x_test
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
                    x_test = dict_ens[i][0].predict(X).astype(int)
                    x_test2 = dict_ens[i][0].predict_proba(X)
                    y_predicted1 = x_test.astype(int)
                    fscore = f1_score(y, y_predicted1, average='macro')
                    if fscore >= dict_ens[i][4]:
                        score = 1
                    else:
                        score = 0
                    zz = 0
                    while zz < len(y):
                        dict_ens[i][3].write(str(y[zz]) + ',' + str(x_test2[zz][1]) + '\n')
                        zz = zz + 1
                    if strategy == 'D':
                        in_drift, in_warning = dict_ens[i][2].update(score)
                        if in_drift:
                            print('DRIFT detected model->', i, 'index->', j)
                            dict_ens[i][5] = dict_ens[i][5][len(dict_ens[i][5])-int(dict_ens[i][2].width):]
                            list_p = []

                            for p in dict_ens[i][5]:
                                list_p.append(p)

                            past_df = pd.concat(list_p)
                            X_past = past_df.iloc[:, 0:-1].to_numpy()
                            X_past = scaler.transform(X_past)
                            y_past = past_df.iloc[:, -1].to_numpy()

                            dist = (np.unique(y_past, return_counts=True))
                            class_weight = {0: ((len(X_past)) / (2 * dist[1][0])),
                                            1: ((len(X_past)) / (2 * dist[1][1]))}
                            if method == 'RF':
                                model = RandomForestClassifier(random_state=42, class_weight=class_weight)
                            elif method == 'LR':
                                model = LogisticRegression(random_state=42, class_weight=class_weight, C=2**best['C'])
                            elif method == 'XGB':
                                model = XGBClassifier(random_state=42, scale_pos_weight=sum(y_past == 0) / sum(y_past == 1),
                                                      objective='binary:logistic',
                                                      n_estimators=500,
                                                      learning_rate=best['learning_rate'],
                                                      subsample=best['subsample'],
                                                      max_depth=int(best['max_depth']),
                                                      colsample_bytree=best['colsample_bytree'],
                                                      min_child_weight=int(best['min_child_weight']))

                            model.fit(X_past, y_past)
                            dict_ens[i][0] = model

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
