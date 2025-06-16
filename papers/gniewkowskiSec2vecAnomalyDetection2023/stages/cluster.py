import argparse
import os
import simplejson as json
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

from sklearn import metrics


def get_args():

    parser = argparse.ArgumentParser(description="Classifier")

    parser.add_argument(
        "input_data", help="Data used for training"
    )
    parser.add_argument(
        "output_path", help="Path used for saving results."
    )
    return parser.parse_args()


def main():

    args = get_args()
    os.makedirs(args.output_path + "/saved", exist_ok=True)
    with open(f"{args.input_data}/result.json") as f:
        data = json.load(f)

    labels = data["labels"]
    labels = np.array(labels)
    ids = data["ids"]
    data = data["data"]
    data = pd.DataFrame.from_dict(data).to_numpy()


    clusterers = {
        "dbscan": DBSCAN(),
        "kmeans": KMeans(n_clusters=2),
        "ac": AgglomerativeClustering(n_clusters=2)
    }

    scores = {
        "dbscan": dict(),
        "kmeans": dict(),
        "ac": dict(),
    }

    for clust_name, clust in clusterers.items():
        pred_labels = clust.fit_predict(data)
        filename = f'{args.output_path}/saved/{clust_name}.pickle'
        pickle.dump(clust, open(filename, 'wb'))
        with open(f"{args.output_path}/saved/predictions-{clust_name}.json", "wt") as f:
            json.dump([int(a) for a in list(pred_labels)], f)

        ami = metrics.adjusted_mutual_info_score(labels, pred_labels)
        ari = metrics.adjusted_rand_score(labels, pred_labels)
        fmc = metrics.fowlkes_mallows_score(labels, pred_labels)

        scores[clust_name] = {
            "ami": ami,
            "ari": ari,
            "fmc": fmc
        }

    with open(f"{args.output_path}/metrics.json", "wt") as f:
        json.dump(scores, f)


    

if __name__ == "__main__":
    main()
