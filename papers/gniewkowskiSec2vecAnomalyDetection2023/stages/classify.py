import argparse
import os
import simplejson as json
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

class Classifier(object):

    def __init__(self, method, **args):
        clfs = {
            "svc": SVC,
            "dt": DecisionTreeClassifier,
            "lr": LogisticRegression
        }
        self.clf = clfs[method](**args)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.clf.predict(x_test)


def get_args():

    parser = argparse.ArgumentParser(description="Classifier")

    parser.add_argument(
        "input_data", help="Data used for training"
    )
    parser.add_argument(
        "output_path", help="Path used for saving results."
    )
    parser.add_argument(
        "--n-runs", type=int, default=5
    )

    return parser.parse_args()


def main():

    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    with open(f"{args.input_data}/result.json") as f:
        data = json.load(f)

    labels = data["labels"]
    labels = np.array(labels)
    ids = data["ids"]
    data = data["data"]
    data = pd.DataFrame.from_dict(data).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

    classifiers = [
        #LogisticRegression(n_jobs=-1, max_iter=500, verbose=1, random_state=0),
        RandomForestClassifier(random_state=0, n_jobs=-1),
        MLPClassifier(random_state=0)
        #SVC(random_state=0, kernel="linear", probability=True, max_iter=1000),
    ]
    fig, ax = plt.subplots()

    def predict(clasif, threshold, X_test):
        y_pred = (clasif.predict_proba(X_test)[:,1] >= threshold).astype(bool)
        return y_pred

    def get_fpr_index(tpr, threshold):
        min_ = 1
        min_ix = 0
        for it, t in enumerate(tpr):
            diff = abs(t - threshold)
            if diff < min_:
                min_ = diff
                min_ix = it
        return min_ix

    scores = {
        "lr": dict(),
        "svc": dict(),
        "rf": dict(),
        "mlp": dict()
    }
    split = list(cv.split(data, labels))[:args.n_runs]
    train_ids = list(np.array(ids, dtype=str)[split[0][0]])
    test_ids = list(np.array(ids, dtype=str)[split[0][1]])
    os.makedirs(f'{args.output_path}/saved/', exist_ok=True)
    with open(f"{args.output_path}/saved/split.json", "wt") as f:
        json.dump({"test": test_ids, "train": train_ids}, f)

    for c, classifier in enumerate(classifiers):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        color = {0: "r", 1: "b", 2: "g"}[c]
        clf_name = {0: "rf", 1: "mlp", 2: "svc"}[c]
        #clf_name = "rf"
        scores[clf_name]["f1"] = []
        scores[clf_name]["mcc"] = []
        scores[clf_name]["fpr90"] = []
        scores[clf_name]["fpr99"] = []

        for i, (train, test) in enumerate(split):
            classifier.fit(data[train], labels[train])
            if i == 0:
                filename = f'{args.output_path}/saved/{clf_name}.pickle'
                pickle.dump(classifier, open(filename, 'wb'))
                pred_labels = classifier.predict(data[test])
                predictions_save = {
                    _id: l for _id, l in zip(test_ids, pred_labels)
                }
                with open(f"{args.output_path}/saved/predictions-{clf_name}.json", "wt") as f:
                    json.dump(predictions_save, f)
            viz = plot_roc_curve(classifier, data[test], labels[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=None, pos_label="anomaly")
            fpr = viz.fpr
            tpr = viz.tpr
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            pred_labels = classifier.predict(data[test])
            f1 = f1_score(labels[test], pred_labels, pos_label="anomaly")
            mcc = matthews_corrcoef(labels[test], pred_labels)
            fpr_score99 = fpr[get_fpr_index(tpr, 0.99)] 
            fpr_score9 = fpr[get_fpr_index(tpr, 0.9)] 
            scores[clf_name]["f1"].append(f1)
            scores[clf_name]["mcc"].append(mcc)
            scores[clf_name]["fpr90"].append(fpr_score9)
            scores[clf_name]["fpr99"].append(fpr_score99)
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color="black", alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=color,
                label='{} Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'.format(clf_name, mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)


    result_scores = dict()
    for k, v in scores.items():
        result_scores[k] = dict()
        for metric, d in scores[k].items():
            #result_scores[k][metric] = f"{np.mean(d)} +- {np.std(d)}"
            result_scores[k][metric] = np.mean(d)
            result_scores[k][f"{metric}_std"] = np.std(d)
    with open(f"{args.output_path}/metrics.json", "wt") as f:
        json.dump(result_scores, f)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example",
           xlabel="False Positive Rate",
           ylabel="True Positive Rate"
    )
    ax.legend(loc="lower right")
    fig.savefig(f"{args.output_path}/roc.png")

    

if __name__ == "__main__":
    main()
