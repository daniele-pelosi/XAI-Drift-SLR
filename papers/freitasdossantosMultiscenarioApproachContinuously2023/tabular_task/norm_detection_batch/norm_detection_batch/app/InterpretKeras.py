import pathlib
import pickle
import statistics
import time
from itertools import repeat
import multiprocessing
from pprint import pprint

from joblib import Parallel, delayed

import numpy
import numpy as np
import pandas as pd
from lime import lime_tabular
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from norm_detection_batch.util import UtilDataSetFiles

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

GLOBAL_TRAINED_MODELS = []
GLOBAL_COLUMNS_DATA_FRAME = []
GLOBAL_STANDARD_VALUES = StandardScaler()


def get_relevant_features_by_lime(ensemble_size):
    complete_dataset = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/",
                                                               "en_train_no_null_features_2removed.csv")
    complete_dataset, statistical_values = UtilDataSetFiles.handle_missing_values(complete_dataset.copy())

    complete_dataset_standard, scaled_values = UtilDataSetFiles.apply_standardization(complete_dataset.copy(),
                                                                                      [57])

    UtilDataSetFiles.save_data_set(complete_dataset_standard, "/experiments/article_datasets/",
                                   "complete_dataset_standard.csv")

    # I just wanna to get the relevant features for the "future" editions. The "future" editions are the end of this
    # dataframe.
    testing_dataset = complete_dataset[16220:len(complete_dataset)-1]
    testing_dataset_standard = complete_dataset_standard[16220:len(complete_dataset_standard)-1]

    base_directory = str(pathlib.Path(__file__).parent.parent.parent)
    models_path = base_directory + "/experiments/article_datasets/trained_keras_models/"
    trained_models = get_trained_keras_models(ensemble_size, models_path)

    interpret(complete_dataset, complete_dataset_standard, testing_dataset, testing_dataset_standard, trained_models,
              scaled_values)


def interpret(complete_dataset, complete_dataset_standard, testing_dataset, testing_dataset_standard, trained_models,
              scaled_values):
    x_train = complete_dataset.drop('LABEL', axis=1)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(x_train),
        feature_names=x_train.columns,
        categorical_features=[0, 1, 3, 4, 22, 24, 32, 33, 51, 52, 54, 55, 56],
        categorical_names=["USER_IS_IP", "USER_IS_BOT", "USER_BLKED_BEFORE", "USER_BLKED_EVER", "COMM_HAS_SECT",
                           "COMM_VAND", "HIST_USER_HAS_RB", "HASH_REVERTED", "PREV_USER_IS_IP", "PREV_USER_SAME",
                           "NEXT_USER_IS_IP", "NEXT_USER_SAME", "NEXT_COMMENT_VAND"],
        discretize_continuous=True,
        class_names=[0, 1],
        mode="classification",
        sample_around_instance=True,
        verbose=True,
        feature_selection="highest_weights"
    )

    testing_data_vandalism = testing_dataset[testing_dataset["LABEL"] == 1]
    testing_data_vandalism_standard = testing_dataset_standard[testing_dataset_standard["LABEL"] == 1]

    start = time.time()

    global GLOBAL_TEST
    GLOBAL_TEST = trained_models

    num_cores = multiprocessing.cpu_count()
    '''relevant_features = Parallel(n_jobs=num_cores)(delayed(run_lime_parallel)(explainer,
                                                                              edition_index,
                                                                              testing_data_vandalism,
                                                                              testing_data_vandalism_standard,
                                                                              trained_models,
                                                                              scaled_values)
                                                   for edition_index, edition
                                                   in testing_data_vandalism.iterrows())'''

    relevant_features = []
    for edition_index, edition in testing_data_vandalism.iterrows():
        relevant_features.append(run_lime_parallel(explainer, edition_index, testing_data_vandalism,
                                                   testing_data_vandalism_standard, trained_models, scaled_values))

    end = time.time()
    print(str(end - start))

    for i in range(0, len(relevant_features)):
        edition_index = relevant_features[i][0]
        relevant_features_to_print = relevant_features[i][1]
        testing_dataset.loc[edition_index, 'RELEVANT_FEATURES'] = relevant_features_to_print
        testing_dataset_standard.loc[edition_index, 'RELEVANT_FEATURES'] = relevant_features_to_print

    UtilDataSetFiles.save_data_set(testing_dataset, "/experiments/article_datasets/",
                                   "testing_data_relevant_features_keras.csv")

    UtilDataSetFiles.save_data_set(testing_dataset_standard, "/experiments/article_datasets/",
                                   "testing_data_standard_relevant_features_keras.csv")

    print("ENDDDD!")


def run_lime_parallel(lime_explainer, edition_index, testing_data, testing_data_standard, trained_models,
                      scaled_values):
    features_names = testing_data.columns[0:57].values  # All columns, with the exception of the Label

    global GLOBAL_COLUMNS_DATA_FRAME
    GLOBAL_COLUMNS_DATA_FRAME = testing_data.columns.values

    global GLOBAL_STANDARD_VALUES
    GLOBAL_STANDARD_VALUES = scaled_values

    global GLOBAL_TRAINED_MODELS
    #GLOBAL_TRAINED_MODELS = training_models
    GLOBAL_TRAINED_MODELS = trained_models

    current_edition_to_explain = testing_data.loc[edition_index, features_names]  # Get all columns, except Label

    exp = lime_explainer.explain_instance(
        data_row=current_edition_to_explain,
        predict_fn=get_predict_probabilities,
        num_samples=100,
        num_features=3
    )

    #print("Score: " + str(exp.score))
    #exp.save_to_file("lime" + str(edition_index) + ".html")
    print(edition_index)

    relevant_features = numpy.array(exp.as_list())[:, 0]  # Only get the name of the relevant feature.
    relevant_features_to_print = ', '.join([item for item in relevant_features])
    #testing_data.loc[edition_index, 'RELEVANT_FEATURES'] = relevant_features_to_print

    edition_with_relevant_features = [edition_index, relevant_features_to_print]

    return edition_with_relevant_features


def get_predict_probabilities(data_to_predict):
    # Since we are using this method mainly for LIME, I'm going to process the data we are evaluating here, because
    # I can't pass the pre-processed data to LIME.

    data_to_predict = pd.DataFrame(data_to_predict, columns=GLOBAL_COLUMNS_DATA_FRAME[:-1])
    data_to_predict_standard = UtilDataSetFiles.apply_standardization_defined(data_to_predict, GLOBAL_STANDARD_VALUES,
                                                                              columns_not_altered=[])

    probabilities_predictions = get_probabilities_predictions_keras(GLOBAL_TRAINED_MODELS, data_to_predict_standard)

    return probabilities_predictions


def get_probabilities_predictions_keras(trained_models, x_test):
    models_predictions = []

    for model in trained_models:
        predictions = model.predict(x_test)
        models_predictions.append(predictions)

    summed_probabilities = np.sum(models_predictions, axis=0)
    ensemble_probabilities = np.divide(summed_probabilities, len(trained_models))

    return ensemble_probabilities


def get_trained_keras_models(ensemble_size, models_path):
    trained_models = []

    folder_name = "model"
    for model_index in range(0, ensemble_size):
        model_folder = models_path + folder_name + str(model_index)
        trained_models.append(keras.models.load_model(model_folder))

    return trained_models


def get_all_relevant_features():
    complete_dataset = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/",
                                                               "testing_data_relevant_features_keras.csv")

    relevant_features = complete_dataset[complete_dataset["RELEVANT_FEATURES"].notnull()].groupby(["RELEVANT_FEATURES"])["RELEVANT_FEATURES"]
    count_relevant_features = complete_dataset[complete_dataset["RELEVANT_FEATURES"].notnull()].groupby(["RELEVANT_FEATURES"])["RELEVANT_FEATURES"].count()

    relevant_features_order = []
    for features in relevant_features:
        features_group = features[0]
        features_editions_count = len(features[1])
        features_group = features_group.replace("'", "")
        features_group = features_group.replace(" ", "")
        individual_features = features_group.split(",")
        individual_features.sort()

        order_index = 0
        features_exist = False
        for added_relevant_features in relevant_features_order:
            if (individual_features[0] == added_relevant_features["FEATURES"][0] and
                    (individual_features[1] == added_relevant_features["FEATURES"][1]) and
                    (individual_features[2] == added_relevant_features["FEATURES"][2])):
                relevant_features_order[order_index]["COUNT"] += features_editions_count
                features_exist = True
            order_index += 1

        if not features_exist:
            ordered_relevant = {"FEATURES": individual_features, "COUNT": features_editions_count}
            relevant_features_order.append(ordered_relevant)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(count_relevant_features)

        print(relevant_features_order)


def get_all_relevant_features_one():
    complete_dataset = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/",
                                                               "testing_data_relevant_features_keras.csv")

    relevant_features = \
    complete_dataset[complete_dataset["RELEVANT_FEATURES"].notnull()].groupby(["RELEVANT_FEATURES"])[
        "RELEVANT_FEATURES"]
    count_relevant_features = \
    complete_dataset[complete_dataset["RELEVANT_FEATURES"].notnull()].groupby(["RELEVANT_FEATURES"])[
        "RELEVANT_FEATURES"].count()

    relevant_features_order = []
    for features in relevant_features:
        features_group = features[0]
        features_editions_count = len(features[1])
        features_group = features_group.replace("'", "")
        features_group = features_group.replace(" ", "")
        individual_features = features_group.split(",")
        individual_features.sort()

        order_index = 0
        features_exist = False
        for added_relevant_features in relevant_features_order:
            if (individual_features[0] == added_relevant_features["FEATURES"][0] or
                    (individual_features[1] == added_relevant_features["FEATURES"][1]) and
                    (individual_features[2] == added_relevant_features["FEATURES"][2])):
                relevant_features_order[order_index]["COUNT"] += features_editions_count
                features_exist = True

        if not features_exist:
            ordered_relevant = {"FEATURES": individual_features, "COUNT": features_editions_count}
            relevant_features_order.append(ordered_relevant)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(count_relevant_features)

        print(relevant_features_order)

