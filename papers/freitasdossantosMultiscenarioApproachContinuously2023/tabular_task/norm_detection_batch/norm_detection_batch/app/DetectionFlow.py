import math
import os.path
import pathlib
import pickle
import timeit
from time import process_time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from norm_detection_batch.util import UtilDataSetFiles

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def start_process():
    # testing pickle files
    """
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)
    directory_to_save = "/experiments/keras/trained_models/" + str(0)
    information_file_name = "/" + str(1) + "_step_test_results.pickle"

    information_file_path = base_directory + directory_to_save + information_file_name
    file = open(information_file_path, 'rb')
    data = pickle.load(file)
    """

    # First experiment. With Concept Drift
    # DetectionFlow.start_process()
    # DetectionFlow.generate_k_fold(None, "/experiments/keras/datasets/")
    # DetectionFlow.simulate_re_label("/experiments/keras/datasets/", "/experiments/keras/datasets/", 30)
    train("/experiments/keras/datasets/", "/experiments/keras/trained_models/", "keras", 30, 2, evaluate_by_step=True)


def start_process_interpretability():
    train_for_interpretability("/experiments/keras/datasets/", "/experiments/keras/trained_models/", "keras", 30, 2,
                               "/experiments/keras/interpretability_data/", "concept_drift", evaluate_by_step=False)


def train(dataset_directory, directory_to_save, ml_model, number_of_executions, cluster_class, evaluate_by_step=False,
          current_dataset_weight=0.1, future_dataset_weight=10, new_label_value=0, old_label_value=1,
          columns_to_drop=["LABEL", "CLUSTER"], step_size=512):
    print("Training process started!!!")
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    start_execution = 0
    for current_execution in range(start_execution, number_of_executions):
        print("Execution: " + str(current_execution+1) + " started!!!")
        current_dataset = get_working_dataset("current_train", dataset_directory, current_execution)
        current_dataset_no_null, _ = UtilDataSetFiles.handle_missing_values(current_dataset, [57])
        current_dataset_standardized, _ = UtilDataSetFiles.apply_standardization(current_dataset_no_null, [57])

        future_dataset_standardized = get_working_dataset("future_train_standardized", dataset_directory,
                                                          current_execution)

        test_dataset_standardized = get_working_dataset("test_standardized", dataset_directory, current_execution)

        # Different weights for future data and past data. The weights were found empirically.
        current_dataset_standardized["WEIGHT"] = current_dataset_weight
        future_dataset_standardized["WEIGHT"] = future_dataset_weight

        # Simulating the re-label feedback based on the clusters found by K-Means
        future_dataset_re_label, only_re_label_future_dataset = apply_re_label(future_dataset_standardized,
                                                                               cluster_class, new_label_value,
                                                                               old_label_value)
        test_dataset_re_label, only_re_label_test_dataset = apply_re_label(test_dataset_standardized, cluster_class,
                                                                           new_label_value, old_label_value,
                                                                           replicate_re_label=False)

        complete_training_dataset = pd.concat([current_dataset_standardized, future_dataset_re_label,
                                               only_re_label_future_dataset]).sample(frac=1)

        # Flow of execution: the final trained model is trained on all the available data.
        all_balanced_datasets, _ = UtilDataSetFiles.handle_imbalanced_dataset(complete_training_dataset, "", "", False)

        if evaluate_by_step:
            # We want to evaluate how the batch training works when you use different sizes for the batch.
            # This is only to get different set of results, it's not part of the normal flow of execution.
            run_with_different_batch_size(current_dataset_weight, future_dataset_weight, dataset_directory,
                                          current_execution, step_size, ml_model, columns_to_drop, cluster_class,
                                          directory_to_save)
        else:
            trained_models = []
            train_execution_time = 0
            for dataset in range(0, len(all_balanced_datasets)):
                balanced_dataset = all_balanced_datasets[dataset]

                x_train = balanced_dataset.drop(columns_to_drop, axis=1)
                y_train = balanced_dataset['LABEL']

                training_weights = []
                dataset_contain_weight = False
                if "WEIGHT" in balanced_dataset.columns.values:
                    dataset_contain_weight = True
                    training_weights = balanced_dataset['WEIGHT']
                    x_train = x_train.drop(["WEIGHT"], axis=1)

                if ml_model == "keras":
                    model, elapsed_time = train_with_keras(x_train, y_train, dataset_contain_weight, training_weights)
                    trained_models.append(model)
                    train_execution_time += elapsed_time
                else:
                    raise Exception("ML Model not recognized!")

            information_to_save = {
                "current_execution": current_execution,
                "test_results": {},
                "test_re_label_results": {},
                "train_execution_time": train_execution_time
            }
            if ml_model == "keras":
                information_to_save["test_results"] = evaluate_ensemble_models_keras(trained_models, test_dataset_re_label,
                                                                                     columns_to_drop)

                if len(only_re_label_test_dataset) > 0:
                    only_re_label_test_dataset = only_re_label_test_dataset.drop(["CLUSTER"], axis=1)
                    test_re_label_results = evaluate_re_label(trained_models, only_re_label_test_dataset)
                    information_to_save["test_re_label_results"] = test_re_label_results

                current_execution_directory = directory_to_save + str(current_execution)
                save_keras_model(trained_models, "model", current_execution_directory)

                information_file_path = base_directory + current_execution_directory + "/overall_test_results" + ".pickle"
                pickle.dump(information_to_save, open(information_file_path, 'wb'))

        print("Execution: " + str(current_execution + 1) + " finished!!!")


def train_for_interpretability(dataset_directory, directory_to_save, ml_model, number_of_executions, cluster_class,
                               interpretability_path, interpretability_type, evaluate_by_step=False,
                               current_dataset_weight=0.1, future_dataset_weight=10, new_label_value=0,
                               old_label_value=1, columns_to_drop=["LABEL", "CLUSTER"], step_size=512):
    print("Training process started!!!")
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)
    current_execution = 0

    # The datasets were already separated. The same datasets are used for the different training approaches.
    if interpretability_type == "no_concept_drift":
        complete_dataset = get_working_dataset("current_train", dataset_directory, current_execution)
        complete_dataset_no_null, _ = UtilDataSetFiles.handle_missing_values(complete_dataset, [57])
        complete_dataset_standardized, scaled_values = UtilDataSetFiles.apply_standardization(complete_dataset_no_null.copy(),
                                                                                              [57])
        complete_dataset_standardized["WEIGHT"] = 1
        complete_dataset_standardized["CLUSTER"] = ""

        test_dataset = get_working_dataset("test", dataset_directory, current_execution)
        test_no_null, _ = UtilDataSetFiles.handle_missing_values(test_dataset, [57])
        final_test_dataset = UtilDataSetFiles.apply_standardization_defined(test_no_null.copy(), scaled_values, [57])

        complete_training_dataset = pd.concat([complete_dataset_standardized]).sample(frac=1)
    else:
        current_dataset = get_working_dataset("current_train", dataset_directory, current_execution)
        current_dataset_no_null, _ = UtilDataSetFiles.handle_missing_values(current_dataset, [57])
        current_dataset_standardized, _ = UtilDataSetFiles.apply_standardization(current_dataset_no_null.copy(),
                                                                                             [57])

        future_dataset = get_working_dataset("future_train_cluster", dataset_directory, current_execution)
        future_dataset_no_null, _ = UtilDataSetFiles.handle_missing_values(future_dataset, [57, 58])
        future_dataset_standardized, scaled_values = UtilDataSetFiles.apply_standardization(future_dataset_no_null.copy(),
                                                                                            [57, 58])

        # Different weights for future data and past data. The weights were found empirically.
        current_dataset_standardized["WEIGHT"] = current_dataset_weight
        future_dataset_standardized["WEIGHT"] = future_dataset_weight

        # Simulating the re-label feedback based on the clusters found by K-Means
        future_dataset_re_label, only_re_label_future_dataset = apply_re_label(future_dataset_standardized,
                                                                               cluster_class, new_label_value,
                                                                               old_label_value)

        test_dataset = get_working_dataset("test_cluster", dataset_directory, current_execution)
        test_no_null, _ = UtilDataSetFiles.handle_missing_values(test_dataset, [57, 58])
        test_dataset_standardized = UtilDataSetFiles.apply_standardization_defined(test_no_null.copy(), scaled_values,
                                                                                   [57, 58])
        test_no_null, _ = apply_re_label(test_no_null, cluster_class, new_label_value, old_label_value,
                                         replicate_re_label=False)
        final_test_dataset, _ = apply_re_label(test_dataset_standardized, cluster_class, new_label_value,
                                               old_label_value, replicate_re_label=False)

        complete_training_dataset = pd.concat([current_dataset_standardized, future_dataset_re_label,
                                               only_re_label_future_dataset]).sample(frac=1)
        complete_dataset_no_null = pd.concat([current_dataset_no_null, future_dataset_no_null]).sample(frac=1)

    # Flow of execution: the final trained model is trained on all the available data.
    all_balanced_datasets, _ = UtilDataSetFiles.handle_imbalanced_dataset(complete_training_dataset, "", "", False)

    trained_models = []
    train_execution_time = 0
    for dataset in range(0, len(all_balanced_datasets)):
        balanced_dataset = all_balanced_datasets[dataset]

        x_train = balanced_dataset.drop(columns_to_drop, axis=1)
        y_train = balanced_dataset['LABEL']

        training_weights = []
        dataset_contain_weight = False
        if "WEIGHT" in balanced_dataset.columns.values:
            dataset_contain_weight = True
            training_weights = balanced_dataset['WEIGHT']
            x_train = x_train.drop(["WEIGHT"], axis=1)

        if ml_model == "keras":
            model, elapsed_time = train_with_keras(x_train, y_train, dataset_contain_weight, training_weights)
            trained_models.append(model)
            train_execution_time += elapsed_time
        else:
            raise Exception("ML Model not recognized!")

    directory_interpretability = base_directory + interpretability_path + interpretability_type
    save_interpretability_data(complete_dataset_no_null, complete_training_dataset, test_no_null,
                               final_test_dataset, scaled_values, directory_interpretability,
                               trained_models)


def save_interpretability_data(last_complete_data_block, last_complete_data_block_standardized, last_no_null_testing,
                               last_no_null_testing_standardized, last_scaled_values, directory_interpretability,
                               trained_models):

    save_pickle(last_complete_data_block, directory_interpretability + "/training" + ".pickle")
    save_pickle(last_complete_data_block_standardized, directory_interpretability + "/training_standard" + ".pickle")
    save_pickle(last_no_null_testing, directory_interpretability + "/testing" + ".pickle")
    save_pickle(last_no_null_testing_standardized, directory_interpretability + "/testing_standard" + ".pickle")
    save_pickle(last_scaled_values, directory_interpretability + "/scaled_values" + ".pickle")

    ensemble_path = directory_interpretability + "/ensemble/"
    folder_name = "model"
    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = ensemble_path + "/" + folder_name + str(model_index)
        model.save(file_path)


def save_pickle(data, path):
    pickle.dump(data, open(path, 'wb'))


def get_working_dataset(file_name, directory, index):
    complete_file_name = str(index) + "_" + file_name + ".csv"
    dataset = UtilDataSetFiles.get_data_set_from_file(directory, complete_file_name)

    return dataset


def run_with_different_batch_size(current_dataset_weight, future_dataset_weight, dataset_directory, current_execution,
                                  step_size, ml_model, columns_to_drop, cluster_class, directory_to_save):
    current_dataset = get_working_dataset("current_train", dataset_directory, current_execution)
    current_dataset["CLUSTER"] = -1  # Adding this to have similar columns in the datasets.
    current_dataset["WEIGHT"] = current_dataset_weight

    future_dataset = get_working_dataset("future_train_cluster", dataset_directory, current_execution)
    future_dataset["WEIGHT"] = future_dataset_weight

    test_dataset = get_working_dataset("test_cluster", dataset_directory, current_execution)

    aux_complete_training_dataset = pd.concat([current_dataset, future_dataset])
    train_with_step(aux_complete_training_dataset, step_size, test_dataset, columns_to_drop,
                    ml_model, cluster_class, current_execution, directory_to_save, old_weight=current_dataset_weight,
                    new_weight=future_dataset_weight)


def train_with_step(complete_training_dataset, step_size, test_dataset, columns_to_drop, ml_model, cluster_class,
                    current_execution, directory_to_save, new_label_value=0, old_label_value=1, old_weight=0.1,
                    new_weight=10):
    number_of_steps = math.ceil(len(complete_training_dataset) / step_size)
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    past_index_step = 0
    all_training_blocks = pd.DataFrame(columns=complete_training_dataset.columns)
    for current_step in range(0, number_of_steps):
        current_index_step = (current_step + 1) * step_size
        if current_index_step > len(complete_training_dataset):
            current_index_step = len(complete_training_dataset) - 1

        new_training_block = complete_training_dataset[past_index_step:current_index_step]

        # if there is any data point with the new weight, then we will pre-process with only other editions with
        # this weight as well.
        same_weight = old_weight
        different_weight = new_weight
        if len(new_training_block[new_training_block["WEIGHT"] == new_weight]) > 0:
            same_weight = new_weight
            different_weight = old_weight

        training_blocks_same_weight = all_training_blocks[all_training_blocks["WEIGHT"] == same_weight]
        training_blocks_different_weight = all_training_blocks[all_training_blocks["WEIGHT"] == different_weight]

        all_new_data_training = pd.concat([new_training_block, training_blocks_same_weight]).sample(frac=1)
        all_new_data_training_standardized, test_dataset_standardized = pre_process(all_new_data_training,
                                                                                    test_dataset)

        all_new_data_training_re_label, only_re_label_dataset = apply_re_label(all_new_data_training_standardized,
                                                                               cluster_class, new_label_value,
                                                                               old_label_value)

        if len(training_blocks_different_weight) > 0:
            training_blocks_different_weight_standardized = pre_process_training(training_blocks_different_weight)
            complete_training_block = pd.concat([training_blocks_different_weight_standardized,
                                                 all_new_data_training_re_label, only_re_label_dataset]).sample(frac=1)
        else:
            complete_training_block = pd.concat([all_new_data_training_re_label, only_re_label_dataset]).sample(frac=1)

        all_balanced_datasets, _ = UtilDataSetFiles.handle_imbalanced_dataset(complete_training_block, "", "", False)

        trained_models = []
        train_execution_time = 0
        for dataset in range(0, len(all_balanced_datasets)):
            balanced_dataset = all_balanced_datasets[dataset]

            x_train = balanced_dataset.drop(columns_to_drop, axis=1)
            y_train = balanced_dataset['LABEL']

            training_weights = []
            dataset_contain_weight = False
            if "WEIGHT" in balanced_dataset.columns.values:
                x_train = x_train.drop(["WEIGHT"], axis=1)

                if same_weight == new_weight:
                    dataset_contain_weight = True
                    training_weights = balanced_dataset['WEIGHT']

            if ml_model == "keras":
                model, elapsed_time = train_with_keras(x_train, y_train, dataset_contain_weight, training_weights)
                trained_models.append(model)
                train_execution_time += elapsed_time
            else:
                raise Exception("ML Model not recognized!")

        information_to_save = {
            "current_execution": current_execution,
            "test_results": {},
            "test_re_label_results": {},
            "train_execution_time": train_execution_time,
            "current_step": current_step
        }

        overall_recall = 0
        if ml_model == "keras":
            if same_weight == new_weight:
                test_dataset_re_label, test_only_re_label_dataset = apply_re_label(test_dataset_standardized,
                                                                                   cluster_class, new_label_value,
                                                                                   old_label_value)

                test_dataset_re_label = test_dataset_re_label.drop(["CLUSTER"], axis=1)
                information_to_save["test_results"] = evaluate_ensemble_models_keras(trained_models,
                                                                                     test_dataset_re_label)
                # save_results_by_step(trained_models, test_dataset_re_label, current_execution, current_step)

                test_only_re_label_dataset = test_only_re_label_dataset.drop(["CLUSTER"], axis=1)
                information_to_save["test_re_label_results"] = evaluate_re_label(trained_models,
                                                                                 test_only_re_label_dataset)
                # save_results_re_label_by_step(trained_models, test_only_re_label_dataset, current_execution,
                # current_step, jump_lines=True)
            else:
                test_dataset_standardized = test_dataset_standardized.drop(["CLUSTER"], axis=1)
                information_to_save["test_results"] = evaluate_ensemble_models_keras(trained_models,
                                                                                     test_dataset_standardized)
                # save_results_by_step(trained_models, test_dataset_standardized, current_execution, current_step)

        current_execution_directory = directory_to_save + str(current_execution)
        if current_step == 0:  # just check to create the directory for the first step.
            create_folder_to_save(base_directory + current_execution_directory)

        information_file_name = "/" + str(current_step) + "_step_test_results.pickle"
        information_file_path = base_directory + current_execution_directory + information_file_name
        pickle.dump(information_to_save, open(information_file_path, 'wb'))

        # If it's the last step, then it's training with the complete dataset, and we want to save this information.
        if current_step == number_of_steps - 1:
            save_keras_model(trained_models, "model", current_execution_directory)

        all_training_blocks = pd.concat([all_training_blocks, new_training_block])
        past_index_step = current_index_step


def generate_k_fold(dataset, directory_to_save, number_of_k_folders=10, number_of_repeats=3,
                    stratified_k_validation=True):
    print("Generating K-Fold started!!!")

    if dataset is None:
        dataset = UtilDataSetFiles.get_data_set_from_file("/experiments/keras/",
                                                          "en_train_no_null_features_2removed.csv")

    x_train = dataset.drop('LABEL', axis=1)
    y_train = pd.Series(data=dataset['LABEL'], dtype='int32')

    repeated_stratified = RepeatedStratifiedKFold(n_splits=number_of_k_folders, n_repeats=number_of_repeats)

    count_split = 0
    for train_index, test_index in repeated_stratified.split(x_train, y_train):
        print("TRAIN: ", train_index, "TEST: ", test_index)
        x_train_split, x_test_split = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_split, y_test_split = y_train[train_index], y_train[test_index]

        # In the current repeated stratified, I already want to separate current and future datasets. In which,
        # the future dataset will suffer the concept drift.
        current_future_stratified = StratifiedKFold(n_splits=2).split(x_train_split, y_train_split)
        current_future_folders = list(current_future_stratified)

        # We are only interested in the first folder, separating the dataset into current and future.
        current_train_indexes = current_future_folders[0][0]
        future_train_indexes = current_future_folders[0][1]

        # Building the dataframes that is going to be saved as a file. We put together the label as well.
        current_train = x_train_split.iloc[current_train_indexes]
        current_train = current_train.assign(LABEL=y_train_split.iloc[current_train_indexes])
        current_train_file_name = str(count_split) + "_current_train" + ".csv"

        future_train = x_train_split.iloc[future_train_indexes]
        future_train = future_train.assign(LABEL=y_train_split.iloc[future_train_indexes])
        future_train_file_name = str(count_split) + "_future_train" + ".csv"

        x_test_split = x_test_split.assign(LABEL=y_test_split)
        test_file_name = str(count_split) + "_test" + ".csv"

        UtilDataSetFiles.save_data_set(current_train, directory_to_save, current_train_file_name)
        UtilDataSetFiles.save_data_set(future_train, directory_to_save, future_train_file_name)
        UtilDataSetFiles.save_data_set(x_test_split,  directory_to_save, test_file_name)

        count_split += 1

        '''
        # Code used just to validate that the split was done correctly.
        number_of_vand_y_train = len(y_train_split[y_train_split == 1])
        number_of_vand_y_test = len(y_test_split[y_test_split == 1])
        number_of_reg_y_train = len(y_train_split[y_train_split == 0])
        number_of_reg_y_test = len(y_test_split[y_test_split == 0])

        print("Vand Y_Train: ", number_of_vand_y_train, "Vand Y_Test:", number_of_vand_y_test, "Reg Y_Train:",
              number_of_reg_y_train, "Reg Y_Test: ", number_of_reg_y_test)

        print("Count: ", count_k_fold_runs)
        '''


def simulate_re_label(dataset_directory, directory_to_save, number_of_datasets):
    base_dataset_to_standard = pd.DataFrame()
    k_means_model = KMeans()

    for dataset_index in range(0, number_of_datasets):
        # current_file_name = str(dataset_index) + "_current_train" + ".csv"
        future_file_name = str(dataset_index) + "_future_train" + ".csv"
        test_file_name = str(dataset_index) + "_test" + ".csv"

        # current_dataset = UtilDataSetFiles.get_data_set_from_file(dataset_directory, current_file_name)
        future_dataset = UtilDataSetFiles.get_data_set_from_file(dataset_directory, future_file_name)
        test_dataset = UtilDataSetFiles.get_data_set_from_file(dataset_directory, test_file_name)

        if dataset_index == 0:
            base_dataset_to_standard = future_dataset
            future_dataset_standardized, test_dataset_standardized = pre_process(future_dataset,
                                                                                 test_dataset,
                                                                                 [57],
                                                                                 [57])

            k_means_model = train_simulation_model(future_dataset_standardized, ["LABEL"])
        else:
            # We use the base dataset to standard to pre-process all the other datasets, using as well the same K-Means
            # model that was built initially. In this way, we keep the cluster information consistent throughout all
            # datasets.
            base_dataset_standardized, future_dataset_standardized = pre_process(base_dataset_to_standard,
                                                                                 future_dataset,
                                                                                 [57, 58],
                                                                                 [57])
            base_dataset_standardized, test_dataset_standardized = pre_process(base_dataset_to_standard,
                                                                               test_dataset,
                                                                               [57, 58],
                                                                               [57])

        # Initializing the values for the column CLUSTER.
        # If it's not changed by the KMeans cluster, then it's -1, not used.
        future_dataset["CLUSTER"] = -1
        test_dataset["CLUSTER"] = -1

        future_dataset = add_cluster_prediction(future_dataset, future_dataset_standardized, k_means_model)
        test_dataset = add_cluster_prediction(test_dataset, test_dataset_standardized, k_means_model)

        # Since I have to pre-process based on future_dataset. I have to do this after I found the cluster classes.
        # These datasets are going to be saved.
        final_future_dataset_standardized, final_test_dataset_standardized = pre_process(future_dataset,
                                                                                         test_dataset,
                                                                                         [57, 58],
                                                                                         [57, 58])

        future_cluster_file_name = str(dataset_index) + "_future_train_cluster" + ".csv"
        future_standardized_file_name = str(dataset_index) + "_future_train_standardized" + ".csv"
        test_cluster_file_name = str(dataset_index) + "_test_cluster" + ".csv"
        test_standardized_file_name = str(dataset_index) + "_test_standardized" + ".csv"

        UtilDataSetFiles.save_data_set(future_dataset, directory_to_save, future_cluster_file_name)
        UtilDataSetFiles.save_data_set(final_future_dataset_standardized, directory_to_save,
                                       future_standardized_file_name)
        UtilDataSetFiles.save_data_set(test_dataset, directory_to_save, test_cluster_file_name)
        UtilDataSetFiles.save_data_set(final_test_dataset_standardized, directory_to_save, test_standardized_file_name)


def train_simulation_model(dataset, columns_to_drop=["LABEL"]):
    train_kmeans_data = dataset[dataset["LABEL"] == 1].drop(columns_to_drop, axis=1)
    model = KMeans(n_clusters=4, init="k-means++", max_iter=1000)
    fitted_model = model.fit(train_kmeans_data)

    return fitted_model


def add_cluster_prediction(dataset, standardized_dataset, k_means_model):
    for vandalism_index, vandalism_edition in standardized_dataset.iterrows():
        if vandalism_edition["LABEL"] == 1:
            data_to_predict = vandalism_edition.drop(["LABEL"])
            data_to_predict = np.array(data_to_predict).reshape(1, -1)
            prediction = k_means_model.predict(data_to_predict)

            # Adding the cluster that the edition belongs to, so we can later re_label it.
            dataset.loc[vandalism_index, "CLUSTER"] = prediction
            #standardized_dataset.loc[vandalism_index, "CLUSTER"] = prediction

    return dataset #, standardized_dataset


def apply_re_label(dataset, cluster_class_to_re_label, new_label_value, old_label_value, replicate_re_label=True):
    dataset.loc[dataset["CLUSTER"] == cluster_class_to_re_label, "LABEL"] = new_label_value
    re_label_data = dataset[dataset["CLUSTER"] == cluster_class_to_re_label]
    number_of_re_labelled_data = len(re_label_data)

    if replicate_re_label and number_of_re_labelled_data > 0:
        number_of_instances_after_re_label = len(dataset[dataset["LABEL"] == old_label_value])
        amount_to_replicate = np.math.ceil(number_of_instances_after_re_label / number_of_re_labelled_data)

        re_label_data_replicated = pd.concat([re_label_data] * amount_to_replicate, ignore_index=True)

        return dataset, re_label_data_replicated
    else:
        return dataset, re_label_data


def get_trained_models(ml_model, number_of_balanced_groups):
    if ml_model == "sparse_lr":
        return load_trained_models("model", "/experiments/article_datasets/trained_sparse_lr_models/",
                                   number_of_balanced_groups)
    elif ml_model == "rf":
        return load_trained_models("model", "/experiments/article_datasets/trained_rf_models/",
                                   number_of_balanced_groups)
    else:
        raise Exception("ML Model not recognized!")


def get_trained_keras_models(ensemble_size, models_path):
    trained_models = []

    folder_name = "model"
    for model_index in range(0, ensemble_size):
        model_folder = models_path + folder_name + str(model_index)
        trained_models.append(keras.models.load_model(model_folder))

    return trained_models


def pre_process(training_dataset, testing_dataset, columns_not_altered=[57], columns_not_altered_test=[57]):
    number_of_columns_training_dataset = len(training_dataset.columns)
    if number_of_columns_training_dataset > 58:
        for i in range(57, number_of_columns_training_dataset):
            columns_not_altered.append(i)

    number_of_columns_testing_dataset = len(testing_dataset.columns)
    if number_of_columns_testing_dataset > 58:
        for i in range(57, number_of_columns_testing_dataset):
            columns_not_altered_test.append(i)

    training_dataset, testing_dataset = get_dataset_no_null_values(training_dataset, testing_dataset,
                                                                   columns_not_altered)

    scaled_training, scaled_values = UtilDataSetFiles.apply_standardization(training_dataset.copy(),
                                                                            columns_not_altered)
    scaled_testing = UtilDataSetFiles.apply_standardization_defined(testing_dataset.copy(), scaled_values,
                                                                    columns_not_altered_test)

    return scaled_training, scaled_testing


def pre_process_training(training_dataset, columns_not_altered=[57]):
    number_of_columns_training_dataset = len(training_dataset.columns)
    if number_of_columns_training_dataset > 58:
        for i in range(57, number_of_columns_training_dataset):
            columns_not_altered.append(i)

    training_dataset, _ = UtilDataSetFiles.handle_missing_values(training_dataset.copy(), columns_not_altered)

    scaled_training, _ = UtilDataSetFiles.apply_standardization(training_dataset.copy(),
                                                                columns_not_altered)

    return scaled_training


def get_dataset_no_null_values(training_dataset, testing_dataset, columns_not_altered=[57]):
    training_dataset, columns_statistical_values = UtilDataSetFiles.handle_missing_values(training_dataset.copy(),
                                                                                          columns_not_altered)
    testing_dataset = UtilDataSetFiles.update_missing_values_media_mode(testing_dataset.copy(),
                                                                        columns_statistical_values)

    return training_dataset, testing_dataset


def train_sparse_lr(x_train, y_train, x_test, y_test):
    model = LogisticRegression(random_state=42, penalty="l1", solver="liblinear", max_iter=15000)
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_test)
    calculated_classification_report = classification_report(y_test, y_predicted)
    print(calculated_classification_report)

    return model.score(x_test, y_test)


def train_sparse_lr_simple(x_train, y_train, train_with_weight=False, weights=[]):
    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=15000)

    if train_with_weight:
        model.fit(x_train, y_train, weights)
    else:
        model.fit(x_train, y_train)

    return model


def train_rf(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_test)
    calculated_classification_report = classification_report(y_test, y_predicted)
    print(calculated_classification_report)

    return model.score(x_test, y_test)


def train_with_keras(x_train, y_train, train_with_weight=False, weights=[]):
    #start_time = process_time()
    start_time = timeit.default_timer()

    model = keras.Sequential([
        layers.Dense(6, activation="relu", input_shape=(57,), name="first_hidden_layer"),
        layers.Dense(3, activation="relu", name="second_hidden_layer"),
        layers.Dense(2, activation="softmax", name="output_layer")
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    if train_with_weight:
        model.fit(x_train, y_train, epochs=512, batch_size=len(x_train), sample_weight=weights, verbose=0)
    else:
        model.fit(x_train, y_train, epochs=512, batch_size=len(x_train), verbose=0)

    #finish_time = process_time()
    finish_time = timeit.default_timer()
    elapsed_time = finish_time - start_time

    return model, elapsed_time


def evaluate_ensemble_models_keras(trained_models, testing_dataset, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = pd.Series(data=testing_dataset['LABEL'], dtype='int32')

    '''
    y_real = []
    for data_point_index in range(0, len(x_test)):
        y_real.append(y_test.iloc[data_point_index])
    '''
    y_real = y_test.values.tolist()
    y_predicted, y_probabilities = evaluate_ensemble_keras(trained_models, x_test)

    calculated_classification_report = classification_report(y_real, y_predicted)
    print(calculated_classification_report)

    calculated_confusion_matrix = confusion_matrix(y_test, y_predicted)
    print(calculated_confusion_matrix)

    corrected_classified_regular = calculated_confusion_matrix[0][0] / (calculated_confusion_matrix[0][0] +
                                                                        calculated_confusion_matrix[0][1])

    corrected_classified_vandalism = calculated_confusion_matrix[1][1] / (calculated_confusion_matrix[1][0] +
                                                                          calculated_confusion_matrix[1][1])

    overall_recall = (corrected_classified_regular + corrected_classified_vandalism) / 2

    print("OVERALL RECALL: " + str(overall_recall))

    test_results = {
        "overall_recall": overall_recall,
        "confusion_matrix": calculated_confusion_matrix,
        "y_real": y_real,
        "y_probabilities": y_probabilities
    }

    # return overall_recall, calculated_confusion_matrix
    return test_results


def save_results_by_step(trained_models, testing_dataset, current_execution, step_index, labels_to_drop=["LABEL"]):
    overall_recall, calculated_confusion_matrix = evaluate_ensemble_models_keras(trained_models, testing_dataset,
                                                                                 labels_to_drop)

    corrected_classified_regular = calculated_confusion_matrix[0][0] / (calculated_confusion_matrix[0][0] +
                                                                        calculated_confusion_matrix[0][1])

    corrected_classified_vandalism = calculated_confusion_matrix[1][1] / (calculated_confusion_matrix[1][0] +
                                                                          calculated_confusion_matrix[1][1])

    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/ExperimentsResults/"
    file_name = "results_k_validation_" + str(current_execution) + ".txt"
    text_file_path = file_path + file_name
    text_file = open(text_file_path, 'a')

    text_file.write("Data Block Index: " + str(step_index) + "\n")
    text_file.write("Overall Recall: " + str(overall_recall) + "\n")
    text_file.write("Regular Recall: " + str(corrected_classified_regular) + "\n")
    text_file.write("Vandalism Recall: " + str(corrected_classified_vandalism) + "\n")

    text_file.write("Correctly Classified Regular: " + str(calculated_confusion_matrix[0][0]) + "\n")
    text_file.write("Incorrectly Classified Regular: " + str(calculated_confusion_matrix[0][1]) + "\n")
    text_file.write("Correctly Classified Vandalism: " + str(calculated_confusion_matrix[1][1]) + "\n")
    text_file.write("Incorrectly Classified Vandalism: " + str(calculated_confusion_matrix[1][0]) + "\n")

    text_file.write("Number of Regular Editions: " + str(calculated_confusion_matrix[0][0] +
                                                         calculated_confusion_matrix[0][1]) + "\n")
    text_file.write("Number of Vandalism Editions: " + str(calculated_confusion_matrix[1][1] +
                                                           calculated_confusion_matrix[1][0]) + "\n\n")

    text_file.close()


def evaluate_re_label(trained_models, testing_dataset, re_label_to=0, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = pd.Series(data=testing_dataset['LABEL'], dtype='int32')

    '''
    y_real = []
    for data_point_index in range(0, len(x_test)):
        y_real.append(y_test.iloc[data_point_index])
    '''
    y_real = y_test.values.tolist()
    y_predicted, y_probabilities = evaluate_ensemble_keras(trained_models, x_test)

    calculated_classification_report = classification_report(y_real, y_predicted)
    print(calculated_classification_report)

    calculated_confusion_matrix = confusion_matrix(y_test, y_predicted)
    print(calculated_confusion_matrix)

    test_results = {
        "confusion_matrix": calculated_confusion_matrix,
        "y_real": y_real,
        "y_probabilities": y_probabilities
    }

    # return calculated_confusion_matrix
    return test_results


def save_results_re_label_by_step(trained_models, testing_dataset, current_execution, step_index, re_label_to=0,
                                  labels_to_drop=["LABEL"], jump_lines=False):
    calculated_confusion_matrix = evaluate_re_label(trained_models, testing_dataset, re_label_to, labels_to_drop)

    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/ExperimentsResults/"
    file_name = "results_k_validation_" + str(current_execution) + ".txt"
    text_file_path = file_path + file_name
    text_file = open(text_file_path, 'a')

    if re_label_to == 0:
        # Then we are dealing with re_label from vandalism to regular.
        no_longer_vandalism_recall = 1
        if len(calculated_confusion_matrix[0]) > 1:
            no_longer_vandalism_recall = calculated_confusion_matrix[0][0] / (calculated_confusion_matrix[0][0] +
                                                                              calculated_confusion_matrix[0][1])

        text_file.write("No Longer Vandalism Recall: " + str(no_longer_vandalism_recall) + "\n")
        text_file.write("Correctly Classified No Longer Vandalism: " + str(calculated_confusion_matrix[0][0]) + "\n")

        if len(calculated_confusion_matrix[0]) > 1:
            text_file.write("Incorrectly Classified No Longer Vandalism: " + str(calculated_confusion_matrix[0][1]) + "\n")

            text_file.write("Number of No Longer Vandalism Editions: " + str(calculated_confusion_matrix[0][0] +
                                                                             calculated_confusion_matrix[0][1]) + "\n\n")
        else:
            text_file.write("Incorrectly Classified No Longer Vandalism: " + str(0) + "\n")
            text_file.write("Number of No Longer Vandalism Editions: " + str(calculated_confusion_matrix[0][0]) + "\n\n")

        if jump_lines:
            text_file.write("\n\n\n\n")

    text_file.close()


def predict_ensemble_models_keras(trained_models, data_to_predict):
    models_predictions = []
    reshaped_data_to_predict = np.array(data_to_predict).reshape(1, -1)

    for model in trained_models:
        models_predictions.append(model.predict(reshaped_data_to_predict)[0])

    classes_highest_values = np.argmax(models_predictions, axis=1)

    return get_ensemble_results(classes_highest_values)


def evaluate_ensemble_keras(trained_models, x_test):
    models_predictions = []

    for model in trained_models:
        predictions = model.predict(x_test)
        #classes_highest_probability = np.argmax(predictions, axis=1)  # This gets the class with highest probability
        models_predictions.append(predictions)

    classes_summed = np.sum(models_predictions, axis=0)
    ensemble_prediction = np.argmax(classes_summed, axis=1)
    classes_probabilities = np.mean(models_predictions, axis=0)

    return ensemble_prediction, classes_probabilities


def get_ensemble_results(models_predictions):
    vandalism_count = 0
    regular_count = 0
    for predictor_result in models_predictions:
        if predictor_result:
            vandalism_count += 1
        else:
            regular_count += 1

    if vandalism_count >= regular_count:
        return 1

    return 0


def get_ensemble_results_proba(models_predictions):
    vandalism_count = 0
    regular_count = 0
    for predictor_result in models_predictions:
        if predictor_result:
            vandalism_count += 1
        else:
            regular_count += 1

    prob_regular = regular_count / (regular_count + vandalism_count)
    prob_vand = vandalism_count / (regular_count + vandalism_count)

    return [prob_regular, prob_vand]


def save_trained_models(trained_models, file_name, path_to_save):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = base_directory + path_to_save + file_name + str(model_index) + ".pickle"
        pickle.dump(model, open(file_path, 'wb'))


def create_folder_to_save(complete_path_to_save):
    if not os.path.exists(complete_path_to_save):
        os.mkdir(complete_path_to_save)


def save_keras_model(trained_models, folder_name, path_to_save):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    complete_path_to_save = base_directory + path_to_save
    create_folder_to_save(complete_path_to_save)

    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = complete_path_to_save + "/" + folder_name + str(model_index)
        model.save(file_path)


def load_trained_models(file_name, models_path, number_of_models):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)
    loaded_models = []

    for model_index in range(0, number_of_models):
        file_path = base_directory + models_path + file_name + str(model_index) + ".pickle"
        model = pickle.load(open(file_path, "rb"))
        loaded_models.append(model)

    return loaded_models

