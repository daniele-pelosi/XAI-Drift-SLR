import math
import os
import pathlib
import pickle
import timeit
from statistics import mean

import numpy as np
import pandas as pd
from keras.applications.densenet import layers
from river.drift import ADWIN
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from norm_detection_online.util import UtilDataSetFiles

tf.compat.v1.disable_eager_execution()


def run_first_experiment():
    # First experiment. No Concept Drift
    train("/experiments/keras/datasets/", "/experiments/keras/trained_models/", "keras", 30, 2, evaluate_by_step=True)


def run_second_experiment():
    # Second experiment. Concept Drift
    train("/experiments/keras/datasets/", "/experiments/keras/trained_models/", "keras", 30, 2, evaluate_by_step=True)


def run_interpretability():
    train_for_interpretability("/experiments/keras/datasets/", "/experiments/keras/trained_models/", "keras", 30, 2,
                               "/experiments/keras/interpretability_data/", "concept_drift", evaluate_by_step=True)


def train(dataset_directory, directory_to_save, ml_model, number_of_executions, cluster_class, batch_size=512,
          max_number_of_classifiers=20, post_balance_ratio=0.5, trade_off_performance_stability=0.5,
          number_of_epochs=100, allowed_changed_in_distribution=0.3, evaluate_by_step=False, new_label_value=0,
          old_label_value=1, columns_to_drop=["LABEL", "CLUSTER"], initial_ensemble_size=12):
    print("Training process started!!!")
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    start_execution = 29
    number_of_executions = 30
    for current_execution in range(start_execution, number_of_executions):
        print("Execution: " + str(current_execution + 1) + " started!!!")

        complete_path_to_save = base_directory + directory_to_save + str(current_execution)
        if not os.path.exists(complete_path_to_save):
            os.mkdir(complete_path_to_save)

        # The datasets were already separated. The same datasets are used for the different training approaches.
        current_dataset = get_working_dataset("current_train", dataset_directory, current_execution)
        future_dataset = get_working_dataset("future_train_cluster", dataset_directory, current_execution)
        test_dataset = get_working_dataset("test_cluster", dataset_directory, current_execution)

        complete_training_dataset = pd.concat([current_dataset, future_dataset])

        # Initialize classifiers with the ensemble of size = initial_ensemble_size
        trained_models, training_time = create_ml_models(initial_ensemble_size, ml_model)

        # execute_online_algorithm(complete_training_dataset, test_dataset, trained_models, cluster_class,
        #                          current_execution, directory_to_save, base_directory)
        execute_online_algorithm_ensemble(complete_training_dataset, test_dataset, trained_models, cluster_class,
                                          current_execution, directory_to_save, base_directory, training_time)


def train_for_interpretability(dataset_directory, directory_to_save, ml_model, number_of_executions, cluster_class,
                               interpretability_path, interpretability_type,
                               batch_size=512, max_number_of_classifiers=20, post_balance_ratio=0.5,
                               trade_off_performance_stability=0.5, number_of_epochs=100,
                               allowed_changed_in_distribution=0.3, evaluate_by_step=False, new_label_value=0,
                               old_label_value=1, columns_to_drop=["LABEL", "CLUSTER"], initial_ensemble_size=12):
    print("Training process started!!!")
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)
    current_execution = 0

    # The datasets were already separated. The same datasets are used for the different training approaches.
    current_dataset = get_working_dataset("current_train", dataset_directory, current_execution)
    future_dataset = get_working_dataset("future_train_cluster", dataset_directory, current_execution)
    test_dataset = get_working_dataset("test_cluster", dataset_directory, current_execution)

    complete_training_dataset = pd.concat([current_dataset, future_dataset])

    # Initialize classifiers with the ensemble of size = initial_ensemble_size
    trained_models = create_ml_models(initial_ensemble_size, ml_model)

    execute_online_algorithm_interpretability(complete_training_dataset, test_dataset, trained_models, cluster_class,
                                              current_execution, directory_to_save, base_directory,
                                              interpretability_path, interpretability_type)


def execute_online_algorithm(training_dataset,
                             testing_dataset,
                             trained_models,
                             cluster_class,
                             current_execution,
                             directory_to_save,
                             base_directory,
                             initial_ensemble_size=12,
                             desired_distribution={0: 0.5, 1: 0.5},
                             sampling_rate=1,
                             threshold_calculate_distribution=512,
                             threshold_evaluate_by_step=512,
                             allowed_change_in_distribution=0.3,
                             max_number_of_classifiers=20,
                             ml_model="keras",
                             new_label_value=0,
                             old_label_value=1,
                             amount_to_replicate=2):
    running_statistical_values = {}
    features = training_dataset.columns.values
    actual_distribution = {0: 0, 1: 0}
    # This is for the general algorithm
    last_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}
    # This is for the evaluation by step (getting results for the paper)
    by_step_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}

    standard_editions = pd.DataFrame()
    no_longer_vandalism_standard = []

    # Initialize running statistics
    for individual_feature in features:
        # Initial array for the feature. The first position is the current mean and the second, the current N.
        # The third position is the number of true, the fourth position number of false. These are used for the mode.
        # The fifth is the sum of squares (used to calculate variance). The sixth is the standard deviation.
        running_statistical_values[individual_feature] = [0, 0, 0, 0, 0, 0]

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0
    number_of_data_points = 0

    training_dataset = training_dataset.replace(False, 0)
    training_dataset = training_dataset.replace("False", 0)
    training_dataset = training_dataset.replace(True, 1)
    training_dataset = training_dataset.replace("True", 1)

    # Preparing to evaluate the algorithm by steps (this is important to show in the article).
    current_evaluation_step = 0
    last_index_evaluation = 0
    no_longer_vandalism_blocks = []
    running_statistical_values_by_step = []
    data_distribution_by_step = []
    re_label_dataset = []
    training_time = 0
    for _ in range(0, math.ceil(len(training_dataset)/threshold_evaluate_by_step)):
        no_longer_vandalism_blocks.append([])
        running_statistical_values_by_step.append({})
        data_distribution_by_step.append({
            0: 0,
            1: 0
        })

    # Preparing the drift detection method
    drift_detector = ADWIN()
    simulated_window = []
    count_editions = 0

    # training_dataset = training_dataset.iloc[14600:]
    # real_number_of_training = [2][12]
    real_number_of_training = np.zeros((12, 2))
    real_number_of_training_vand = [12]

    for edition_index, edition in training_dataset.iterrows():
        real_label = edition["LABEL"]
        real_cluster = edition["CLUSTER"]
        if real_cluster == cluster_class:
            re_label_dataset.append(edition)
            real_label = 0

        edition = edition.drop(["LABEL", "CLUSTER"])

        number_of_data_points += 1

        if real_label:
            number_of_vandalism_editions += 1
        else:
            number_of_regular_editions += 1

        edition_no_null, edition_standardized, running_statistical_values = \
            UtilDataSetFiles.apply_pre_process(edition.copy(), running_statistical_values)

        actual_distribution[real_label] += 1
        data_distribution_by_step[current_evaluation_step][real_label] += 1
        rate = sampling_rate * desired_distribution[real_label] / (
                actual_distribution[real_label] / number_of_data_points)

        for classifier_index in range(0, len(trained_models)):
            random_generator = np.random.RandomState()
            number_of_data_points_to_add = random_generator.poisson(rate)
            # for _ in range(random_generator.poisson(rate)):
            #    number_of_data_points_to_add += 1

            real_number_of_training[classifier_index][int(real_label)] += number_of_data_points_to_add

            # That re_labeled by the community (feedback data).
            if real_cluster == cluster_class:
                no_longer_vandalism_standard.append(edition_standardized)

                # Here we are emphasizing the re_label data by oversampling
                number_of_data_points_to_add += amount_to_replicate

            standardized_inputs = [edition_standardized] * number_of_data_points_to_add
            labels = [real_label] * number_of_data_points_to_add

            if number_of_data_points_to_add > 0:
                keras_output = pd.Series(data=[real_label], dtype='int32')
                for aux_input in standardized_inputs:
                    keras_input = pd.DataFrame([aux_input])
                    trained_models[classifier_index], elapsed_time = train_keras_model(trained_models[classifier_index],
                                                                                       keras_input, keras_output)
                    training_time += elapsed_time

        simulated_window.append(real_label)
        drift_detected, warning = drift_detector.update(real_label)
        if drift_detected:
            window_size = int(drift_detector.width)
            number_items_per_window = int(window_size / 2)
            len_simulated_window = len(simulated_window)

            initial_index_w0 = len_simulated_window - window_size
            del simulated_window[:initial_index_w0]

            w0 = simulated_window[:number_items_per_window]
            w0_mean = mean(w0)
            w0_number_reg = w0.count(0)
            w0_number_vand = w0.count(1)

            w1 = simulated_window[number_items_per_window:]
            w1_mean = mean(w1)
            w1_number_reg = w1.count(0)
            w1_number_vand = w1.count(1)

            # If the old mean is bigger, than it means that we see less of the "1" class. So we increase the desired
            # distribution to see more of the "1" class and keep it more balanced.
            if w0_mean > w1_mean:
                '''if desired_distribution[1] < 0.89:
                    desired_distribution[1] = desired_distribution[1] + ()
                    desired_distribution[0] = 1 - desired_distribution[1]'''
                desired_distribution[1] += 0.2
            else:
                '''if desired_distribution[0] < 0.89:
                    desired_distribution[0] = desired_distribution[0] + 0.2
                    desired_distribution[1] = 1 - desired_distribution[0]'''
                desired_distribution[0] = 0.5

            desired_distribution[0] = 0.5
            desired_distribution[1] = 0.5

            # The desired distribution is proportional inverse to the data distribution.
            # desired_distribution[1] = 1 - (w1_number_vand / w1_number_reg)
            # desired_distribution[0] = 1 - desired_distribution[1]

        '''if count_editions > last_data_distribution["count_index_edition"] + threshold_calculate_distribution:
            last_data_distribution["edition_index"] = edition_index
            last_data_distribution["count_index_edition"] = count_editions

            calculated_actual_distribution = actual_distribution[1] / actual_distribution[0]

            if last_data_distribution["distribution"][1] == 0 or last_data_distribution["distribution"][1] == 0:
                last_data_distribution["distribution"][0] = actual_distribution[0]
                last_data_distribution["distribution"][1] = actual_distribution[1]

            calculated_last_distribution = last_data_distribution["distribution"][1] / \
                                           last_data_distribution["distribution"][0]

            if current_evaluation_step > 28:
                calculated_last_step_data_distribution = data_distribution_by_step[current_evaluation_step-2][1] / \
                                                         data_distribution_by_step[current_evaluation_step-2][0]
                calculated_current_step_data_distribution = data_distribution_by_step[current_evaluation_step-1][1] / \
                                                            data_distribution_by_step[current_evaluation_step-1][0]

                percentage_change = 1 - (calculated_current_step_data_distribution /
                                         calculated_last_step_data_distribution)
                if (calculated_last_step_data_distribution - calculated_current_step_data_distribution > 0) and \
                        (percentage_change < 1 - allowed_change_in_distribution):
                    desired_distribution[1] = 0.5 / percentage_change
                    desired_distribution[0] = 1 - desired_distribution[0]
        '''

        # desired_distribution[0] = 0.2
        # desired_distribution[1] = 0.8
        count_editions += 1

        # This is to get results by steps for the paper, to keep the same scale as mini_batch and batch learning.
        if number_of_data_points % threshold_evaluate_by_step == 0:
            # last_index_evaluation = number_of_data_points

            no_null_testing = UtilDataSetFiles.update_missing_values_media_mode(testing_dataset,
                                                                                running_statistical_values)
            standard_testing = standardize_dataframe(no_null_testing, running_statistical_values)

            only_re_label_test = pd.DataFrame(columns=testing_dataset.columns)
            if len(re_label_dataset) > 0:
                standard_testing, only_re_label_test = apply_re_label(standard_testing, cluster_class,
                                                                      new_label_value, old_label_value)

            run_test_by_step(trained_models, ml_model, training_time, standard_testing, only_re_label_test,
                             current_execution, current_evaluation_step, directory_to_save, base_directory)

            # Restart the training time counter.
            training_time = 0
            current_evaluation_step += 1

    print("finish this!!! TEMP")


def execute_online_algorithm_ensemble(training_dataset,
                                      testing_dataset,
                                      trained_models,
                                      cluster_class,
                                      current_execution,
                                      directory_to_save,
                                      base_directory,
                                      training_time=0,
                                      initial_ensemble_size=12,
                                      desired_distribution={0: 0.5, 1: 0.5},
                                      sampling_rate=1,
                                      threshold_calculate_distribution=512,
                                      threshold_evaluate_by_step=512,
                                      allowed_change_in_distribution=0.3,
                                      max_number_of_classifiers=20,
                                      ml_model="keras",
                                      new_label_value=0,
                                      old_label_value=1,
                                      amount_to_replicate=0):
    running_statistical_values = {}
    features = training_dataset.columns.values
    actual_distribution = {0: 0, 1: 0}
    # This is for the general algorithm
    last_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}
    # This is for the evaluation by step (getting results for the paper)
    by_step_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}

    standard_editions = pd.DataFrame()
    no_longer_vandalism_standard = []

    # Initialize running statistics
    for individual_feature in features:
        # Initial array for the feature. The first position is the current mean and the second, the current N.
        # The third position is the number of true, the fourth position number of false. These are used for the mode.
        # The fifth is the sum of squares (used to calculate variance). The sixth is the standard deviation.
        running_statistical_values[individual_feature] = [0, 0, 0, 0, 0, 0]

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0
    number_of_data_points = 0

    training_dataset = training_dataset.replace(False, 0)
    training_dataset = training_dataset.replace("False", 0)
    training_dataset = training_dataset.replace(True, 1)
    training_dataset = training_dataset.replace("True", 1)

    # Preparing to evaluate the algorithm by steps (this is important to show in the article).
    current_evaluation_step = 0
    last_index_evaluation = 0
    no_longer_vandalism_blocks = []
    running_statistical_values_by_step = []
    data_distribution_by_step = []
    re_label_dataset = []
    # training_time = 0
    for _ in range(0, math.ceil(len(training_dataset)/threshold_evaluate_by_step)):
        no_longer_vandalism_blocks.append([])
        running_statistical_values_by_step.append({})
        data_distribution_by_step.append({
            0: 0,
            1: 0
        })

    # Preparing the drift detection method
    drift_detector = ADWIN()
    simulated_window = []
    count_editions = 0

    # Fixed size because we are interested in the last editions, if the widow size is too big, it gets information from
    # past editions that are not of interest to us.
    window_fixed_size = 64
    old_drift_window = []
    new_drift_window = []
    drift_window = []
    under_sample_majority = False

    # training_dataset = training_dataset.iloc[14600:]
    # real_number_of_training = [2][12]
    real_number_of_training = np.zeros((12, 2))
    real_number_of_training_vand = [12]

    index_classifier_to_train = 0
    index_classifier_re_label = 0
    number_of_data_points_to_add = 1

    # These value must be updated as the relation between majority and minority class changes (what was the majority
    # class at one point becomes minority class as interactions happen).
    majority_class = 0
    minority_class = 1

    for edition_index, edition in training_dataset.iterrows():
        real_label = edition["LABEL"]
        real_cluster = edition["CLUSTER"]
        if real_cluster == cluster_class:
            re_label_dataset.append(edition)
            real_label = 0

        edition = edition.drop(["LABEL", "CLUSTER"])

        number_of_data_points += 1

        if real_label:
            number_of_vandalism_editions += 1
        else:
            number_of_regular_editions += 1

        edition_no_null, edition_standardized, running_statistical_values = \
            UtilDataSetFiles.apply_pre_process(edition.copy(), running_statistical_values)

        actual_distribution[real_label] += 1
        data_distribution_by_step[current_evaluation_step][real_label] += 1
        rate = sampling_rate * desired_distribution[real_label] / (
                actual_distribution[real_label] / number_of_data_points)

        amount_of_data_points_added = 1
        # if the current edition is of the majority class, then we train only one classifier.
        if real_label == majority_class:
            if not under_sample_majority: # or real_cluster == cluster_class:
                majority_class_to_add = 1
                # That re_labeled by the community (feedback data).
                if real_cluster == cluster_class:
                    no_longer_vandalism_standard.append(edition_standardized)

                    # Here we are emphasizing the re_label data by oversampling
                    majority_class_to_add += amount_to_replicate

                standardized_inputs = [edition_standardized] * majority_class_to_add
                labels = [real_label] * majority_class_to_add

                # If it's relabel, then we train half of the classifiers + 1. If normal edition, then only one classifier.
                if real_cluster == cluster_class:  # REMOVE second condition.
                    '''for classifier_index in range(0, len(trained_models)):
                        real_number_of_training[classifier_index][int(real_label)] += majority_class_to_add
                        keras_output = pd.Series(data=[real_label], dtype='int32')
                        for aux_input in standardized_inputs:
                            keras_input = pd.DataFrame([aux_input])
                            trained_models[classifier_index], elapsed_time = \
                                train_keras_model(trained_models[classifier_index], keras_input, keras_output)
                            training_time += elapsed_time

                    # This is important for the balance ratio. Adding repeatdly the re_label data, unbalace the dataset
                    amount_of_data_points_added = len(trained_models)'''
                    classifiers_indexes = []
                    for count in range(0, 7):
                        keras_output = pd.Series(data=[real_label], dtype='int32')
                        real_number_of_training[index_classifier_to_train][int(real_label)] += majority_class_to_add
                        for aux_input in standardized_inputs:
                            keras_input = pd.DataFrame([aux_input])
                            trained_models[index_classifier_to_train], elapsed_time = \
                                train_keras_model(trained_models[index_classifier_to_train], keras_input, keras_output)
                            training_time += elapsed_time

                        if index_classifier_to_train < (len(trained_models) - 1):
                            index_classifier_to_train += 1
                        else:
                            index_classifier_to_train = 0
                    amount_of_data_points_added = 7
                    '''keras_output = pd.Series(data=[real_label], dtype='int32')
                    real_number_of_training[index_classifier_re_label][int(real_label)] += majority_class_to_add
                    for aux_input in standardized_inputs:
                        keras_input = pd.DataFrame([aux_input])
                        trained_models[index_classifier_re_label], elapsed_time = \
                            train_keras_model(trained_models[index_classifier_re_label], keras_input, keras_output)
                        training_time += elapsed_time

                    if index_classifier_re_label < (len(trained_models) - 1):
                        index_classifier_re_label += 1
                    else:
                        index_classifier_re_label = 0'''
                else:
                    keras_output = pd.Series(data=[real_label], dtype='int32')
                    real_number_of_training[index_classifier_to_train][int(real_label)] += majority_class_to_add
                    for aux_input in standardized_inputs:
                        keras_input = pd.DataFrame([aux_input])
                        trained_models[index_classifier_to_train], elapsed_time = \
                            train_keras_model(trained_models[index_classifier_to_train], keras_input, keras_output)
                        training_time += elapsed_time

                    if index_classifier_to_train < (len(trained_models) - 1):
                        index_classifier_to_train += 1
                    else:
                        index_classifier_to_train = 0
            else:
                amount_of_data_points_added = 0

        else:
            # if the current edition is the minority class, then we train all classifiers.
            # That re_labeled by the community (feedback data).
            if real_cluster == cluster_class:
                no_longer_vandalism_standard.append(edition_standardized)

                # Here we are emphasizing the re_label data by oversampling
                number_of_data_points_to_add += amount_to_replicate

            standardized_inputs = [edition_standardized] * number_of_data_points_to_add
            labels = [real_label] * number_of_data_points_to_add
            amount_of_data_points_added = number_of_data_points_to_add

            for classifier_index in range(0, len(trained_models)):
                real_number_of_training[classifier_index][int(real_label)] += number_of_data_points_to_add
                keras_output = pd.Series(data=[real_label], dtype='int32')
                for aux_input in standardized_inputs:
                    keras_input = pd.DataFrame([aux_input])
                    trained_models[classifier_index], elapsed_time = \
                        train_keras_model(trained_models[classifier_index], keras_input, keras_output)
                    training_time += elapsed_time

        # real_number_of_training[index_classifier_to_train][int(real_label)] += number_of_data_points_to_add

        '''simulated_window.append(real_label)
        drift_detected, warning = drift_detector.update(real_label)
        if drift_detected:
            window_size = int(drift_detector.width)
            number_items_per_window = int(window_size / 2)
            len_simulated_window = len(simulated_window)

            initial_index_w0 = len_simulated_window - window_size
            del simulated_window[:initial_index_w0]

            w0 = simulated_window[:number_items_per_window]
            w0_mean = mean(w0)
            w0_number_reg = w0.count(0)
            w0_number_vand = w0.count(1)

            w1 = simulated_window[number_items_per_window:]
            w1_mean = mean(w1)
            w1_number_reg = w1.count(0)
            w1_number_vand = w1.count(1)

            # If the old mean is bigger, than it means that we see less of the "1" class. So we increase the desired
            # distribution to see more of the "1" class and keep it more balanced.
            # if w0_mean > w1_mean:

            imbalance_difference = int(w1_number_reg / w1_number_vand)
            number_of_classifiers = len(trained_models)
            if imbalance_difference > number_of_classifiers:
                number_of_data_points_to_add = imbalance_difference - number_of_classifiers
            else:
                number_of_data_points_to_add = number_of_classifiers - imbalance_difference
        '''

        # Delete the first item on the window, and add the new item.
        if len(new_drift_window) >= window_fixed_size:
            # Delete from the simulated window the same amount that is going to be added.
            del new_drift_window[0:amount_of_data_points_added]
            # new_drift_window.clear()

        # Replicating the labels that were added due to the replication of re_label editions.
        new_drift_window.extend([real_label]*amount_of_data_points_added)
        drift_detected, updated_amount_to_replicate, under_sample_majority = detect_drift_one_window(new_drift_window,
                                                                                                     len(trained_models),
                                                                                                     number_of_data_points_to_add,
                                                                                                     window_fixed_size)
        if drift_detected:
            number_of_data_points_to_add = updated_amount_to_replicate

        count_editions += 1
        # number_of_data_points_to_add = 3

        # This is to get results by steps for the paper, to keep the same scale as mini_batch and batch learning.
        if number_of_data_points % threshold_evaluate_by_step == 0 or number_of_data_points == len(training_dataset):
            # last_index_evaluation = number_of_data_points
            '''a = real_number_of_training[classifier_index][0]
            b = data_distribution_by_step[0][0]
            c = len(no_longer_vandalism_standard)
            for i in range(0, len(trained_models)):
                real_number_of_training[i][0] = 0
                real_number_of_training[i][1] = 0'''

            no_null_testing = UtilDataSetFiles.update_missing_values_media_mode(testing_dataset,
                                                                                running_statistical_values)
            standard_testing = standardize_dataframe(no_null_testing, running_statistical_values)

            only_re_label_test = pd.DataFrame(columns=testing_dataset.columns)
            if len(re_label_dataset) > 0:
                standard_testing, only_re_label_test = apply_re_label(standard_testing, cluster_class,
                                                                      new_label_value, old_label_value)

            run_test_by_step(trained_models, ml_model, training_time, standard_testing, only_re_label_test,
                             current_execution, current_evaluation_step, directory_to_save, base_directory)

            # Restart the training time counter.
            print("Training Time: " + str(training_time))
            training_time = 0
            current_evaluation_step += 1

    current_execution_directory = directory_to_save + str(current_execution)
    save_keras_model(trained_models, "model", current_execution_directory)

    print("finish this!!! TEMP")


def execute_online_algorithm_interpretability(training_dataset,
                                              testing_dataset,
                                              trained_models,
                                              cluster_class,
                                              current_execution,
                                              directory_to_save,
                                              base_directory,
                                              interpretability_path,
                                              interpretability_type,
                                              initial_ensemble_size=12,
                                              desired_distribution={0: 0.5, 1: 0.5},
                                              sampling_rate=1,
                                              threshold_calculate_distribution=512,
                                              threshold_evaluate_by_step=512,
                                              allowed_change_in_distribution=0.3,
                                              max_number_of_classifiers=20,
                                              ml_model="keras",
                                              new_label_value=0,
                                              old_label_value=1,
                                              amount_to_replicate=0):
    running_statistical_values = {}
    features = training_dataset.columns.values
    actual_distribution = {0: 0, 1: 0}
    # This is for the general algorithm
    last_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}
    # This is for the evaluation by step (getting results for the paper)
    by_step_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}

    standard_editions = pd.DataFrame()
    no_longer_vandalism_standard = []

    # Initialize running statistics
    for individual_feature in features:
        # Initial array for the feature. The first position is the current mean and the second, the current N.
        # The third position is the number of true, the fourth position number of false. These are used for the mode.
        # The fifth is the sum of squares (used to calculate variance). The sixth is the standard deviation.
        running_statistical_values[individual_feature] = [0, 0, 0, 0, 0, 0]

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0
    number_of_data_points = 0

    training_dataset = training_dataset.replace(False, 0)
    training_dataset = training_dataset.replace("False", 0)
    training_dataset = training_dataset.replace(True, 1)
    training_dataset = training_dataset.replace("True", 1)

    # Preparing to evaluate the algorithm by steps (this is important to show in the article).
    current_evaluation_step = 0
    last_index_evaluation = 0
    no_longer_vandalism_blocks = []
    running_statistical_values_by_step = []
    data_distribution_by_step = []
    re_label_dataset = []
    training_time = 0
    for _ in range(0, math.ceil(len(training_dataset)/threshold_evaluate_by_step)):
        no_longer_vandalism_blocks.append([])
        running_statistical_values_by_step.append({})
        data_distribution_by_step.append({
            0: 0,
            1: 0
        })

    # Preparing the drift detection method
    drift_detector = ADWIN()
    simulated_window = []
    count_editions = 0

    # Fixed size because we are interested in the last editions, if the widow size is too big, it gets information from
    # past editions that are not of interest to us.
    window_fixed_size = 64
    old_drift_window = []
    new_drift_window = []
    drift_window = []
    under_sample_majority = False

    # training_dataset = training_dataset.iloc[14600:]
    # real_number_of_training = [2][12]
    real_number_of_training = np.zeros((12, 2))
    real_number_of_training_vand = [12]

    index_classifier_to_train = 0
    index_classifier_re_label = 0
    number_of_data_points_to_add = 1

    # These value must be updated as the relation between majority and minority class changes (what was the majority
    # class at one point becomes minority class as interactions happen).
    majority_class = 0
    minority_class = 1

    last_complete_data_block = []
    last_complete_data_block_standardized = []
    current_complete_data_block = []
    current_complete_data_block_standardized = []
    last_no_null_testing = pd.DataFrame(columns=testing_dataset.columns)
    last_no_null_testing_standardized = pd.DataFrame(columns=testing_dataset.columns)
    last_running_statistical_values = 0
    current_index_complete_data_block = 0

    for edition_index, edition in training_dataset.iterrows():
        real_label = edition["LABEL"]
        real_cluster = edition["CLUSTER"]
        if real_cluster == cluster_class:
            re_label_dataset.append(edition)
            real_label = 0

        edition = edition.drop(["LABEL", "CLUSTER"])

        number_of_data_points += 1

        if real_label:
            number_of_vandalism_editions += 1
        else:
            number_of_regular_editions += 1

        edition_no_null, edition_standardized, running_statistical_values = \
            UtilDataSetFiles.apply_pre_process(edition.copy(), running_statistical_values)

        current_complete_data_block.append(edition_no_null.copy())
        current_complete_data_block[current_index_complete_data_block]["LABEL"] = real_label
        current_complete_data_block[current_index_complete_data_block]["CLUSTER"] = real_cluster

        current_complete_data_block_standardized.append(edition_standardized.copy())
        current_complete_data_block_standardized[current_index_complete_data_block]["LABEL"] = real_label
        current_complete_data_block_standardized[current_index_complete_data_block]["CLUSTER"] = real_cluster

        current_index_complete_data_block += 1

        actual_distribution[real_label] += 1
        data_distribution_by_step[current_evaluation_step][real_label] += 1
        rate = sampling_rate * desired_distribution[real_label] / (
                actual_distribution[real_label] / number_of_data_points)

        amount_of_data_points_added = 1
        # if the current edition is of the majority class, then we train only one classifier.
        if real_label == majority_class:
            if not under_sample_majority: # or real_cluster == cluster_class:
                majority_class_to_add = 1
                # That re_labeled by the community (feedback data).
                if real_cluster == cluster_class:
                    no_longer_vandalism_standard.append(edition_standardized)

                    # Here we are emphasizing the re_label data by oversampling
                    majority_class_to_add += amount_to_replicate

                standardized_inputs = [edition_standardized] * majority_class_to_add
                labels = [real_label] * majority_class_to_add

                # If it's relabel, then we train all classifiers. If normal edition, then only one classifier.
                if real_cluster == cluster_class:  # REMOVE second condition.
                    classifiers_indexes = []
                    for count in range(0, 7):
                        keras_output = pd.Series(data=[real_label], dtype='int32')
                        real_number_of_training[index_classifier_to_train][int(real_label)] += majority_class_to_add
                        for aux_input in standardized_inputs:
                            keras_input = pd.DataFrame([aux_input])
                            trained_models[index_classifier_to_train], elapsed_time = \
                                train_keras_model(trained_models[index_classifier_to_train], keras_input, keras_output)
                            training_time += elapsed_time

                        if index_classifier_to_train < (len(trained_models) - 1):
                            index_classifier_to_train += 1
                        else:
                            index_classifier_to_train = 0
                    amount_of_data_points_added = 7
                else:
                    keras_output = pd.Series(data=[real_label], dtype='int32')
                    real_number_of_training[index_classifier_to_train][int(real_label)] += majority_class_to_add
                    for aux_input in standardized_inputs:
                        keras_input = pd.DataFrame([aux_input])
                        trained_models[index_classifier_to_train], elapsed_time = \
                            train_keras_model(trained_models[index_classifier_to_train], keras_input, keras_output)
                        training_time += elapsed_time

                    if index_classifier_to_train < (len(trained_models) - 1):
                        index_classifier_to_train += 1
                    else:
                        index_classifier_to_train = 0
            else:
                amount_of_data_points_added = 0

        else:
            # if the current edition is the minority class, then we train all classifiers.
            # That re_labeled by the community (feedback data).
            if real_cluster == cluster_class:
                no_longer_vandalism_standard.append(edition_standardized)

                # Here we are emphasizing the re_label data by oversampling
                number_of_data_points_to_add += amount_to_replicate

            standardized_inputs = [edition_standardized] * number_of_data_points_to_add
            labels = [real_label] * number_of_data_points_to_add
            amount_of_data_points_added = number_of_data_points_to_add

            for classifier_index in range(0, len(trained_models)):
                real_number_of_training[classifier_index][int(real_label)] += number_of_data_points_to_add
                keras_output = pd.Series(data=[real_label], dtype='int32')
                for aux_input in standardized_inputs:
                    keras_input = pd.DataFrame([aux_input])
                    trained_models[classifier_index], elapsed_time = \
                        train_keras_model(trained_models[classifier_index], keras_input, keras_output)
                    training_time += elapsed_time

        # real_number_of_training[index_classifier_to_train][int(real_label)] += number_of_data_points_to_add

        # Delete the first item on the window, and add the new item.
        if len(new_drift_window) >= window_fixed_size:
            # Delete from the simulated window the same amount that is going to be added.
            del new_drift_window[0:amount_of_data_points_added]
            # new_drift_window.clear()

        # Replicating the labels that were added due to the replication of re_label editions.
        new_drift_window.extend([real_label]*amount_of_data_points_added)
        drift_detected, updated_amount_to_replicate, under_sample_majority = detect_drift_one_window(new_drift_window,
                                                                                                     len(trained_models),
                                                                                                     number_of_data_points_to_add,
                                                                                                     window_fixed_size)
        if drift_detected:
            number_of_data_points_to_add = updated_amount_to_replicate

        count_editions += 1
        # number_of_data_points_to_add = 3

        # This is to get results by steps for the paper, to keep the same scale as mini_batch and batch learning.
        if number_of_data_points % threshold_evaluate_by_step == 0 or number_of_data_points == len(training_dataset):
            # last_index_evaluation = number_of_data_points

            if interpretability_type == "no_concept_drift" and len(re_label_dataset) > 0:
                break
            if interpretability_type == "concept_drift" and number_of_data_points == len(training_dataset):
                break

            last_complete_data_block = pd.DataFrame(columns=training_dataset.columns, data=current_complete_data_block)
            # last_complete_data_block = pd.concat(current_complete_data_block.copy())
            last_complete_data_block_standardized = pd.DataFrame(columns=training_dataset.columns,
                                                                 data=current_complete_data_block_standardized)
            # last_complete_data_block_standardized = pd.concat(current_complete_data_block_standardized.copy())

            current_complete_data_block = []
            current_complete_data_block_standardized = []

            last_running_statistical_values = running_statistical_values

            if len(re_label_dataset) > 0 and interpretability_type == "concept_drift":
                testing_dataset, _ = apply_re_label(testing_dataset, cluster_class, new_label_value, old_label_value)

            last_no_null_testing = UtilDataSetFiles.update_missing_values_media_mode(testing_dataset.copy(),
                                                                                     last_running_statistical_values)

            last_no_null_testing_standardized = standardize_dataframe(last_no_null_testing.copy(),
                                                                      last_running_statistical_values)

            current_index_complete_data_block = 0

            # Restart the training time counter.
            training_time = 0
            current_evaluation_step += 1

    directory_interpretability = base_directory + interpretability_path + interpretability_type
    save_interpretability_data(last_complete_data_block, last_complete_data_block_standardized, last_no_null_testing,
                               last_no_null_testing_standardized, last_running_statistical_values,
                               directory_interpretability, trained_models)

    print("finish this!!! TEMP")


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


def detect_drift_one_window(new_drift_window, balance_goal, last_amount_to_replicate, window_fixed_size=512):
    if len(new_drift_window) < window_fixed_size:
        return False, 1, False

    new_number_reg = new_drift_window.count(0)
    new_number_vand = new_drift_window.count(1)
    under_sample_majority = False

    optimal_number_vand = new_number_reg / balance_goal
    if new_number_vand != 0:
        ratio_optimal_real = optimal_number_vand / new_number_vand
        amount_to_replicate = round(ratio_optimal_real)
        if amount_to_replicate == 0:
            amount_to_replicate = 1
    else:
        amount_to_replicate = round(optimal_number_vand)  # last_amount_to_replicate + 1

    # If it's necessary more than double the minority class, then we are under sampling the majority class.
    if amount_to_replicate > 2:
        under_sample_majority = True

    # if the amount_to_replicate is different from the last replication vale, then it means that a concept
    # drift was detected, since the data distribution changed.
    if amount_to_replicate != last_amount_to_replicate:
        return True, amount_to_replicate, under_sample_majority

    return False, amount_to_replicate, under_sample_majority


def detect_drift_fixed_window(old_drift_window, new_drift_window, balance_goal):
    if len(old_drift_window) != len(new_drift_window):
        return False, 1

    old_window_mean = mean(old_drift_window)
    old_number_reg = old_drift_window.count(0)
    old_number_vand = old_drift_window.count(1)
    old_balance_value = old_number_reg / old_number_vand

    new_window_mean = mean(new_drift_window)
    new_number_reg = new_drift_window.count(0)
    new_number_vand = new_drift_window.count(1)
    new_balance_value = new_number_reg / new_number_vand

    optimal_number_vand = new_number_reg / balance_goal
    ratio_optimal_real = optimal_number_vand / new_number_vand
    amount_to_replicate = round(ratio_optimal_real)

    # if the amount_to_replicate is bigger than 1, then it means that a concept drift was detected, since the data
    # distribution changed.
    if amount_to_replicate > 1:
        return True, amount_to_replicate

    return False, 1


def run_test_by_step(trained_models, ml_model, train_execution_time, current_testing, only_re_label_testing,
                     current_execution, current_evaluation_step, directory_to_save, base_directory):
    information_to_save = {
        "current_execution": current_execution,
        "test_results": {},
        "test_re_label_results": {},
        "train_execution_time": train_execution_time,
        "current_step": current_evaluation_step
    }

    information_to_save["test_results"] = evaluate_ensemble_models_keras(trained_models, current_testing,
                                                                         ["LABEL", "CLUSTER"])

    if len(only_re_label_testing) > 0:
        information_to_save["test_re_label_results"] = evaluate_re_label_keras(trained_models, only_re_label_testing,
                                                                               ["LABEL", "CLUSTER"])

    current_execution_directory = directory_to_save + str(current_execution)
    information_file_name = "/" + str(current_evaluation_step) + "_step_test_results.pickle"
    information_file_path = base_directory + current_execution_directory + information_file_name
    pickle.dump(information_to_save, open(information_file_path, 'wb'))


def standardize_dataframe(dataframe_to_standardize, running_statistical_values):
    all_edition_standardized = []
    for edition_index, edition in dataframe_to_standardize.iterrows():
        temp_edition_standard = pd.DataFrame([UtilDataSetFiles.apply_standardization(edition,
                                                                                     running_statistical_values)])
        all_edition_standardized.append(temp_edition_standard)

    return pd.concat(all_edition_standardized)


def apply_re_label(dataset, cluster_class_to_re_label, new_label_value, old_label_value, replicate_re_label=False):
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


def evaluate_ensemble_models_keras(trained_models, testing_dataset, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = pd.Series(data=testing_dataset['LABEL'], dtype='int32')

    y_real = y_test.values.tolist()
    y_predicted, y_probabilities = get_keras_predictions(trained_models, x_test)

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

    return test_results


def evaluate_re_label_keras(trained_models, testing_dataset, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = pd.Series(data=testing_dataset['LABEL'], dtype='int32')

    y_real = y_test.values.tolist()
    y_predicted, y_probabilities = get_keras_predictions(trained_models, x_test)

    calculated_classification_report = classification_report(y_real, y_predicted, zero_division=0)
    print(calculated_classification_report)

    calculated_confusion_matrix = confusion_matrix(y_test, y_predicted)
    print(calculated_confusion_matrix)

    test_results = {
        "confusion_matrix": calculated_confusion_matrix,
        "y_real": y_real,
        "y_probabilities": y_probabilities
    }

    return test_results


def get_keras_predictions(trained_models, x_test):
    models_predictions = []

    for model in trained_models:
        predictions = model.predict(x_test)
        models_predictions.append(predictions)

    classes_summed = np.sum(models_predictions, axis=0)
    ensemble_prediction = np.argmax(classes_summed, axis=1)
    classes_probabilities = np.mean(models_predictions, axis=0)

    return ensemble_prediction, classes_probabilities


def create_ml_models(ensemble_size, ml_model):
    ensemble = []
    training_time = 0

    for _ in range(0, ensemble_size):
        if ml_model == "keras":
            classifier, time_aux = create_keras_model()
            ensemble.append(classifier)
            training_time += time_aux

    return ensemble, training_time


def create_keras_model():
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

    finish_time = timeit.default_timer()
    elapsed_time = finish_time - start_time

    return model, elapsed_time


def train_keras_model(trained_model, x_train, y_train):
    # batch_size has the size of x_train (for the online learning case, x_train contains one data point that was
    # replicated according to the re-sampling strategy).
    # batch_size=len(x_train)
    start_time = timeit.default_timer()
    trained_model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

    finish_time = timeit.default_timer()
    elapsed_time = finish_time - start_time

    return trained_model, elapsed_time


def get_working_dataset(file_name, directory, index):
    complete_file_name = str(index) + "_" + file_name + ".csv"
    dataset = UtilDataSetFiles.get_data_set_from_file(directory, complete_file_name)

    return dataset


def save_keras_model(trained_models, folder_name, path_to_save):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    complete_path_to_save = base_directory + path_to_save
    if not os.path.exists(complete_path_to_save):
        os.mkdir(complete_path_to_save)

    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = complete_path_to_save + "/" + folder_name + str(model_index)
        model.save(file_path)

