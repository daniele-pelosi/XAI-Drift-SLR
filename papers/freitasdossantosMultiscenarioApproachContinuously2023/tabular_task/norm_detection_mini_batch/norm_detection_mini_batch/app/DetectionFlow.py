import gc
import math
import os
import pathlib
import pickle
import timeit
from time import process_time

import numpy as np
import pandas as pd
from keras.applications.densenet import layers
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

from norm_detection_mini_batch.util import UtilDataSetFiles

tf.compat.v1.disable_eager_execution()


def start_process():
    # First experiment. No Concept Drift
    train("/experiments/keras/datasets/", "/experiments/keras/trained_models/", "keras", 30, 2, evaluate_by_step=True)


def start_process_interpretability():
    train_for_interpretability("/experiments/keras/datasets/", "/experiments/keras/trained_models/", "keras",
                               "/experiments/keras/interpretability_data/", "concept_drift",  30, 2,
                               evaluate_by_step=False)


def train(dataset_directory, directory_to_save, ml_model, number_of_executions, cluster_class, batch_size=512,
          max_number_of_classifiers=12, post_balance_ratio=0.5, trade_off_performance_stability=0.5,
          number_of_epochs=128, allowed_changed_in_distribution=0.3, evaluate_by_step=False, new_label_value=0,
          old_label_value=1, columns_to_drop=["LABEL", "CLUSTER"]):
    print("Training process started!!!")
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    start_execution = 0
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

        # Separating the training dataset into batches that are going to be presented to the model for training.
        number_of_data_blocks = math.ceil(len(complete_training_dataset) / batch_size)
        mini_batches = np.array_split(complete_training_dataset, number_of_data_blocks)

        trained_models = []

        # It will contain data to evaluate how the data distribution changed in time.
        data_distribution_in_time = []
        time_stamp_last_ensemble_change = 0

        # It will contain data points that suffered re_label. Later it will be used to check no longer vandalism.
        re_label_all_blocks = pd.DataFrame(columns=complete_training_dataset.columns)

        # All data block should contain data points representing the minority class. This is only to ensure that.
        last_vandalism_set = pd.DataFrame(columns=complete_training_dataset.columns)

        total_training_time = 0
        for index_current_data_block, current_data_block in zip(range(0, number_of_data_blocks), mini_batches):
            # and index_current_data_block > 32  # Use this line for testing
            if len(current_data_block) > 0:
                # Applying pre-process chunk by chunk. Only considering present data.
                columns_not_altered = [57, 58]  # Label and Cluster columns should not be changed.
                no_null_block, statistical_values = UtilDataSetFiles.handle_missing_values(current_data_block,
                                                                                           columns_not_altered)
                standard_current_block, scaled_values = UtilDataSetFiles.apply_standardization(no_null_block.copy(),
                                                                                               columns_not_altered)

                no_null_testing = UtilDataSetFiles.update_missing_values_media_mode(test_dataset, statistical_values)
                standard_testing = UtilDataSetFiles.apply_standardization_defined(no_null_testing.copy(), scaled_values,
                                                                                  [57, 58])

                # Simulating the re-label feedback based on the clusters found by K-Means
                current_block_re_label, only_re_label_current_block = apply_re_label(standard_current_block,
                                                                                     cluster_class, new_label_value,
                                                                                     old_label_value,
                                                                                     replicate_re_label=False)

                # Adding the current re_label data to the DT with all no longer vandalism data points.
                re_label_all_blocks = pd.concat([re_label_all_blocks, only_re_label_current_block])

                current_number_of_minority_instances = len(current_block_re_label[current_block_re_label["LABEL"] == 1])
                current_number_of_majority_instances = len(current_block_re_label[current_block_re_label["LABEL"] == 0])

                # This is done just to ensure that all data blocks contain minority instances.
                if current_number_of_minority_instances <= 0:
                    current_number_of_minority_instances = len(last_vandalism_set)
                    current_block_re_label = pd.concat([current_block_re_label, last_vandalism_set])

                # Getting the best number of classifiers that should be in the ensemble.
                current_best_number_of_classifiers = math.ceil(current_number_of_majority_instances /
                                                               current_number_of_minority_instances)

                if current_best_number_of_classifiers > max_number_of_classifiers:
                    # In this case, the oversampling technique is solving the imbalance.
                    # current_best_number_of_classifiers = max_number_of_classifiers

                    current_block_re_label, current_best_number_of_classifiers = duplicate_minority_instances(
                        current_block_re_label, max_number_of_classifiers, current_number_of_majority_instances)

                    current_number_of_minority_instances = len(current_block_re_label[current_block_re_label["LABEL"] == 1])

                current_data_distribution = current_number_of_minority_instances / current_number_of_majority_instances
                data_distribution_in_time.append(current_data_distribution)

                number_of_new_classifiers = 0
                '''data_distribution_to_compare = data_distribution_in_time[time_stamp_last_ensemble_change]
                if (data_distribution_to_compare - current_data_distribution > 0) and \
                        (current_data_distribution / data_distribution_to_compare < 1 - allowed_changed_in_distribution):
                    number_of_new_classifiers = current_best_number_of_classifiers - len(trained_models)
                    time_stamp_last_ensemble_change = index_current_data_block
                    # time_stamp_last_ensemble_change = 2
                '''

                current_ensemble_size = len(trained_models)
                if current_best_number_of_classifiers > current_ensemble_size:
                    number_of_new_classifiers = current_best_number_of_classifiers - current_ensemble_size

                number_of_re_label = len(only_re_label_current_block)
                if number_of_re_label > 0:
                    amount_to_replicate = round(current_number_of_minority_instances / number_of_re_label)
                    if amount_to_replicate > 0 and number_of_re_label < current_number_of_minority_instances:
                        re_label_replicated = pd.concat([only_re_label_current_block] * amount_to_replicate,
                                                        ignore_index=True)
                        number_replicated = len(re_label_replicated)

                        reg_editions = current_block_re_label[current_block_re_label["LABEL"] == 0]
                        reg_editions = reg_editions[reg_editions["CLUSTER"] != 2]
                        reg_editions = reg_editions[number_replicated:]

                        vand_editions = current_block_re_label[current_block_re_label["LABEL"] == 1]

                        current_block_re_label = pd.concat([reg_editions, vand_editions, re_label_replicated])\
                            .sample(frac=1)

                current_block_re_label = current_block_re_label.drop(["CLUSTER"], axis=1)

                trained_models, training_time = start_modified_due(trained_models, ml_model, current_block_re_label,
                                                                   number_of_new_classifiers, post_balance_ratio,
                                                                   number_of_epochs, current_best_number_of_classifiers)

                total_training_time += training_time

                last_vandalism_set = current_block_re_label[current_block_re_label["LABEL"] == 1]

                if evaluate_by_step:
                    only_re_label_test = pd.DataFrame(columns=standard_testing.columns)
                    # If this DT has any data point, then it means re_label was triggered.
                    if len(re_label_all_blocks) > 0:
                        standard_testing, only_re_label_test = apply_re_label(standard_testing, cluster_class,
                                                                              new_label_value, old_label_value)

                    run_test_by_step(trained_models, ml_model, training_time, standard_testing, only_re_label_test,
                                     current_execution, index_current_data_block, directory_to_save, base_directory)
                    print("Training Time:  " + str(training_time))

        run_test_overall(trained_models, ml_model, total_training_time, standard_testing, only_re_label_test,
                         current_execution, directory_to_save, base_directory)


def train_for_interpretability(dataset_directory, directory_to_save, ml_model, interpretability_path,
                               interpretability_type, number_of_executions, cluster_class,
                               batch_size=512, max_number_of_classifiers=12, post_balance_ratio=0.5,
                               trade_off_performance_stability=0.5, number_of_epochs=128,
                               allowed_changed_in_distribution=0.3, evaluate_by_step=False, new_label_value=0,
                               old_label_value=1, columns_to_drop=["LABEL", "CLUSTER"]):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)
    current_execution = 0
    print("Training for Interpretability started!!!")

    complete_path_to_save = base_directory + directory_to_save

    # The datasets were already separated. The same datasets are used for the different training approaches.
    current_dataset = get_working_dataset("current_train", dataset_directory, current_execution)
    future_dataset = get_working_dataset("future_train_cluster", dataset_directory, current_execution)
    test_dataset = get_working_dataset("test_cluster", dataset_directory, current_execution)

    complete_training_dataset = pd.concat([current_dataset, future_dataset])

    # Separating the training dataset into batches that are going to be presented to the model for training.
    number_of_data_blocks = math.ceil(len(complete_training_dataset) / batch_size)
    mini_batches = np.array_split(complete_training_dataset, number_of_data_blocks)

    trained_models = []

    # It will contain data to evaluate how the data distribution changed in time.
    data_distribution_in_time = []
    time_stamp_last_ensemble_change = 0

    # It will contain data points that suffered re_label. Later it will be used to check no longer vandalism.
    re_label_all_blocks = pd.DataFrame(columns=complete_training_dataset.columns)

    # All data block should contain data points representing the minority class. This is only to ensure that.
    last_vandalism_set = pd.DataFrame(columns=complete_training_dataset.columns)

    total_training_time = 0

    last_complete_data_block = pd.DataFrame(columns=complete_training_dataset.columns)
    last_complete_data_block_standardized = pd.DataFrame(columns=complete_training_dataset.columns)
    last_no_null_testing = pd.DataFrame(columns=test_dataset.columns)
    last_no_null_testing_standardized = pd.DataFrame(columns=test_dataset.columns)
    last_scaled_values = 0

    for index_current_data_block, current_data_block in zip(range(0, number_of_data_blocks), mini_batches):
        if len(current_data_block) > 0:
            # Applying pre-process chunk by chunk. Only considering present data.
            columns_not_altered = [57, 58]  # Label and Cluster columns should not be changed.
            no_null_block, statistical_values = UtilDataSetFiles.handle_missing_values(current_data_block,
                                                                                       columns_not_altered)
            standard_current_block, scaled_values = UtilDataSetFiles.apply_standardization(no_null_block.copy(),
                                                                                           columns_not_altered)

            no_null_testing = UtilDataSetFiles.update_missing_values_media_mode(test_dataset, statistical_values)
            standard_testing = UtilDataSetFiles.apply_standardization_defined(no_null_testing.copy(), scaled_values,
                                                                              [57, 58])

            # Simulating the re-label feedback based on the clusters found by K-Means
            current_block_re_label, only_re_label_current_block = apply_re_label(standard_current_block,
                                                                                 cluster_class, new_label_value,
                                                                                 old_label_value,
                                                                                 replicate_re_label=False)

            # Adding the current re_label data to the DT with all no longer vandalism data points.
            re_label_all_blocks = pd.concat([re_label_all_blocks, only_re_label_current_block])

            current_number_of_minority_instances = len(current_block_re_label[current_block_re_label["LABEL"] == 1])
            current_number_of_majority_instances = len(current_block_re_label[current_block_re_label["LABEL"] == 0])

            # If it's no concept drift, then we want to stop before the re_labelling starts.
            if interpretability_type == "no_concept_drift" and len(only_re_label_current_block) > 0:
                break

            last_complete_data_block, _ = apply_re_label(no_null_block, cluster_class, new_label_value,
                                                         old_label_value, replicate_re_label=False)
            last_complete_data_block_standardized = current_block_re_label

            if interpretability_type == "no_concept_drift":
                last_no_null_testing = no_null_testing.copy()
                last_no_null_testing_standardized = standard_testing.copy()
            else:
                last_no_null_testing, _ = apply_re_label(no_null_testing.copy(), cluster_class, new_label_value,
                                                         old_label_value, replicate_re_label=False)
                last_no_null_testing_standardized, _ = apply_re_label(standard_testing.copy(), cluster_class,
                                                                      new_label_value, old_label_value,
                                                                      replicate_re_label=False)

            last_scaled_values = scaled_values

            '''number_vand = current_number_of_minority_instances
            number_reg = current_number_of_majority_instances
            number_re_label = len(only_re_label_current_block)
            testA = "fhdu"'''

            # This is done just to ensure that all data blocks contain minority instances.
            if current_number_of_minority_instances <= 0:
                current_number_of_minority_instances = len(last_vandalism_set)
                current_block_re_label = pd.concat([current_block_re_label, last_vandalism_set])

            # Getting the best number of classifiers that should be in the ensemble.
            current_best_number_of_classifiers = math.ceil(current_number_of_majority_instances /
                                                           current_number_of_minority_instances)

            if current_best_number_of_classifiers > max_number_of_classifiers:
                # In this case, the oversampling technique is solving the imbalance.
                # current_best_number_of_classifiers = max_number_of_classifiers

                current_block_re_label, current_best_number_of_classifiers = duplicate_minority_instances(
                    current_block_re_label, max_number_of_classifiers, current_number_of_majority_instances)

                current_number_of_minority_instances = len(
                    current_block_re_label[current_block_re_label["LABEL"] == 1])

            current_data_distribution = current_number_of_minority_instances / current_number_of_majority_instances
            data_distribution_in_time.append(current_data_distribution)

            number_of_new_classifiers = 0
            '''data_distribution_to_compare = data_distribution_in_time[time_stamp_last_ensemble_change]
            if (data_distribution_to_compare - current_data_distribution > 0) and \
                    (current_data_distribution / data_distribution_to_compare < 1 - allowed_changed_in_distribution):
                number_of_new_classifiers = current_best_number_of_classifiers - len(trained_models)
                time_stamp_last_ensemble_change = index_current_data_block
                # time_stamp_last_ensemble_change = 2
            '''

            current_ensemble_size = len(trained_models)
            if current_best_number_of_classifiers > current_ensemble_size:
                number_of_new_classifiers = current_best_number_of_classifiers - current_ensemble_size

            number_of_re_label = len(only_re_label_current_block)
            if number_of_re_label > 0:
                amount_to_replicate = round(current_number_of_minority_instances / number_of_re_label)
                if amount_to_replicate > 0 and number_of_re_label < current_number_of_minority_instances:
                    re_label_replicated = pd.concat([only_re_label_current_block] * amount_to_replicate,
                                                    ignore_index=True)
                    number_replicated = len(re_label_replicated)

                    reg_editions = current_block_re_label[current_block_re_label["LABEL"] == 0]
                    reg_editions = reg_editions[reg_editions["CLUSTER"] != 2]
                    reg_editions = reg_editions[number_replicated:]

                    vand_editions = current_block_re_label[current_block_re_label["LABEL"] == 1]

                    current_block_re_label = pd.concat([reg_editions, vand_editions, re_label_replicated]) \
                        .sample(frac=1)

            current_block_re_label = current_block_re_label.drop(["CLUSTER"], axis=1)

            trained_models, training_time = start_modified_due(trained_models, ml_model, current_block_re_label,
                                                               number_of_new_classifiers, post_balance_ratio,
                                                               number_of_epochs, current_best_number_of_classifiers)

            total_training_time += training_time

            last_vandalism_set = current_block_re_label[current_block_re_label["LABEL"] == 1]

            if evaluate_by_step:
                only_re_label_test = pd.DataFrame(columns=standard_testing.columns)
                # If this DT has any data point, then it means re_label was triggered.
                if len(re_label_all_blocks) > 0:
                    standard_testing, only_re_label_test = apply_re_label(standard_testing, cluster_class,
                                                                          new_label_value, old_label_value)

                run_test_by_step(trained_models, ml_model, training_time, standard_testing, only_re_label_test,
                                 current_execution, index_current_data_block, directory_to_save, base_directory)

    testeA = last_no_null_testing[last_no_null_testing["LABEL"] == 1]
    directory_interpretability = base_directory + interpretability_path + interpretability_type
    save_interpretability_data(last_complete_data_block, last_complete_data_block_standardized, last_no_null_testing,
                               last_no_null_testing_standardized, last_scaled_values, directory_interpretability,
                               trained_models)

    '''run_test_overall(trained_models, ml_model, total_training_time, standard_testing, only_re_label_test,
                     current_execution, directory_to_save, base_directory)'''


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


def start_modified_due(trained_models, ml_model, current_data_block, number_of_new_classifiers, post_balance_ratio,
                       number_of_epochs, current_best_number_of_classifiers):

    # Building the balanced datasets from the current data block.
    balanced_datasets = create_balanced_datasets(current_best_number_of_classifiers, current_data_block)

    training_time = 0
    current_ensemble_size = len(trained_models)
    if current_ensemble_size <= 0:
        # Initial Phase. The ensemble of classifiers starts to be built.
        trained_models, training_time = train_with_keras(balanced_datasets, number_of_epochs)

    elif number_of_new_classifiers > 0:
        # Training by adding new classifiers.

        # Get dataset wrongly classified. This is for future work.
        '''wrongly_predicted_editions = get_wrongly_predicted_editions(ensemble, current_data_block, ["LABEL",
                                                                                                   "WEIGHT"])
         balanced_datasets = add_wrongly_predicted_editions(wrongly_predicted_editions, balanced_datasets)'''

        # Train the new classifiers on the firsts balanced dataset, and the rest of the datasets I'm going to use
        # to update the older classifiers.
        data_to_train_new_classifiers = balanced_datasets[0:number_of_new_classifiers]
        data_to_update_previous_classifiers = balanced_datasets[number_of_new_classifiers:
                                                                current_best_number_of_classifiers]

        # This is for future work. We shall further investigate changing the weights to update old classifiers.
        '''balanced_datasets_with_weight = compute_weight_balanced_datasets_per_classifier(ensemble,
                                                                            data_to_update_previous_classifiers)'''
        balanced_datasets_with_weight = data_to_update_previous_classifiers

        new_trained_classifiers, training_time = train_with_keras(data_to_train_new_classifiers, number_of_epochs)
        updated_classifiers, update_time = update_previous_classifiers(balanced_datasets_with_weight, trained_models,
                                                                       number_of_epochs)

        trained_models = new_trained_classifiers + updated_classifiers

        # In this specific situation, the complete training time must consider both training the classifiers and
        # updating the old ones.
        training_time += update_time

    else:
        # Training only by updating previous classifiers.

        # balanced_datasets_with_weight = compute_weight_balanced_datasets_per_classifier(ensemble, balanced_datasets)
        balanced_datasets_with_weight = balanced_datasets

        # After getting the weights for the instances, it's time to update the previous models.
        trained_models, training_time = update_previous_classifiers(balanced_datasets_with_weight, trained_models,
                                                                    number_of_epochs)

    return trained_models, training_time


def duplicate_minority_instances(current_block_re_label, max_number_of_classifiers,
                                 current_number_of_majority_instances):
    '''
    minority_duplicated = current_block_re_label[current_block_re_label["LABEL"] == 1]
    current_block_re_label = pd.concat([current_block_re_label, minority_duplicated]).sample(frac=1)

    current_number_of_minority_instances = len(current_block_re_label[current_block_re_label["LABEL"] == 1])
    current_best_number_of_classifiers = math.ceil(current_number_of_majority_instances /
                                                   current_number_of_minority_instances)

    if current_best_number_of_classifiers > max_number_of_classifiers:
        return duplicate_minority_instances(current_block_re_label, max_number_of_classifiers,
                                            current_number_of_majority_instances)

    return current_block_re_label, current_best_number_of_classifiers
    '''
    # new code is not duplicating, but instead it's oversampling.
    dataset_minority_editions = current_block_re_label[current_block_re_label["LABEL"] == 1]
    number_minority_editions = len(dataset_minority_editions)
    desired_number_of_minority = np.math.ceil(current_number_of_majority_instances / max_number_of_classifiers)
    number_of_minority_to_oversample = desired_number_of_minority - number_minority_editions

    if number_of_minority_to_oversample > number_minority_editions:
        # Then just replicate and remove some regular editions (under sampling).
        editions_oversampled = dataset_minority_editions.sample(n=number_minority_editions)
        number_regular_to_remove = (number_of_minority_to_oversample - number_minority_editions) * \
                                   max_number_of_classifiers

        reg_editions = current_block_re_label[current_block_re_label["LABEL"] == 0]
        re_label_editions = current_block_re_label[current_block_re_label["CLUSTER"] == 2]
        vand_editions = current_block_re_label[current_block_re_label["LABEL"] == 1]
        reg_editions = reg_editions[reg_editions["CLUSTER"] != 2]
        reg_editions = reg_editions[number_regular_to_remove:]

        current_block_re_label = pd.concat([reg_editions, re_label_editions, vand_editions])
    else:
        editions_oversampled = dataset_minority_editions.sample(n=number_of_minority_to_oversample)

    current_block_re_label = pd.concat([current_block_re_label, editions_oversampled]).sample(frac=1)

    return current_block_re_label, max_number_of_classifiers


def create_balanced_datasets(number_of_classifiers_to_train, current_data_block):
    current_data_block["WEIGHT"] = 1
    minority_instances = current_data_block[current_data_block["LABEL"] == 1]
    majority_instances = current_data_block[current_data_block["LABEL"] == 0]

    number_of_minority_instances = len(minority_instances)
    number_of_majority_instances = len(majority_instances)

    balanced_datasets = []

    # if current_imbalance_ratio <= ratio_number_of_classifiers:
    if number_of_majority_instances / number_of_classifiers_to_train > number_of_minority_instances:
        # Use Oversampling to generate minority data.
        oversampling_generated_data = generate_data_oversampling(current_data_block, number_of_minority_instances,
                                                                 number_of_majority_instances,
                                                                 number_of_classifiers_to_train)

        # Now after Oversampling, we have more instances of vandalism editions.
        minority_instances_with_oversampling = oversampling_generated_data[oversampling_generated_data["LABEL"] == 1]
        number_of_minority_instances = len(minority_instances_with_oversampling)

        number_of_balanced_dataset = math.ceil(number_of_majority_instances / number_of_minority_instances)
        balanced_datasets = np.array_split(majority_instances, number_of_balanced_dataset)

        # number_of_balanced_dataset should be equal to number_of_classifiers_to_train
        for index_balanced_dataset in range(0, number_of_balanced_dataset):
            aux_balanced_dataset = [balanced_datasets[index_balanced_dataset], minority_instances_with_oversampling]
            balanced_datasets[index_balanced_dataset] = pd.concat(aux_balanced_dataset).sample(frac=1)
    else:
        # No need to use Oversampling, flow to balance the dataset.
        # number_of_minority_instances_with_ratio = number_of_majority_instances * \
        #                                          post_balance_ratio / number_of_classifiers_to_train
        # In the algorithm this value is calculated as above, I changed to use the same size for both
        # minority and majority.
        number_of_minority_instances_with_ratio = number_of_majority_instances / number_of_classifiers_to_train
        number_of_minority_instances_with_ratio = math.ceil(number_of_minority_instances_with_ratio)

        # number_of_majority_instances_with_classifier = number_of_majority_instances / number_of_classifiers_to_train
        non_overlapping_majority_blocks = np.array_split(majority_instances,
                                                         number_of_classifiers_to_train)

        for index_candidate_classifier in range(0, number_of_classifiers_to_train):
            temp_minority_instances = minority_instances.sample(n=number_of_minority_instances_with_ratio)
            temp_majority_instances = non_overlapping_majority_blocks[index_candidate_classifier]

            # In our current application, all weights are going to be 1. But we leave this for future work
            # (in case of the need in changing the weights)
            aux_balanced_dataset = [temp_minority_instances, temp_majority_instances]
            balanced_datasets.append(pd.concat(aux_balanced_dataset).sample(frac=1))

    return balanced_datasets


def generate_data_oversampling(current_data_block, number_of_minority_instances, number_of_majority_instances,
                               number_of_classifiers_to_train):

    number_instances_to_generate = (number_of_majority_instances / number_of_classifiers_to_train) - \
                                   number_of_minority_instances
    number_instances_to_generate = math.ceil(number_instances_to_generate)

    if number_instances_to_generate > len(current_data_block[current_data_block["LABEL"] == 1]):
        number_to_sample = len(current_data_block[current_data_block["LABEL"] == 1])
        left_number_to_sample = number_instances_to_generate - number_to_sample
        minority_data = current_data_block[current_data_block["LABEL"] == 1].sample(n=number_to_sample)

        for _ in range(0, left_number_to_sample):
            minority_data = pd.concat([minority_data, current_data_block[current_data_block["LABEL"] == 1].sample(n=1)])
    else:
        minority_data = current_data_block[current_data_block["LABEL"] == 1].sample(n=number_instances_to_generate)

    current_data_block = pd.concat([current_data_block, minority_data]).sample(frac=1)

    return current_data_block


def train_with_keras(balanced_datasets, number_of_epochs):
    number_of_balanced_datasets = len(balanced_datasets)
    trained_classifiers = []

    accumulated_time = 0
    for index_balanced in range(0, number_of_balanced_datasets):
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
        accumulated_time += finish_time - start_time

        # Do not compute the time for these operations, it's not what we are measuring.
        x_train = balanced_datasets[index_balanced].drop(["LABEL", "WEIGHT"], axis=1)
        y_train = balanced_datasets[index_balanced]["LABEL"]
        weights = balanced_datasets[index_balanced]["WEIGHT"]
        number_of_data_points = len(x_train)

        start_time = timeit.default_timer()
        model.fit(x_train, y_train, sample_weight=weights, epochs=number_of_epochs, batch_size=number_of_data_points,
                  verbose=0)
        finish_time = timeit.default_timer()
        accumulated_time += finish_time - start_time

        trained_classifiers.append(model)

    return trained_classifiers, accumulated_time


def update_previous_classifiers(balanced_datasets_with_weight, previous_classifiers, number_of_epochs):
    accumulated_time = 0
    index = 0
    for balanced_dataset in balanced_datasets_with_weight:
        x_train = balanced_dataset.drop(["LABEL", "WEIGHT"], axis=1)
        y_train = balanced_dataset["LABEL"]
        x_weights = balanced_dataset["WEIGHT"]
        number_of_data_points = len(balanced_dataset)

        start_time = timeit.default_timer()
        previous_classifiers[index].fit(x_train, y_train, epochs=number_of_epochs, batch_size=number_of_data_points,
                                        sample_weight=x_weights, verbose=0)
        finish_time = timeit.default_timer()
        accumulated_time += finish_time - start_time

        index += 1

    return previous_classifiers, accumulated_time


def run_test_overall(trained_models, ml_model, train_execution_time, testing_dataset, only_re_label_testing,
                     current_execution, directory_to_save, base_directory):

    information_to_save = {
        "current_execution": current_execution,
        "test_results": {},
        "test_re_label_results": {},
        "total_train_execution_time": train_execution_time,
    }

    information_to_save["test_results"] = evaluate_ensemble_models_keras(trained_models, testing_dataset,
                                                                         ["LABEL", "CLUSTER"])

    if len(only_re_label_testing) > 0:
        only_re_label_test_dataset = only_re_label_testing.drop(["CLUSTER"], axis=1)
        information_to_save["test_re_label_results"] = evaluate_re_label_keras(trained_models,
                                                                               only_re_label_test_dataset)

    current_execution_directory = directory_to_save + str(current_execution)
    save_keras_model(trained_models, "model", current_execution_directory)

    information_file_path = base_directory + current_execution_directory + "/overall_test_results" + ".pickle"
    pickle.dump(information_to_save, open(information_file_path, 'wb'))


def run_test_by_step(trained_models, ml_model, train_execution_time, current_testing, only_re_label_testing,
                     current_execution, index_current_data_block, directory_to_save, base_directory):

    information_to_save = {
        "current_execution": current_execution,
        "test_results": {},
        "test_re_label_results": {},
        "train_execution_time": train_execution_time,
        "current_step": index_current_data_block
    }

    information_to_save["test_results"] = evaluate_ensemble_models_keras(trained_models, current_testing,
                                                                         ["LABEL", "CLUSTER"])

    if len(only_re_label_testing) > 0:
        information_to_save["test_re_label_results"] = evaluate_re_label_keras(trained_models, only_re_label_testing,
                                                                               ["LABEL", "CLUSTER"])

    current_execution_directory = directory_to_save + str(current_execution)
    information_file_name = "/" + str(index_current_data_block) + "_step_test_results.pickle"
    information_file_path = base_directory + current_execution_directory + information_file_name
    pickle.dump(information_to_save, open(information_file_path, 'wb'))


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

    # return overall_recall, calculated_confusion_matrix
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

    # return calculated_confusion_matrix
    return test_results


def get_keras_predictions(trained_models, x_test):
    models_predictions = []

    for model in trained_models:
        predictions = model.predict(x_test)
        #classes_highest_probability = np.argmax(predictions, axis=1)  # This gets the class with highest probability
        models_predictions.append(predictions)

    classes_summed = np.sum(models_predictions, axis=0)
    ensemble_prediction = np.argmax(classes_summed, axis=1)
    classes_probabilities = np.mean(models_predictions, axis=0)

    return ensemble_prediction, classes_probabilities


def get_working_dataset(file_name, directory, index):
    complete_file_name = str(index) + "_" + file_name + ".csv"
    dataset = UtilDataSetFiles.get_data_set_from_file(directory, complete_file_name)

    return dataset


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


def save_keras_model(trained_models, folder_name, path_to_save):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    complete_path_to_save = base_directory + path_to_save
    if not os.path.exists(complete_path_to_save):
        os.mkdir(complete_path_to_save)

    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = complete_path_to_save + "/" + folder_name + str(model_index)
        model.save(file_path)





