import math
import pathlib
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers

from norm_detection_online.util import UtilDataSetFiles


def start_training_k_validation(batch_size=1000, initial_ensemble_size=12, max_number_of_ensembles=40,
                                number_k_of_folders=10, post_balance_ratio=0.5, trade_off_performance_stability=0.5,
                                number_of_epochs=100, allowed_change_in_distribution=0.30, directory_to_save=""):
    # First I'm going train in the first part of the training set. The changes will be in the second part, with the
    # file future_training
    old_training = UtilDataSetFiles.get_data_set_from_file("/experiments/keras/",
                                                           "current_training.csv")

    # Now the rest of the processing. From now on we code for the re_label part.
    future_training, prediction_labels = get_dataset_with_kmeans_label()
    prediction_label_used = prediction_labels[0]

    # TEMPORARILY SHUFFLING
    future_training = future_training.sample(frac=1)

    # Separating the future dataset into k folders.
    folders = np.array_split(future_training, number_k_of_folders)

    dataset_columns = future_training.columns
    training_folders, testing_folders = UtilDataSetFiles.split_training_test_k_validation(folders, dataset_columns,
                                                                                          "",
                                                                                          "current_training.csv",
                                                                                          "future_training.csv",
                                                                                          False,
                                                                                          number_k_of_folders)

    # current_testing is going to be used for testing, and the other folders are training.
    for current_training, current_testing in zip(training_folders, testing_folders):
        process_directory_path = str(pathlib.Path(__file__).parent.parent.parent) + directory_to_save

        old_training["PREDICTION"] = -1
        complete_training_dataset = pd.concat([old_training, current_training])
        trained_ensemble, no_longer_vandalism_standard, statistical_values = online_algorithm(complete_training_dataset,
                                                                                              prediction_label_used)

        current_testing = current_testing.replace(False, 0)
        current_testing = current_testing.replace("False", 0)
        current_testing = current_testing.replace(True, 1)
        current_testing = current_testing.replace("True", 1)
        no_null_testing = UtilDataSetFiles.update_missing_values_media_mode_dict(current_testing, statistical_values)

        all_test_edition_standard = []
        for edition_index, edition in no_null_testing.iterrows():
            temp_edition_standard = pd.DataFrame([UtilDataSetFiles.apply_standardization(edition, statistical_values)])
            all_test_edition_standard.append(temp_edition_standard)

        testing_standard = pd.DataFrame(columns=no_null_testing.columns, data=all_test_edition_standard)

        dt_no_longer_vandalism_test = testing_standard.loc[testing_standard["PREDICTION"] == prediction_label_used]

        evaluate_keras_with_testing_data(trained_ensemble, testing_standard, ["LABEL", "PREDICTION"])
        no_longer_vandalism_standard["LABEL"] = 0
        evaluate_ensemble_no_longer_vandalism(trained_ensemble, no_longer_vandalism_standard, ["LABEL"])
        evaluate_ensemble_no_longer_vandalism(trained_ensemble, dt_no_longer_vandalism_test, ["LABEL", "PREDICTION"])


def start_training_k_validation_per_step(batch_size=1000, initial_ensemble_size=12, max_number_of_ensembles=40,
                                         number_k_of_folders=10, post_balance_ratio=0.5,
                                         trade_off_performance_stability=0.5, number_of_epochs=100,
                                         allowed_change_in_distribution=0.30, directory_to_save="",
                                         threshold_evaluate_by_step=512, load_from_disk=False):
    if load_from_disk:
        for k_validation_index in range(0, number_k_of_folders):
            complete_training_dataset, balanced_dataset_per_classifier, no_longer_vandalism_standard, \
            statistical_values, dataset_ranges_to_evaluate_by_step, no_longer_vandalism_blocks, \
            running_statistical_values_by_step, current_testing, prediction_labels = \
                get_dataset_online_information(k_validation_index)

            prediction_label_used = prediction_labels[0]

            trained_ensemble = []
            for _ in range(0, initial_ensemble_size):
                classifier = define_classifier()
                trained_ensemble.append(classifier)

            for current_step in range(0, math.ceil(len(complete_training_dataset)/threshold_evaluate_by_step)):
                aux_balanced_dataset_per_classifier = []
                for classifier_index in range(0, len(trained_ensemble)):
                    initial_index = dataset_ranges_to_evaluate_by_step[classifier_index]["initial_indexes"][current_step]
                    final_index = dataset_ranges_to_evaluate_by_step[classifier_index]["final_indexes"][current_step]

                    aux_balanced_dataset_per_classifier.append({
                        "x_train": balanced_dataset_per_classifier[classifier_index]["x_train"][initial_index:final_index],
                        "y_train": balanced_dataset_per_classifier[classifier_index]["y_train"][initial_index:final_index]
                    })

                trained_ensemble = train_ensemble_keras(trained_ensemble, aux_balanced_dataset_per_classifier)

                # We are evaluating just after the community starts to change the behavior (concept drift).
                if current_step >= 31:
                    no_longer_vandalism_blocks_dt = pd.DataFrame(data=no_longer_vandalism_blocks[current_step])
                    evaluate_ensemble_by_step(trained_ensemble, current_testing.copy(),
                                              no_longer_vandalism_blocks_dt.copy(),
                                              statistical_values, prediction_label_used, k_validation_index, current_step)

            current_testing = current_testing.replace(False, 0)
            current_testing = current_testing.replace("False", 0)
            current_testing = current_testing.replace(True, 1)
            current_testing = current_testing.replace("True", 1)
            no_null_testing = UtilDataSetFiles.update_missing_values_media_mode_dict(current_testing, statistical_values)

            all_test_edition_standard = []
            for edition_index, edition in no_null_testing.iterrows():
                temp_edition_standard = pd.DataFrame([UtilDataSetFiles.apply_standardization(edition, statistical_values)])
                all_test_edition_standard.append(temp_edition_standard)

            testing_standard = pd.concat(all_test_edition_standard)
            #testing_standard = pd.DataFrame(columns=no_null_testing.columns, data=all_test_edition_standard)

            dt_no_longer_vandalism_test = testing_standard.loc[testing_standard["PREDICTION"] == prediction_label_used]

            evaluate_keras_with_testing_data(trained_ensemble, testing_standard, ["LABEL", "PREDICTION"])
            no_longer_vandalism_standard["LABEL"] = 0
            evaluate_ensemble_no_longer_vandalism(trained_ensemble, no_longer_vandalism_standard, ["LABEL"])
            evaluate_ensemble_no_longer_vandalism(trained_ensemble, dt_no_longer_vandalism_test, ["LABEL", "PREDICTION"])

    else:

        # First I'm going train in the first part of the training set. The changes will be in the second part, with the
        # file future_training
        old_training = UtilDataSetFiles.get_data_set_from_file("/experiments/keras/",
                                                               "current_training.csv")

        # Now the rest of the processing. From now on we code for the re_label part.
        future_training, prediction_labels = get_dataset_with_kmeans_label()
        prediction_label_used = prediction_labels[0]
        future_training.loc[future_training["PREDICTION"] == prediction_label_used, "LABEL"] = 0

        # Separating the future dataset into k folders.
        folders = np.array_split(future_training, number_k_of_folders)

        dataset_columns = future_training.columns
        training_folders, testing_folders = UtilDataSetFiles.split_training_test_k_validation(folders, dataset_columns,
                                                                                              "",
                                                                                              "current_training.csv",
                                                                                              "future_training.csv",
                                                                                              False,
                                                                                              number_k_of_folders)

        k_validation_index = 0
        # current_testing is going to be used for testing, and the other folders are training.
        for current_training, current_testing in zip(training_folders, testing_folders):
            process_directory_path = str(pathlib.Path(__file__).parent.parent.parent) + directory_to_save

            old_training["PREDICTION"] = -1
            complete_training_dataset = pd.concat([old_training, current_training])
            trained_ensemble, balanced_dataset_per_classifier, no_longer_vandalism_standard, statistical_values, \
               dataset_ranges_to_evaluate_by_step, no_longer_vandalism_blocks, running_statistical_values_by_step = \
                generate_train_dataset(complete_training_dataset, prediction_label_used)

            save_dataset_online_information(complete_training_dataset, balanced_dataset_per_classifier,
                                            no_longer_vandalism_standard,
                                            statistical_values, dataset_ranges_to_evaluate_by_step,
                                            no_longer_vandalism_blocks, running_statistical_values_by_step,
                                            current_testing, prediction_labels, k_validation_index)

            for current_step in range(0, math.ceil(len(complete_training_dataset)/threshold_evaluate_by_step)):
                aux_balanced_dataset_per_classifier = []
                for classifier_index in range(0, len(trained_ensemble)):
                    initial_index = dataset_ranges_to_evaluate_by_step[classifier_index]["initial_indexes"][current_step]
                    final_index = dataset_ranges_to_evaluate_by_step[classifier_index]["final_indexes"][current_step]

                    aux_balanced_dataset_per_classifier.append({
                        "x_train": balanced_dataset_per_classifier[classifier_index]["x_train"][initial_index:final_index],
                        "y_train": balanced_dataset_per_classifier[classifier_index]["y_train"][initial_index:final_index]
                    })

                trained_ensemble = train_ensemble_keras(trained_ensemble, aux_balanced_dataset_per_classifier)

                # We are evaluating just after the community starts to change the behavior (concept drift).
                if current_step >= 31:
                    no_longer_vandalism_blocks_dt = pd.DataFrame(data=no_longer_vandalism_blocks[current_step])
                    evaluate_ensemble_by_step(trained_ensemble, current_testing.copy(),
                                              no_longer_vandalism_blocks_dt.copy(),
                                              statistical_values, prediction_label_used, k_validation_index,
                                              current_step)

            current_testing = current_testing.replace(False, 0)
            current_testing = current_testing.replace("False", 0)
            current_testing = current_testing.replace(True, 1)
            current_testing = current_testing.replace("True", 1)
            no_null_testing = UtilDataSetFiles.update_missing_values_media_mode_dict(current_testing, statistical_values)

            all_test_edition_standard = []
            for edition_index, edition in no_null_testing.iterrows():
                temp_edition_standard = pd.DataFrame([UtilDataSetFiles.apply_standardization(edition, statistical_values)])
                all_test_edition_standard.append(temp_edition_standard)

            testing_standard = pd.concat(all_test_edition_standard)
            # testing_standard = pd.DataFrame(columns=no_null_testing.columns, data=all_test_edition_standard)

            dt_no_longer_vandalism_test = testing_standard.loc[testing_standard["PREDICTION"] == prediction_label_used]

            evaluate_keras_with_testing_data(trained_ensemble, testing_standard, ["LABEL", "PREDICTION"])
            no_longer_vandalism_standard["LABEL"] = 0
            evaluate_ensemble_no_longer_vandalism(trained_ensemble, no_longer_vandalism_standard, ["LABEL"])
            evaluate_ensemble_no_longer_vandalism(trained_ensemble, dt_no_longer_vandalism_test, ["LABEL", "PREDICTION"])

            k_validation_index += 1


def save_dataset_online_information(complete_training_dataset, balanced_dataset_per_classifier,
                                    no_longer_vandalism_standard, statistical_values,
                                    dataset_ranges_to_evaluate_by_step, no_longer_vandalism_blocks,
                                    running_statistical_values_by_step, current_testing, prediction_labels,
                                    k_validation_index):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/DatasetInformation/" + \
                str(k_validation_index) + "/"

    complete_training_dataset_file_name = "complete_dataset.pkl"
    complete_training_dataset_path = file_path + complete_training_dataset_file_name
    UtilDataSetFiles.save_pickle_objects(complete_training_dataset, complete_training_dataset_path)

    balanced_dataset_index = 0
    '''for balanced_dataset in balanced_dataset_per_classifier:
        balanced_file_name = "balanced_dataset" + str(balanced_dataset_index) + ".csv"
        # balanced_file_path = file_path + balanced_file_name
        UtilDataSetFiles.save_data_set(balanced_dataset, file_path, balanced_file_name)

        balanced_dataset_index += 1'''
    balanced_dataset_per_classifier_file_name = "balanced_dataset_per_classifier.pkl"
    complete_balanced_dataset_path = file_path + balanced_dataset_per_classifier_file_name
    UtilDataSetFiles.save_pickle_objects(balanced_dataset_per_classifier, complete_balanced_dataset_path)

    '''no_longer_vandalism_standard_file_name = "no_longer_vandalism_standard.csv"
    UtilDataSetFiles.save_data_set(no_longer_vandalism_standard, file_path, no_longer_vandalism_standard_file_name)'''
    no_longer_vandalism_standard_file_name = "no_longer_vandalism_standard.pkl"
    complete_no_longer_vandalism_standard_path = file_path + no_longer_vandalism_standard_file_name
    UtilDataSetFiles.save_pickle_objects(no_longer_vandalism_standard, complete_no_longer_vandalism_standard_path)

    statistical_values_file_name = "statistical_values.pkl"
    complete_statistical_values_path = file_path + statistical_values_file_name
    UtilDataSetFiles.save_pickle_objects(statistical_values, complete_statistical_values_path)

    range_to_evaluate_file_name = "dataset_range_to_evaluate_by_step.pkl"
    complete_range_to_evaluate_path = file_path + range_to_evaluate_file_name
    UtilDataSetFiles.save_pickle_objects(dataset_ranges_to_evaluate_by_step, complete_range_to_evaluate_path)

    no_longer_vandalism_block_file_name = "no_longer_vandalism_blocks.pkl"
    complete_no_longer_vandalism_blocks_path = file_path + no_longer_vandalism_block_file_name
    UtilDataSetFiles.save_pickle_objects(no_longer_vandalism_blocks, complete_no_longer_vandalism_blocks_path)

    running_statistical_values_by_step_file_name = "running_statistical_values_by_step.pkl"
    complete_running_statistical_values_by_step_path = file_path + running_statistical_values_by_step_file_name
    UtilDataSetFiles.save_pickle_objects(running_statistical_values_by_step,
                                         complete_running_statistical_values_by_step_path)

    current_testing_file_name = "current_testing.pkl"
    complete_current_testing_path = file_path + current_testing_file_name
    UtilDataSetFiles.save_pickle_objects(current_testing, complete_current_testing_path)

    prediction_labels_file_name = "prediction_labels.pkl"
    prediction_labels_path = file_path + prediction_labels_file_name
    UtilDataSetFiles.save_pickle_objects(prediction_labels, prediction_labels_path)


def get_dataset_online_information(k_validation_index):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/DatasetInformation/" + \
                str(k_validation_index) + "/"

    complete_training_dataset_file_name = "complete_dataset.pkl"
    complete_training_dataset_path = file_path + complete_training_dataset_file_name
    complete_training_dataset = UtilDataSetFiles.load_pickle_objects(complete_training_dataset_path)

    '''balanced_dataset_per_classifier = UtilDataSetFiles.get_data_set_balanced(file_path, "balanced_dataset", 12)

    no_longer_vandalism_standard = UtilDataSetFiles.get_data_set_from_file(file_path,
                                                                           "no_longer_vandalism_standard.csv")'''
    balanced_dataset_per_classifier_file_name = "balanced_dataset_per_classifier.pkl"
    complete_balanced_per_classifier_path = file_path + balanced_dataset_per_classifier_file_name
    balanced_dataset_per_classifier = UtilDataSetFiles.load_pickle_objects(complete_balanced_per_classifier_path)

    no_longer_vandalism_standard_file_name = "no_longer_vandalism_standard.pkl"
    complete_no_longer_vandalism_path = file_path + no_longer_vandalism_standard_file_name
    no_longer_vandalism_standard = UtilDataSetFiles.load_pickle_objects(complete_no_longer_vandalism_path)

    statistical_values_file_name = "statistical_values.pkl"
    complete_statistical_values_path = file_path + statistical_values_file_name
    statistical_values = UtilDataSetFiles.load_pickle_objects(complete_statistical_values_path)

    range_to_evaluate_file_name = "dataset_range_to_evaluate_by_step.pkl"
    complete_range_to_evaluate_path = file_path + range_to_evaluate_file_name
    dataset_ranges_to_evaluate_by_step = UtilDataSetFiles.load_pickle_objects(complete_range_to_evaluate_path)

    no_longer_vandalism_block_file_name = "no_longer_vandalism_blocks.pkl"
    complete_no_longer_vandalism_blocks_path = file_path + no_longer_vandalism_block_file_name
    no_longer_vandalism_blocks = UtilDataSetFiles.load_pickle_objects(complete_no_longer_vandalism_blocks_path)

    running_statistical_values_by_step_file_name = "running_statistical_values_by_step.pkl"
    complete_running_statistical_values_by_step_path = file_path + running_statistical_values_by_step_file_name
    running_statistical_values_by_step = UtilDataSetFiles.\
        load_pickle_objects(complete_running_statistical_values_by_step_path)

    current_testing_file_name = "current_testing.pkl"
    complete_current_testing_path = file_path + current_testing_file_name
    current_testing = UtilDataSetFiles.load_pickle_objects(complete_current_testing_path)

    prediction_labels_file_name = "prediction_labels.pkl"
    prediction_labels_path = file_path + prediction_labels_file_name
    prediction_labels = UtilDataSetFiles.load_pickle_objects(prediction_labels_path)

    return complete_training_dataset, balanced_dataset_per_classifier, no_longer_vandalism_standard, statistical_values,\
           dataset_ranges_to_evaluate_by_step, no_longer_vandalism_blocks, running_statistical_values_by_step, \
           current_testing, prediction_labels


def get_dataset_with_kmeans_label():
    current_training = UtilDataSetFiles.get_data_set_from_file("/experiments/keras/",
                                                               "current_training.csv")

    future_training = UtilDataSetFiles.get_data_set_from_file("/experiments/keras/",
                                                              "future_training.csv")

    no_null_block, statistical_values = UtilDataSetFiles.handle_missing_values(current_training)
    standard_current_block, scaled_values = UtilDataSetFiles.apply_standardization_in_dataset(no_null_block.copy())

    no_null_testing = UtilDataSetFiles.update_missing_values_media_mode(future_training, statistical_values)
    future_training_standard = UtilDataSetFiles.apply_standardization_defined(no_null_testing.copy(), scaled_values)

    train_kmeans_data = future_training_standard[future_training_standard["LABEL"] == 1].drop(["LABEL"], axis=1)
    model = KMeans(n_clusters=4, init="k-means++", max_iter=1000)
    fitted_model = model.fit(train_kmeans_data)
    recurrent_count = Counter(fitted_model.labels_)
    print(recurrent_count)
    class_to_keep_label = recurrent_count.most_common(1)[0][0]
    class_of_interest = recurrent_count.most_common(2)[1][0]
    third_best = recurrent_count.most_common(3)[2][0]

    predictions_labels = [class_to_keep_label, class_of_interest, third_best]

    # All editions, that are not going to be simulated as re_label, will receive -1 for the prediction column.
    future_training["PREDICTION"] = -1
    for vandalism_index, vandalism_edition in future_training_standard[future_training_standard["LABEL"] == 1].iterrows():
        data_to_predict = vandalism_edition.drop(["LABEL"])
        data_to_predict = np.array(data_to_predict).reshape(1, -1)
        prediction = fitted_model.predict(data_to_predict)

        future_training.loc[vandalism_index, "PREDICTION"] = prediction
        future_training_standard.loc[vandalism_index, "PREDICTION"] = prediction

    return future_training, predictions_labels


def start_training_re_label(initial_ensemble_size=12, max_number_of_ensembles=40, post_balance_ratio=0.5,
                            trade_off_performance_stability=0.5, number_of_epochs=100,
                            allowed_change_in_distribution=0.30,
                            datasets_path="/experiments/keras/re_training/re_label/"):
    training_dataset = UtilDataSetFiles.get_data_set_from_file(datasets_path,
                                                               "re_labeled_standard.csv")
    training_dataset = training_dataset.drop(["WEIGHT", "RELEVANT_FEATURES"], axis=1)
    training_dataset = training_dataset[16220:len(training_dataset) - 1]

    testing_dataset = UtilDataSetFiles.get_data_set_from_file(datasets_path,
                                                              "testing_standard.csv")
    testing_dataset = testing_dataset.drop(["RELEVANT_FEATURES"], axis=1)

    no_longer_vandalism = UtilDataSetFiles.get_data_set_from_file(datasets_path,
                                                                  "re_label_no_longer_vandalism.csv")
    no_longer_vandalism = no_longer_vandalism.drop(["WEIGHT", "RELEVANT_FEATURES"], axis=1)

    no_longer_vandalism_test = UtilDataSetFiles.get_data_set_from_file(datasets_path,
                                                                       "re_label_no_longer_vandalism_test.csv")
    no_longer_vandalism_test = no_longer_vandalism_test.drop(["RELEVANT_FEATURES"], axis=1)

    trained_ensemble, statistical_values = online_algorithm(training_dataset)

    save_keras_model(trained_ensemble, "model", datasets_path + "trained_keras_models/")

    evaluate_keras_with_testing_data(trained_ensemble, testing_dataset)
    evaluate_no_longer_vandalism(trained_ensemble, no_longer_vandalism, no_longer_vandalism_test, statistical_values)


def test_trained_ensemble(ensemble_size=12, datasets_path="/experiments/keras/re_training/re_label/",
                          models_folder="trained_keras_models/"):
    testing_dataset = UtilDataSetFiles.get_data_set_from_file(datasets_path,
                                                              "testing_standard.csv")
    testing_dataset = testing_dataset.drop(["RELEVANT_FEATURES"], axis=1)

    no_longer_vandalism = UtilDataSetFiles.get_data_set_from_file(datasets_path,
                                                                  "re_label_no_longer_vandalism.csv")
    no_longer_vandalism = no_longer_vandalism.drop(["WEIGHT", "RELEVANT_FEATURES"], axis=1)

    no_longer_vandalism_test = UtilDataSetFiles.get_data_set_from_file(datasets_path,
                                                                       "re_label_no_longer_vandalism_test.csv")
    no_longer_vandalism_test = no_longer_vandalism_test.drop(["RELEVANT_FEATURES"], axis=1)

    models_path = datasets_path + models_folder
    trained_ensemble = get_trained_keras_models(ensemble_size, models_path)

    evaluate_keras_with_testing_data(trained_ensemble, testing_dataset)
    # Only temporarily, fix this code to get as well the statistical values.
    statistical_values = []
    evaluate_no_longer_vandalism(trained_ensemble, no_longer_vandalism, no_longer_vandalism_test, statistical_values)


def online_algorithm(training_dataset,
                     prediction_label_used,
                     initial_ensemble_size=12,
                     desired_distribution={0: 0.5, 1: 0.5},
                     sampling_rate=1,
                     threshold_calculate_distribution=500,
                     allowed_change_in_distribution=0.3,
                     max_number_of_classifiers=40):
    running_statistical_values = {}
    features = training_dataset.columns.values
    actual_distribution = {0: 0, 1: 0}
    last_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}

    standard_editions = []
    no_longer_vandalism_standard = []

    # Initialize running statistics
    for individual_feature in features:
        # Initial array for the feature. The first position is the current mean and the second, the current N.
        # The third position is the number of true, the fourth position number of false. These are used for the mode.
        # The fifth is the sum of squares (used to calculate variance). The sixth is the standard deviation.
        running_statistical_values[individual_feature] = [0, 0, 0, 0, 0, 0]

    ensemble = []
    balanced_dataset_per_classifier = []
    # Initialize classifiers with the ensemble of size = initial_ensemble_size
    for _ in range(0, initial_ensemble_size):
        classifier = define_classifier()
        ensemble.append(classifier)
        balanced_dataset_per_classifier.append({
            "x_train": pd.DataFrame(),
            "y_train": pd.Series()
        })

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0
    number_of_data_points = 0
    current_best_number_of_classifiers = len(ensemble)

    training_dataset = training_dataset.replace(False, 0)
    training_dataset = training_dataset.replace("False", 0)
    training_dataset = training_dataset.replace(True, 1)
    training_dataset = training_dataset.replace("True", 1)
    count_editions = 0
    for edition_index, edition in training_dataset.iterrows():
        real_label = edition["LABEL"]
        kmeans_prediction = edition["PREDICTION"]
        edition = edition.drop(["LABEL", "PREDICTION"])

        if real_label == 0:
            number_of_regular_editions += 1
        else:
            number_of_vandalism_editions += 1

        edition_no_null, running_statistical_values = UtilDataSetFiles.fill_null(edition,
                                                                                 running_statistical_values)

        edition_standardized = UtilDataSetFiles.apply_standardization(edition_no_null,
                                                                      running_statistical_values)

        standard_editions.append(edition_standardized)

        actual_distribution[real_label] += 1
        number_of_data_points += 1
        rate = sampling_rate * desired_distribution[real_label] / (
                    actual_distribution[real_label] / number_of_data_points)

        number_of_new_classifiers = 0
        if count_editions > last_data_distribution["count_index_edition"] + threshold_calculate_distribution:
            #current_best_number_of_classifiers = math.ceil(actual_distribution[0] /
            #                                               actual_distribution[1])

            # I update only the value of the edition_index, because I want to visit this part only after the number
            # threshold_calculate_distribution of editions have passed.
            last_data_distribution["edition_index"] = edition_index
            last_data_distribution["count_index_edition"] = count_editions

            calculated_actual_distribution = actual_distribution[1] / actual_distribution[0]

            # If I don't have last_data_distribution, then it means that I'm passing through here for the first time.
            # So I have to give an initial value to this last_data_distribution
            if last_data_distribution["distribution"][1] == 0 or last_data_distribution["distribution"][1] == 0:
                last_data_distribution["distribution"][0] = actual_distribution[0]
                last_data_distribution["distribution"][1] = actual_distribution[1]

            calculated_last_distribution = last_data_distribution["distribution"][1] / \
                                           last_data_distribution["distribution"][0]

            if (calculated_last_distribution - calculated_actual_distribution > 0) and \
                    (calculated_actual_distribution / calculated_last_distribution < 1 - allowed_change_in_distribution):
                #number_of_new_classifiers = current_best_number_of_classifiers - len(ensemble)
                last_data_distribution["edition_index"] = edition_index
                last_data_distribution["count_index_edition"] = count_editions
                last_data_distribution["distribution"] = actual_distribution

                '''for _ in range(0, number_of_new_classifiers):
                    classifier = define_classifier()
                    ensemble.append(classifier)
                    balanced_dataset_per_classifier.append({
                        "x_train": pd.DataFrame(),
                        "y_train": pd.Series()
                    })'''

        for classifier_index in range(0, len(ensemble)):
            random_generator = np.random.RandomState()
            number_of_data_points_to_add = 0
            for _ in range(random_generator.poisson(rate)):
                number_of_data_points_to_add += 1

            standardized_inputs = [edition_standardized] * number_of_data_points_to_add
            labels = [real_label] * number_of_data_points_to_add

            keras_input = pd.DataFrame(standardized_inputs)
            keras_output = pd.Series(data=labels, dtype='int32')

            balanced_dataset_per_classifier[classifier_index]["x_train"] = pd.concat([balanced_dataset_per_classifier[classifier_index]["x_train"],
                                                                                      keras_input])
            balanced_dataset_per_classifier[classifier_index]["y_train"] = pd.concat([balanced_dataset_per_classifier[classifier_index]["y_train"],
                                                                                      keras_output])

        # This would be the data that was re_labeled by the community.
        amount_to_replicate = 2
        if kmeans_prediction == prediction_label_used:
            no_longer_vandalism_standard.append(edition_standardized)
            for classifier_index in range(0, len(ensemble)):
                standardized_inputs = [edition_standardized] * amount_to_replicate
                labels = [real_label] * amount_to_replicate

                keras_input = pd.DataFrame(standardized_inputs)
                keras_output = pd.Series(data=labels, dtype='int32')

                balanced_dataset_per_classifier[classifier_index]["x_train"] = pd.concat(
                    [balanced_dataset_per_classifier[classifier_index]["x_train"],
                     keras_input])
                balanced_dataset_per_classifier[classifier_index]["y_train"] = pd.concat(
                    [balanced_dataset_per_classifier[classifier_index]["y_train"],
                     keras_output])

        count_editions += 1

    ensemble = train_ensemble_keras(ensemble, balanced_dataset_per_classifier)

    dt_no_longer_vandalism_standard = pd.DataFrame(no_longer_vandalism_standard)
    return ensemble, dt_no_longer_vandalism_standard, running_statistical_values


def generate_train_dataset(training_dataset,
                           prediction_label_used,
                           initial_ensemble_size=12,
                           desired_distribution={0: 0.5, 1: 0.5},
                           sampling_rate=1,
                           threshold_calculate_distribution=512,
                           allowed_change_in_distribution=0.3,
                           threshold_evaluate_by_step=512):

    running_statistical_values_by_step = []
    running_statistical_values = {}
    features = training_dataset.columns.values
    actual_distribution = {0: 0, 1: 0}
    last_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}

    standard_editions = []
    no_longer_vandalism_standard = []

    # Initialize running statistics
    for individual_feature in features:
        # Initial array for the feature. The first position is the current mean and the second, the current N.
        # The third position is the number of true, the fourth position number of false. These are used for the mode.
        # The fifth is the sum of squares (used to calculate variance). The sixth is the standard deviation.
        running_statistical_values[individual_feature] = [0, 0, 0, 0, 0, 0]

    ensemble = []
    balanced_dataset_per_classifier = []
    balanced_dataset_per_classifier_aux = []
    dataset_ranges_to_evaluate_by_step = []
    initial_index_to_evaluate_by_step = 0  # We want to evaluate from the beginning of the "new" data points
    # Initialize classifiers with the ensemble of size = initial_ensemble_size
    for _ in range(0, initial_ensemble_size):
        classifier = define_classifier()
        ensemble.append(classifier)
        balanced_dataset_per_classifier.append({
            "x_train": pd.DataFrame(),
            "y_train": pd.Series()
        })

        balanced_dataset_per_classifier_aux.append({
            "x_train": [],
            "y_train": []
        })

        dataset_ranges_to_evaluate_by_step.append({
            "initial_indexes": [],  # It contains all the initial indexes for each "batch" of 512 data points
            "final_indexes": []  # It contains all the final indexes for each "batch" of 512 data points.
        })

    current_evaluate_step = 0
    no_longer_vandalism_blocks = []
    running_statistical_values_by_step = []
    data_distribution_by_step = []
    for _ in range(0, math.ceil(len(training_dataset)/threshold_evaluate_by_step)):
        no_longer_vandalism_blocks.append([])
        running_statistical_values_by_step.append({})
        data_distribution_by_step.append({
            0: 0,
            1: 0
        })

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0
    number_of_data_points = 0

    training_dataset = training_dataset.replace(False, 0)
    training_dataset = training_dataset.replace("False", 0)
    training_dataset = training_dataset.replace(True, 1)
    training_dataset = training_dataset.replace("True", 1)
    count_editions = 0
    for edition_index, edition in training_dataset.iterrows():
        real_label = edition["LABEL"]
        kmeans_prediction = edition["PREDICTION"]
        edition = edition.drop(["LABEL", "PREDICTION"])

        if real_label == 0:
            number_of_regular_editions += 1
        else:
            number_of_vandalism_editions += 1

        edition_no_null, running_statistical_values = UtilDataSetFiles.fill_null(edition,
                                                                                 running_statistical_values)

        edition_standardized = UtilDataSetFiles.apply_standardization(edition_no_null,
                                                                      running_statistical_values)

        standard_editions.append(edition_standardized)

        actual_distribution[real_label] += 1
        data_distribution_by_step[current_evaluate_step][real_label] += 1
        number_of_data_points += 1
        rate = sampling_rate * desired_distribution[real_label] / (
                    actual_distribution[real_label] / number_of_data_points)

        number_of_new_classifiers = 0
        if count_editions > last_data_distribution["count_index_edition"] + threshold_calculate_distribution:
            # I update only the value of the edition_index, because I want to visit this part only after the number
            # threshold_calculate_distribution of editions have passed.
            last_data_distribution["edition_index"] = edition_index
            last_data_distribution["count_index_edition"] = count_editions

            calculated_actual_distribution = actual_distribution[1] / actual_distribution[0]

            # If I don't have last_data_distribution, then it means that I'm passing through here for the first time.
            # So I have to give an initial value to this last_data_distribution
            if last_data_distribution["distribution"][1] == 0 or last_data_distribution["distribution"][1] == 0:
                last_data_distribution["distribution"][0] = actual_distribution[0]
                last_data_distribution["distribution"][1] = actual_distribution[1]

            calculated_last_distribution = last_data_distribution["distribution"][1] / \
                                           last_data_distribution["distribution"][0]

            '''if (calculated_last_distribution - calculated_actual_distribution > 0) and \
                    (calculated_actual_distribution / calculated_last_distribution < 1 - allowed_change_in_distribution):
                last_data_distribution["edition_index"] = edition_index
                last_data_distribution["count_index_edition"] = count_editions
                last_data_distribution["distribution"] = actual_distribution'''

            if current_evaluate_step > 29:
                calculated_last_step_data_distribution = data_distribution_by_step[current_evaluate_step-2][1] / \
                                                         data_distribution_by_step[current_evaluate_step-2][0]
                calculated_current_step_data_distribution = data_distribution_by_step[current_evaluate_step-1][1] / \
                                                            data_distribution_by_step[current_evaluate_step-1][0]

                percentage_change = 1 - (calculated_current_step_data_distribution /
                                         calculated_last_step_data_distribution)
                if (calculated_last_step_data_distribution - calculated_current_step_data_distribution > 0) and \
                        (percentage_change < 1 - allowed_change_in_distribution):
                    # Update the desired distribution as well to emphasize the minority class.
                    # Here we adjust the desired distribution by the percentage of change in class distribution.
                    '''real_change_in_distribution = calculated_actual_distribution / calculated_last_distribution
                    desired_distribution[1] += (desired_distribution[1] * allowed_change_in_distribution)
                    desired_distribution[0] = 1 - desired_distribution[1]'''
                    desired_distribution[1] = 0.5 / percentage_change
                    desired_distribution[0] = 1 - desired_distribution[0]

        for classifier_index in range(0, len(ensemble)):
            random_generator = np.random.RandomState()
            number_of_data_points_to_add = 0
            for _ in range(random_generator.poisson(rate)):
                number_of_data_points_to_add += 1

            standardized_inputs = [edition_standardized] * number_of_data_points_to_add
            labels = [real_label] * number_of_data_points_to_add

            '''keras_input = pd.DataFrame(standardized_inputs)
            keras_output = pd.Series(data=labels, dtype='int32')

            balanced_dataset_per_classifier[classifier_index]["x_train"] = pd.concat([balanced_dataset_per_classifier[classifier_index]["x_train"],
                                                                                      keras_input])
            balanced_dataset_per_classifier[classifier_index]["y_train"] = pd.concat([balanced_dataset_per_classifier[classifier_index]["y_train"],
                                                                                      keras_output])'''
            balanced_dataset_per_classifier_aux[classifier_index]["x_train"].extend(standardized_inputs)
            balanced_dataset_per_classifier_aux[classifier_index]["y_train"].extend(labels)

        if kmeans_prediction == prediction_label_used:
            # This would be the data that was re_labeled by the community.
            # Emphasize by oversampling (in this case, by using the data distribution from the minority class).
            no_longer_vandalism_standard.append(edition_standardized)
            '''rate = sampling_rate * desired_distribution[1] / (
                    actual_distribution[1] / number_of_data_points)
            amount_to_replicate = random_generator.poisson(rate)'''
            amount_to_replicate = 2

            standardized_inputs = [edition_standardized] * amount_to_replicate
            # All the labels are zero because we re_labeled this edition to no longer vandalism.
            labels = [real_label] * amount_to_replicate
            for classifier_index in range(0, len(ensemble)):
                balanced_dataset_per_classifier_aux[classifier_index]["x_train"].extend(standardized_inputs)
                balanced_dataset_per_classifier_aux[classifier_index]["y_train"].extend(labels)

        if count_editions == initial_index_to_evaluate_by_step + threshold_evaluate_by_step or \
                (count_editions == len(training_dataset) - 1):
            for classifier_index in range(0, len(ensemble)):
                aux_final_index = len(balanced_dataset_per_classifier_aux[classifier_index]["x_train"]) - 1
                dataset_ranges_to_evaluate_by_step[classifier_index]["final_indexes"].append(aux_final_index)

            initial_index_to_evaluate_by_step = initial_index_to_evaluate_by_step + threshold_evaluate_by_step + 1
            no_longer_vandalism_blocks[current_evaluate_step] = no_longer_vandalism_standard.copy()
            running_statistical_values_by_step[current_evaluate_step] = running_statistical_values
            current_evaluate_step += 1
        elif count_editions == initial_index_to_evaluate_by_step:
            for classifier_index in range(0, len(ensemble)):
                aux_initial_index = 0
                if count_editions > 0:
                    aux_initial_index = len(balanced_dataset_per_classifier_aux[classifier_index]["x_train"]) - 1

                dataset_ranges_to_evaluate_by_step[classifier_index]["initial_indexes"].append(aux_initial_index)

        count_editions += 1

    for classifier_index in range(0, len(ensemble)):
        keras_input = pd.DataFrame(balanced_dataset_per_classifier_aux[classifier_index]["x_train"])
        keras_output = pd.Series(data=balanced_dataset_per_classifier_aux[classifier_index]["y_train"], dtype='int32')

        balanced_dataset_per_classifier[classifier_index]["x_train"] = keras_input
        balanced_dataset_per_classifier[classifier_index]["y_train"] = keras_output

    dt_no_longer_vandalism_standard = pd.DataFrame(no_longer_vandalism_standard)
    return ensemble, balanced_dataset_per_classifier, dt_no_longer_vandalism_standard, running_statistical_values, \
           dataset_ranges_to_evaluate_by_step, no_longer_vandalism_blocks, running_statistical_values_by_step


def define_classifier():
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

    return model


def train_ensemble_keras(ensemble, balanced_dataset_per_classifier):
    for classifier_index in range(0, len(ensemble)):
        x_train = balanced_dataset_per_classifier[classifier_index]["x_train"]
        y_train = balanced_dataset_per_classifier[classifier_index]["y_train"]

        ensemble[classifier_index].fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

    return ensemble


def evaluate_no_longer_vandalism(ensemble, no_longer_vandalism, no_longer_vandalism_test, statistical_values):
    '''no_null_training = UtilDataSetFiles.update_missing_values_media_mode(no_longer_vandalism, statistical_values)
    standard_training = UtilDataSetFiles.apply_standardization_defined(no_null_training.copy(), scaled_values)

    no_null_testing = UtilDataSetFiles.update_missing_values_media_mode(no_longer_vandalism_test, statistical_values)
    standard_testing = UtilDataSetFiles.apply_standardization_defined(no_null_testing.copy(), scaled_values)'''
    standard_training = no_longer_vandalism
    standard_testing = no_longer_vandalism_test

    print("NO LONGER VANDALISM TRAINING DATA")
    evaluate_keras_with_testing_data(ensemble, standard_training)

    print("NO LONGER VANDALISM TESTING DATA")
    evaluate_keras_with_testing_data(ensemble, standard_testing)


def evaluate_ensemble_by_step(ensemble, current_testing, no_longer_vandalism_all_blocks, statistical_values,
                              prediction_label_used, k_validation_index, data_block_index):
    current_testing = current_testing.replace(False, 0)
    current_testing = current_testing.replace("False", 0)
    current_testing = current_testing.replace(True, 1)
    current_testing = current_testing.replace("True", 1)
    current_testing.loc[current_testing["PREDICTION"] == prediction_label_used, "LABEL"] = 0
    no_null_testing = UtilDataSetFiles.update_missing_values_media_mode_dict(current_testing, statistical_values)

    all_test_edition_standard = []
    for edition_index, edition in no_null_testing.iterrows():
        temp_edition_standard = pd.DataFrame([UtilDataSetFiles.apply_standardization(edition, statistical_values)])
        #temp_edition_standard = UtilDataSetFiles.apply_standardization(edition, statistical_values)
        all_test_edition_standard.append(temp_edition_standard)

    testing_standard = pd.concat(all_test_edition_standard)
    # testing_standard = pd.DataFrame(columns=no_null_testing.columns, data=[all_test_edition_standard])

    dt_no_longer_vandalism_test = testing_standard.loc[testing_standard["PREDICTION"] == prediction_label_used]

    save_ensemble_results_by_step(ensemble, testing_standard, k_validation_index, data_block_index,
                                  ["LABEL", "PREDICTION"])
    no_longer_vandalism_all_blocks["LABEL"] = 0
    save_no_longer_vandalism_results_by_step(ensemble, no_longer_vandalism_all_blocks, k_validation_index, False,
                                             labels_to_drop=["LABEL"])
    save_no_longer_vandalism_results_by_step(ensemble, dt_no_longer_vandalism_test, k_validation_index, True,
                                             ["LABEL", "PREDICTION"])


def save_ensemble_results_by_step(ensemble, testing_dataset, k_validation_index, data_block_index, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = pd.Series(data=testing_dataset['LABEL'], dtype='int32')

    y_real = []

    for data_point_index in range(0, len(x_test)):
        y_real.append(y_test.iloc[data_point_index])

    y_predicted = get_ensemble_result_keras(ensemble, x_test)

    #calculated_classification_report = classification_report(y_real, y_predicted)
    #print(calculated_classification_report)

    calculated_confusion_matrix = confusion_matrix(y_test, y_predicted)
    #print(calculated_confusion_matrix)

    corrected_classified_regular = calculated_confusion_matrix[0][0] / (calculated_confusion_matrix[0][0] +
                                                                        calculated_confusion_matrix[0][1])

    corrected_classified_vandalism = calculated_confusion_matrix[1][1] / (calculated_confusion_matrix[1][0] +
                                                                          calculated_confusion_matrix[1][1])

    overall_recall = (corrected_classified_regular + corrected_classified_vandalism) / 2

    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/ExperimentsResults/"
    file_name = "results_k_validation_" + str(k_validation_index) + ".txt"
    text_file_path = file_path + file_name
    text_file = open(text_file_path, 'a')

    text_file.write("Data Block Index: " + str(data_block_index) + "\n")
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

    return calculated_confusion_matrix


def save_no_longer_vandalism_results_by_step(ensemble, testing_dataset, k_validation_index,
                                             jump_lines=False, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = pd.Series(data=testing_dataset['LABEL'], dtype='int32')

    y_real = []

    for data_point_index in range(0, len(x_test)):
        y_real.append(y_test.iloc[data_point_index])

    y_predicted = get_ensemble_result_keras(ensemble, x_test)

    #calculated_classification_report = classification_report(y_real, y_predicted)
    calculated_confusion_matrix = confusion_matrix(y_test, y_predicted)

    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/ExperimentsResults/"
    file_name = "results_k_validation_" + str(k_validation_index) + ".txt"
    text_file_path = file_path + file_name
    text_file = open(text_file_path, 'a')

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

    return calculated_confusion_matrix


def save_keras_model(trained_models, folder_name, path_to_save):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = base_directory + path_to_save + folder_name + str(model_index)
        model.save(file_path)


def get_trained_keras_models(ensemble_size, models_path):
    trained_models = []

    folder_name = "model"
    for model_index in range(0, ensemble_size):
        model_folder = models_path + folder_name + str(model_index)
        trained_models.append(keras.models.load_model(model_folder))

    return trained_models


def evaluate_ensemble_with_testing_data(record_trained_models=True, ml_model="rf", number_of_experiments=10,
                                        ensemble_size=12, labels_to_drop=["LABEL"]):
    biggest_overall_recall = -1

    for current_experiment in range(0, number_of_experiments):
        dataset_file_path = str(pathlib.Path(__file__).parent) + '/current_testing.csv'
        features, data_input, data_output = get_data_from_csv(dataset_file_path)

        trained_model_per_predictors, running_statistics = train_ensemble(ml_model=ml_model,
                                                                          ensemble_size=ensemble_size)

        standardized_inputs = []
        for input_aux, output_aux in zip(data_input, data_output):
            input_dict = dict(zip(features, input_aux))  # This is necessary because river works with dictionary
            if output_aux == "reg" or not output_aux:
                output_aux = 0
            else:
                output_aux = 1

            input_dict_no_null = UtilDataSetFiles.fill_null_no_update(input_dict, running_statistics)

            input_dict_standardized = UtilDataSetFiles.apply_standardization(input_dict_no_null,
                                                                             running_statistics)
            input_dict_standardized["LABEL"] = output_aux
            standardized_inputs.append(input_dict_standardized)

        testing_dataset = pd.DataFrame(standardized_inputs)

        overall_recall = evaluate_keras_with_testing_data(trained_model_per_predictors, testing_dataset,
                                                          labels_to_drop)

        if record_trained_models and overall_recall > biggest_overall_recall:
            biggest_overall_recall = overall_recall
            if ml_model == "sparse_lr":
                save_trained_models(trained_model_per_predictors, "model",
                                    "/experiments/article_datasets/trained_sparse_lr_models/")
            elif ml_model == "rf":
                save_trained_models(trained_model_per_predictors, "model",
                                    "/experiments/article_datasets/trained_rf_models/")

            elif ml_model == "nn":
                save_trained_models(trained_model_per_predictors, "model",
                                    "/experiments/article_datasets/trained_nn_models/")

            print("Biggest Overall Recall \n \n \n")


def evaluate_keras_with_testing_data(trained_models, testing_dataset, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = testing_dataset['LABEL']

    y_real = []

    for data_point_index in range(0, len(x_test)):
        y_real.append(y_test.iloc[data_point_index])

    y_predicted = get_ensemble_result_keras(trained_models, x_test)

    calculated_classification_report = classification_report(y_real, y_predicted)
    print(calculated_classification_report)

    calculated_confusion_matrix = confusion_matrix(y_test, y_predicted)
    print(calculated_confusion_matrix)

    corrected_classified_regular = calculated_confusion_matrix[0][0] / (calculated_confusion_matrix[0][0] +
                                                                        calculated_confusion_matrix[0][1])

    corrected_classified_vandalism = calculated_confusion_matrix[1][1] / (calculated_confusion_matrix[1][0] +
                                                                          calculated_confusion_matrix[1][1])

    overall_recall = (corrected_classified_regular + corrected_classified_vandalism) / 2

    return overall_recall


def evaluate_ensemble_no_longer_vandalism(ensemble, testing_dataset, labels_to_drop=["LABEL"]):
    x_test = testing_dataset.drop(labels_to_drop, axis=1)
    y_test = pd.Series(data=testing_dataset['LABEL'], dtype='int32')

    y_real = []

    for data_point_index in range(0, len(x_test)):
        y_real.append(y_test.iloc[data_point_index])

    y_predicted = get_ensemble_result_keras(ensemble, x_test)

    calculated_classification_report = classification_report(y_real, y_predicted)
    print(calculated_classification_report)

    calculated_confusion_matrix = confusion_matrix(y_test, y_predicted)
    print(calculated_confusion_matrix)

    '''corrected_classified_regular = calculated_confusion_matrix[0][0] / (calculated_confusion_matrix[0][0] +
                                                                        calculated_confusion_matrix[0][1])'''

    corrected_classified_regular = calculated_confusion_matrix[0][0]

    print("CORRECTLY CLASSIFIED NO LONGER VANDALISM: " + str(corrected_classified_regular))

    return corrected_classified_regular


def get_ensemble_result_keras(trained_models, x_test):
    models_predictions = []

    for model in trained_models:
        predictions = model.predict(x_test)
        #classes_highest_probability = np.argmax(predictions, axis=1)  # This gets the class with highest probability
        models_predictions.append(predictions)

    classes_summed = np.sum(models_predictions, axis=0)
    ensemble_prediction = np.argmax(classes_summed, axis=1)

    return ensemble_prediction


def evaluate_ensemble_re_label_data(record_trained_models=True, ml_model="sparse_lr", number_of_experiments=10,
                                    ensemble_size=12, training_file_name="/re_labeled.csv",
                                    testing_file_name="/re_label_testing.csv",
                                    training_file_no_longer="re_label_no_longer_vandalism.csv",
                                    testing_file_no_longer="re_label_no_longer_vandalism_test.csv",
                                    labels_to_drop=["LABEL"]):
    biggest_overall_recall = -1

    for current_experiment in range(0, number_of_experiments):
        dataset_file_path = str(pathlib.Path(__file__).parent) + '/keras/re_training/re_label' + testing_file_name
        features, data_input, data_output = get_data_from_csv_re_label(dataset_file_path)

        trained_model_per_predictors, running_statistics = train_ensemble("/keras/re_training/re_label",
                                                                          training_file_name,
                                                                          True, ml_model=ml_model,
                                                                          ensemble_size=ensemble_size)

        standardized_inputs = []
        for input_aux, output_aux in zip(data_input, data_output):
            input_dict = dict(zip(features, input_aux))  # This is necessary because river works with dictionary
            if output_aux == "reg" or not output_aux:
                output_aux = 0
            else:
                output_aux = 1

            input_dict_no_null = UtilDataSetFiles.fill_null_no_update(input_dict, running_statistics)

            input_dict_standardized = UtilDataSetFiles.apply_standardization(input_dict_no_null,
                                                                             running_statistics)
            input_dict_standardized["LABEL"] = output_aux
            standardized_inputs.append(input_dict_standardized)

        testing_dataset = pd.DataFrame(standardized_inputs)

        overall_recall = evaluate_keras_with_testing_data(trained_model_per_predictors, testing_dataset,
                                                          labels_to_drop)

        '''if record_trained_models and overall_recall > biggest_overall_recall:
            biggest_overall_recall = overall_recall
            if ml_model == "nn":
                save_trained_models(trained_model_per_predictors, "model",
                                    "/experiments/article_datasets/re_training/re_label/trained_nn_models/")
            else:
                save_trained_models(trained_model_per_predictors, "model",
                                    "/experiments/article_datasets/re_training/re_label/trained_rf_models/")
            print("Biggest Overall Recall \n \n \n")'''

        print("NO LONGER VANDALISM TRAINING")
        training_no_longer_file_path = str(pathlib.Path(__file__).parent) + '/keras/re_training/re_label' + training_file_no_longer
        features_no_longer, data_input_no_longer, data_output_no_longer = get_data_from_csv_re_label(training_no_longer_file_path)
        standardized_inputs_training_no_longer = []
        for input_aux, output_aux in zip(data_input_no_longer, data_output_no_longer):
            input_dict = dict(zip(features_no_longer, input_aux))  # This is necessary because river works with dictionary
            if output_aux == "reg" or not output_aux:
                output_aux = 0
            else:
                output_aux = 1

            input_dict_no_null = UtilDataSetFiles.fill_null_no_update(input_dict, running_statistics)

            input_dict_standardized = UtilDataSetFiles.apply_standardization(input_dict_no_null,
                                                                             running_statistics)
            input_dict_standardized["LABEL"] = output_aux
            standardized_inputs_training_no_longer.append(input_dict_standardized)

        training_dataset_no_longer = pd.DataFrame(standardized_inputs_training_no_longer)

        overall_recall = evaluate_keras_with_testing_data(trained_model_per_predictors, training_dataset_no_longer,
                                                          labels_to_drop)


        testing_no_longer_file_path = str(
            pathlib.Path(__file__).parent) + '/keras/re_training/re_label' + testing_file_no_longer
        features_test_no_longer, data_input_test_no_longer, data_output_test_no_longer = get_data_from_csv_re_label(
            testing_no_longer_file_path)

        print("NO LONGER VANDALISM TESTING")
        standardized_inputs_testing_no_longer = []
        for input_aux, output_aux in zip(data_input_test_no_longer, data_output_test_no_longer):
            input_dict = dict(zip(features_test_no_longer, input_aux))  # This is necessary because river works with dictionary
            if output_aux == "reg" or not output_aux:
                output_aux = 0
            else:
                output_aux = 1

            input_dict_no_null = UtilDataSetFiles.fill_null_no_update(input_dict, running_statistics)

            input_dict_standardized = UtilDataSetFiles.apply_standardization(input_dict_no_null,
                                                                             running_statistics)
            input_dict_standardized["LABEL"] = output_aux
            standardized_inputs_testing_no_longer.append(input_dict_standardized)

        testing_dataset_no_longer = pd.DataFrame(standardized_inputs_testing_no_longer)

        overall_recall = evaluate_keras_with_testing_data(trained_model_per_predictors, testing_dataset_no_longer,
                                                          labels_to_drop)



def train_ensemble(file_directory="", file_name="/current_training.csv", re_label=False, ensemble_size=12,
                   ml_model="sparse_lr"):
    dataset_file_path = str(pathlib.Path(__file__).parent) + file_directory + file_name

    features = []
    data_input = []
    data_output = []

    if re_label:
        features, data_input, data_output = get_data_from_csv_re_label(dataset_file_path)
    else:
        features, data_input, data_output = get_data_from_csv(dataset_file_path)

    # Temporary! I just wanna test the re_training data. No past data.
    data_input = data_input[16220:len(data_input) - 1]
    data_output = data_output[16220:len(data_output) - 1]

    trained_model_per_predictor = []
    evaluation_metric_per_predictor = []

    for predictor in range(0, ensemble_size):
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

        trained_model_per_predictor.append(model)

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0

    running_statistical_values = {}
    balanced_dataset_per_model = []

    for ensemble_index in range(0, ensemble_size):
        # Initialize running statistics
        for individual_feature in features:
            # Initial array for the feature. The first position is the current mean and the second, the current N.
            # The third position is the number of true, the fourth position number of false. These are used for the mode.
            # The fifth is the sum of squares (used to calculate variance). The sixth is the standard deviation.
            running_statistical_values[individual_feature] = [0, 0, 0, 0, 0, 0]

        standardized_inputs = []
        keras_output = []

        desired_distribution = {0: 0.5, 1: 0.5}
        actual_distribution = {0: 0, 1: 0}
        number_of_data_points = 0
        sampling_rate = 1
        random_generator = np.random.RandomState(42)

        for input_aux, output_aux in zip(data_input, data_output):
            input_dict = dict(zip(features, input_aux))  # This is necessary because river works with dictionary
            if output_aux == "reg" or not output_aux:
                output_aux = 0
                number_of_regular_editions += 1
            else:
                output_aux = 1
                number_of_vandalism_editions += 1

            input_dict_no_null, running_statistical_values = UtilDataSetFiles.fill_null(input_dict,
                                                                                        running_statistical_values)

            input_dict_standardized = UtilDataSetFiles.apply_standardization(input_dict_no_null,
                                                                             running_statistical_values)

            actual_distribution[output_aux] += 1
            number_of_data_points += 1

            rate = sampling_rate * desired_distribution[output_aux] / (actual_distribution[output_aux] / number_of_data_points)

            number_of_data_points_to_add = 0
            for _ in range(random_generator.poisson(rate)):
                number_of_data_points_to_add += 1

            standardized_inputs.extend([input_dict_standardized] *  number_of_data_points_to_add)
            keras_output.extend([output_aux] * number_of_data_points_to_add)

        keras_input = pd.DataFrame(standardized_inputs)
        keras_output = pd.Series(data=keras_output, dtype='int32')

        balanced_dataset_per_model.append({"x_train": keras_input, "y_train": keras_output})

    '''for index in range(0, len(keras_input)):
        data_to_train_aux = pd.DataFrame([keras_input.loc[index]])
        data_to_train_y_aux = pd.Series(data=[keras_output.loc[index]], dtype='int32')

        trained_model_per_predictor[0].fit(data_to_train_aux, data_to_train_y_aux, epochs=1, batch_size=1, verbose=0)'''

    for predictor in range(0, ensemble_size):
        x_train = balanced_dataset_per_model[predictor]["x_train"]
        y_train = balanced_dataset_per_model[predictor]["y_train"]
        trained_model_per_predictor[predictor].fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

    return trained_model_per_predictor, running_statistical_values


def get_ensemble_result(editions_vandalism):
    vandalism_count = 0
    regular_count = 0
    for predictor_result in editions_vandalism:
        # For classification, we should use only predictor_result. Checking if it's true.
        if predictor_result >= 0.5:
            vandalism_count += 1
        else:
            regular_count += 1

    if vandalism_count >= regular_count:
        return True

    return False


def get_data_from_csv(dataset_file_path):
    data_set = pd.read_csv(dataset_file_path, na_values="?")
    number_columns = len(data_set.columns)

    data_set = data_set.replace(False, 0)
    data_set = data_set.replace("False", 0)
    data_set = data_set.replace(True, 1)
    data_set = data_set.replace("True", 1)
    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    features = data_set.columns.values
    data_input = data_set.iloc[:, :-1].values  # It gets all rows and columns, except the last column.
    data_output = data_set.iloc[:, number_columns-1]  # It gets the column with the label.

    return features, data_input, data_output


def get_data_from_csv_re_label(dataset_file_path):
    data_set = pd.read_csv(dataset_file_path, na_values="?")
    number_columns = len(data_set.columns)

    data_set = data_set.replace(False, 0)
    data_set = data_set.replace("False", 0)
    data_set = data_set.replace(True, 1)
    data_set = data_set.replace("True", 1)
    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    if number_columns == 60:
        features = data_set.columns.values
        data_input = data_set.iloc[:, :-3].values  # It gets all rows and columns, except the last column.
        data_output = data_set.iloc[:, number_columns - 3]  # It gets the column with the label.

        return features, data_input, data_output
    else:
        features = data_set.columns.values
        data_input = data_set.iloc[:, :-2].values  # It gets all rows and columns, except the last column.
        data_output = data_set.iloc[:, number_columns - 2]  # It gets the column with the label.

        return features, data_input, data_output


def save_trained_models(trained_models, file_name, path_to_save):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = base_directory + path_to_save + file_name + str(model_index) + ".pickle"
        pickle.dump(model, open(file_path, 'wb'))
