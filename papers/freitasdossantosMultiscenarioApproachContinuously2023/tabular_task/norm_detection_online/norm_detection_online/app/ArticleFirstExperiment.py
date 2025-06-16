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


def start_training_k_validation_per_step(initial_ensemble_size=12,
                                         number_k_of_folders=10, directory_to_save="",
                                         threshold_evaluate_by_step=512, load_from_disk=False):
    if load_from_disk:
        for k_validation_index in range(0, number_k_of_folders):
            current_training, balanced_dataset_per_classifier, statistical_values, dataset_ranges_to_evaluate_by_step, \
            running_statistical_values_by_step, current_testing = get_dataset_online_information(k_validation_index)

            trained_ensemble = []
            for _ in range(0, initial_ensemble_size):
                classifier = define_classifier()
                trained_ensemble.append(classifier)

            for current_step in range(0, math.floor(len(current_training)/threshold_evaluate_by_step)):
                aux_balanced_dataset_per_classifier = []
                for classifier_index in range(0, len(trained_ensemble)):
                    initial_index = dataset_ranges_to_evaluate_by_step[classifier_index]["initial_indexes"][current_step]
                    final_index = dataset_ranges_to_evaluate_by_step[classifier_index]["final_indexes"][current_step]

                    aux_balanced_dataset_per_classifier.append({
                        "x_train": balanced_dataset_per_classifier[classifier_index]["x_train"][initial_index:final_index],
                        "y_train": balanced_dataset_per_classifier[classifier_index]["y_train"][initial_index:final_index]
                    })

                trained_ensemble = train_ensemble_keras(trained_ensemble, aux_balanced_dataset_per_classifier)

                evaluate_ensemble_by_step(trained_ensemble, current_testing.copy(), statistical_values,
                                          k_validation_index, current_step)

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

            evaluate_keras_with_testing_data(trained_ensemble, testing_standard, ["LABEL"])
    else:
        old_training = UtilDataSetFiles.get_data_set_from_file("/experiments/keras/",
                                                               "current_training.csv")
        future_training = UtilDataSetFiles.get_data_set_from_file("/experiments/keras/",
                                                                  "future_training.csv")

        complete_dataset = pd.concat([old_training, future_training])

        # Separating the complete dataset into k folders.
        folders = np.array_split(complete_dataset, number_k_of_folders)

        dataset_columns = complete_dataset.columns
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
            trained_ensemble, balanced_dataset_per_classifier, statistical_values, dataset_ranges_to_evaluate_by_step, \
            running_statistical_values_by_step = generate_train_dataset(current_training)

            save_dataset_online_information(current_training, balanced_dataset_per_classifier,
                                            statistical_values, dataset_ranges_to_evaluate_by_step,
                                            running_statistical_values_by_step,
                                            current_testing, k_validation_index)

            '''for current_step in range(0, math.ceil(len(current_training)/threshold_evaluate_by_step)):
                aux_balanced_dataset_per_classifier = []
                for classifier_index in range(0, len(trained_ensemble)):
                    initial_index = dataset_ranges_to_evaluate_by_step[classifier_index]["initial_indexes"][current_step]
                    final_index = dataset_ranges_to_evaluate_by_step[classifier_index]["final_indexes"][current_step]

                    aux_balanced_dataset_per_classifier.append({
                        "x_train": balanced_dataset_per_classifier[classifier_index]["x_train"][initial_index:final_index],
                        "y_train": balanced_dataset_per_classifier[classifier_index]["y_train"][initial_index:final_index]
                    })

                trained_ensemble = train_ensemble_keras(trained_ensemble, aux_balanced_dataset_per_classifier)

                evaluate_ensemble_by_step(trained_ensemble, current_testing.copy(), statistical_values,
                                          k_validation_index, current_step)

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

            evaluate_keras_with_testing_data(trained_ensemble, testing_standard, ["LABEL"])'''

            k_validation_index += 1


def generate_train_dataset(training_dataset,
                           initial_ensemble_size=12,
                           desired_distribution={0: 0.5, 1: 0.5},
                           sampling_rate=1,
                           threshold_calculate_distribution=512,
                           allowed_change_in_distribution=0.3,
                           threshold_evaluate_by_step=512):

    running_statistical_values = {}
    features = training_dataset.columns.values
    actual_distribution = {0: 0, 1: 0}
    last_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 0, "count_index_edition": 0}

    standard_editions = []

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
    running_statistical_values_by_step = []
    for _ in range(0, math.ceil(len(training_dataset)/threshold_evaluate_by_step)):
        running_statistical_values_by_step.append({})

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
        edition = edition.drop(["LABEL"])

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

        '''if count_editions > last_data_distribution["count_index_edition"] + threshold_calculate_distribution:
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
                last_data_distribution["edition_index"] = edition_index
                last_data_distribution["count_index_edition"] = count_editions
                last_data_distribution["distribution"] = actual_distribution
        '''
        for classifier_index in range(0, len(ensemble)):
            random_generator = np.random.RandomState()
            number_of_data_points_to_add = 0
            for _ in range(random_generator.poisson(rate)):
                number_of_data_points_to_add += 1

            standardized_inputs = [edition_standardized] * number_of_data_points_to_add
            labels = [real_label] * number_of_data_points_to_add

            '''balanced_dataset_per_classifier[classifier_index]["x_train"] = pd.concat([balanced_dataset_per_classifier[classifier_index]["x_train"],
                                                                                      keras_input])
            balanced_dataset_per_classifier[classifier_index]["y_train"] = pd.concat([balanced_dataset_per_classifier[classifier_index]["y_train"],
                                                                                      keras_output])'''
            balanced_dataset_per_classifier_aux[classifier_index]["x_train"].extend(standardized_inputs)
            balanced_dataset_per_classifier_aux[classifier_index]["y_train"].extend(labels)

        if count_editions == initial_index_to_evaluate_by_step + threshold_evaluate_by_step or \
                (count_editions == len(training_dataset) - 1):
            for classifier_index in range(0, len(ensemble)):
                aux_final_index = len(balanced_dataset_per_classifier_aux[classifier_index]["x_train"]) - 1
                dataset_ranges_to_evaluate_by_step[classifier_index]["final_indexes"].append(aux_final_index)

            initial_index_to_evaluate_by_step = initial_index_to_evaluate_by_step + threshold_evaluate_by_step + 1
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

    return ensemble, balanced_dataset_per_classifier, running_statistical_values, dataset_ranges_to_evaluate_by_step, \
           running_statistical_values_by_step


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


def evaluate_ensemble_by_step(ensemble, current_testing, statistical_values, k_validation_index, data_block_index):
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

    save_ensemble_results_by_step(ensemble, testing_standard, k_validation_index, data_block_index, ["LABEL"])


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


def save_ensemble_results_by_step(ensemble, testing_dataset, k_validation_index, data_block_index,
                                  labels_to_drop=["LABEL"]):
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


def save_dataset_online_information(current_training, balanced_dataset_per_classifier, statistical_values,
                                    dataset_ranges_to_evaluate_by_step, running_statistical_values_by_step,
                                    current_testing, k_validation_index):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/DatasetInformationNoDrift/" + \
                str(k_validation_index) + "/"

    current_training_file_name = "current_training.pkl"
    current_training_path = file_path + current_training_file_name
    UtilDataSetFiles.save_pickle_objects(current_training, current_training_path)

    balanced_dataset_per_classifier_file_name = "balanced_dataset_per_classifier.pkl"
    complete_balanced_dataset_path = file_path + balanced_dataset_per_classifier_file_name
    UtilDataSetFiles.save_pickle_objects(balanced_dataset_per_classifier, complete_balanced_dataset_path)

    statistical_values_file_name = "statistical_values.pkl"
    complete_statistical_values_path = file_path + statistical_values_file_name
    UtilDataSetFiles.save_pickle_objects(statistical_values, complete_statistical_values_path)

    range_to_evaluate_file_name = "dataset_range_to_evaluate_by_step.pkl"
    complete_range_to_evaluate_path = file_path + range_to_evaluate_file_name
    UtilDataSetFiles.save_pickle_objects(dataset_ranges_to_evaluate_by_step, complete_range_to_evaluate_path)

    running_statistical_values_by_step_file_name = "running_statistical_values_by_step.pkl"
    complete_running_statistical_values_by_step_path = file_path + running_statistical_values_by_step_file_name
    UtilDataSetFiles.save_pickle_objects(running_statistical_values_by_step,
                                         complete_running_statistical_values_by_step_path)

    current_testing_file_name = "current_testing.pkl"
    complete_current_testing_path = file_path + current_testing_file_name
    UtilDataSetFiles.save_pickle_objects(current_testing, complete_current_testing_path)


def get_dataset_online_information(k_validation_index):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/experiments/keras/DatasetInformationNoDrift/" + \
                str(k_validation_index) + "/"

    current_training_file_name = "current_training.pkl"
    current_training_path = file_path + current_training_file_name
    current_training = UtilDataSetFiles.load_pickle_objects(current_training_path)

    balanced_dataset_per_classifier_file_name = "balanced_dataset_per_classifier.pkl"
    complete_balanced_per_classifier_path = file_path + balanced_dataset_per_classifier_file_name
    balanced_dataset_per_classifier = UtilDataSetFiles.load_pickle_objects(complete_balanced_per_classifier_path)

    statistical_values_file_name = "statistical_values.pkl"
    complete_statistical_values_path = file_path + statistical_values_file_name
    statistical_values = UtilDataSetFiles.load_pickle_objects(complete_statistical_values_path)

    range_to_evaluate_file_name = "dataset_range_to_evaluate_by_step.pkl"
    complete_range_to_evaluate_path = file_path + range_to_evaluate_file_name
    dataset_ranges_to_evaluate_by_step = UtilDataSetFiles.load_pickle_objects(complete_range_to_evaluate_path)

    running_statistical_values_by_step_file_name = "running_statistical_values_by_step.pkl"
    complete_running_statistical_values_by_step_path = file_path + running_statistical_values_by_step_file_name
    running_statistical_values_by_step = UtilDataSetFiles.\
        load_pickle_objects(complete_running_statistical_values_by_step_path)

    current_testing_file_name = "current_testing.pkl"
    complete_current_testing_path = file_path + current_testing_file_name
    current_testing = UtilDataSetFiles.load_pickle_objects(complete_current_testing_path)

    return current_training, balanced_dataset_per_classifier, statistical_values, dataset_ranges_to_evaluate_by_step, \
           running_statistical_values_by_step, current_testing


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


def get_ensemble_result_keras(trained_models, x_test):
    models_predictions = []

    for model in trained_models:
        predictions = model.predict(x_test)
        #classes_highest_probability = np.argmax(predictions, axis=1)  # This gets the class with highest probability
        models_predictions.append(predictions)

    classes_summed = np.sum(models_predictions, axis=0)
    ensemble_prediction = np.argmax(classes_summed, axis=1)

    return ensemble_prediction


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


def save_trained_models(trained_models, file_name, path_to_save):
    base_directory = str(pathlib.Path(__file__).parent.parent.parent)

    for model, model_index in zip(trained_models, range(len(trained_models))):
        file_path = base_directory + path_to_save + file_name + str(model_index) + ".pickle"
        pickle.dump(model, open(file_path, 'wb'))
