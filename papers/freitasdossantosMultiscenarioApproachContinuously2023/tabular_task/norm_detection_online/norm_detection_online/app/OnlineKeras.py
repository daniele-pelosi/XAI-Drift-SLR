import math
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras
from tensorflow.keras import layers

from norm_detection_online.util import UtilDataSetFiles


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
                     initial_ensemble_size=12,
                     desired_distribution={0: 0.5, 1: 0.5},
                     sampling_rate=1,
                     threshold_calculate_distribution=500,
                     allowed_change_in_distribution=0.3,
                     max_number_of_classifiers=40):
    running_statistical_values = {}
    features = training_dataset.columns.values
    actual_distribution = {0: 0, 1: 0}
    last_data_distribution = {"distribution": {0: 0, 1: 0}, "edition_index": 16220}

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

    for edition_index, edition in training_dataset.iterrows():
        real_label = edition["LABEL"]
        edition = edition.drop("LABEL")

        if real_label == 0:
            number_of_regular_editions += 1
        else:
            number_of_vandalism_editions += 1

        edition_no_null, running_statistical_values = UtilDataSetFiles.fill_null(edition,
                                                                                 running_statistical_values)

        edition_standardized = UtilDataSetFiles.apply_standardization(edition_no_null,
                                                                      running_statistical_values)

        actual_distribution[real_label] += 1
        number_of_data_points += 1
        rate = sampling_rate * desired_distribution[real_label] / (
                    actual_distribution[real_label] / number_of_data_points)

        number_of_new_classifiers = 0
        if edition_index > last_data_distribution["edition_index"] + threshold_calculate_distribution:
            current_best_number_of_classifiers = math.ceil(actual_distribution[0] /
                                                           actual_distribution[1])

            # I update only the value of the edition_index, because I want to visit this part only after the number
            # threshold_calculate_distribution of editions have passed.
            last_data_distribution["edition_index"] = edition_index

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
                number_of_new_classifiers = current_best_number_of_classifiers - len(ensemble)
                last_data_distribution["edition_index"] = edition_index
                last_data_distribution["distribution"] = actual_distribution

                for _ in range(0, number_of_new_classifiers):
                    classifier = define_classifier()
                    ensemble.append(classifier)
                    balanced_dataset_per_classifier.append({
                        "x_train": pd.DataFrame(),
                        "y_train": pd.Series()
                    })

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

    ensemble = train_ensemble_keras(ensemble, balanced_dataset_per_classifier)

    return ensemble, running_statistical_values


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
