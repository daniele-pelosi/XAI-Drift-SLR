import pathlib
import pickle

import numpy as np
import pandas as pd
from river import compose, ensemble, optim, stats, drift, neural_net
from river import linear_model
from river import metrics
from river import preprocessing
from river import imblearn
from river import evaluate
from sklearn.model_selection import train_test_split

from norm_detection_online.util import Util, UtilDataSetFiles


def train_wikipedia_dataset():

    dataset_file_path = str(pathlib.Path(__file__).parent) + "\current_training.csv"
    # dataset_file_path = str(pathlib.Path(__file__).parent) + "/balanced_dataset0.csv"
    features, data_input, data_output = get_data_from_csv(dataset_file_path)

    # linear_model.LogisticRegression()
    # ensemble.AdaptiveRandomForestClassifier(seed=42)
    '''model = compose.Pipeline(
        preprocessing.StandardScaler() |
        imblearn.RandomOverSampler(
            classifier=ensemble.AdaptiveRandomForestClassifier(seed=42),
            desired_dist={0: .5, 1: .5},
            seed=42
        )
    )'''
    model = compose.Pipeline(
        preprocessing.StandardScaler() |
        imblearn.RandomSampler(
            classifier=ensemble.AdaptiveRandomForestClassifier(seed=42),
            desired_dist={0: .5, 1: .5},
            sampling_rate=1,
            seed=42
        )
    )

    evaluation_metric = metrics.Accuracy()
    #rocauc_metric = metrics.ROCAUC()

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0

    for input_aux, output_aux in zip(data_input, data_output):
        input_dict = dict(zip(features, input_aux))  # This is necessary because river works with dictionary
        if output_aux == "reg" or output_aux == 0:
            output_aux = False
            number_of_regular_editions += 1
        else:
            output_aux = True
            number_of_vandalism_editions += 1

        estimated_prediction = model.predict_proba_one(input_dict)
        estimated_prediction_class = model.predict_one(input_dict)

        model.learn_one(input_dict, output_aux)

        evaluation_metric.update(output_aux, estimated_prediction_class)
        #rocauc_metric.update(output_aux, estimated_prediction_class)

    regular_regular = evaluation_metric.cm.data[False][False]
    regular_vandalism = evaluation_metric.cm.data[False][True]
    vandalism_vandalism = evaluation_metric.cm.data[True][True]
    vandalism_regular = evaluation_metric.cm.data[True][False]

    predictor_path = str(pathlib.Path(__file__).parent)

    #print(str(rocauc_metric))

    text_file_path = predictor_path + "/results.txt"
    text_file = open(text_file_path, 'w')

    text_file.write("Overall Accuracy: " + str(evaluation_metric) + "\n")

    text_file.write("Label Regular and Classified as Regular: " + str(regular_regular) + "\n")
    text_file.write("Label Regular and Classified as Vandalism: " + str(regular_vandalism) + "\n")
    text_file.write("Label Vandalism and Classified as Vandalism: " + str(vandalism_vandalism) + "\n")
    text_file.write("Label Vandalism and Classified as Regular: " + str(vandalism_regular) + "\n")

    text_file.write("Number of Regular Editions: " + str(number_of_regular_editions) + "\n")
    text_file.write("Number of Vandalism Editions: " + str(number_of_vandalism_editions) + "\n")

    text_file.write("Percentage Corrected Classified Regular: " + str(regular_regular / number_of_regular_editions) + "\n")
    text_file.write("Percentage Corrected Classified Vandalism: " + str(vandalism_vandalism / number_of_vandalism_editions) + "\n")

    text_file.close()

    test_online_model(model)

    return model


def test_online_model(model):
    dataset_testing_file_path = str(pathlib.Path(__file__).parent) + "\current_testing.csv"
    features_testing, data_input_testing, data_output_testing = get_data_from_csv(dataset_testing_file_path)

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0
    evaluation_metric = metrics.Accuracy()

    for input_aux, output_aux in zip(data_input_testing, data_output_testing):
        input_dict = dict(zip(features_testing, input_aux))  # This is necessary because river works with dictionary
        if output_aux == "reg" or output_aux == 0:
            output_aux = False
            number_of_regular_editions += 1
        else:
            output_aux = True
            number_of_vandalism_editions += 1

        estimated_prediction_class = model.predict_one(input_dict)

        evaluation_metric.update(output_aux, estimated_prediction_class)
        # rocauc_metric.update(output_aux, estimated_prediction_class)

    regular_regular = evaluation_metric.cm.data[False][False]
    regular_vandalism = evaluation_metric.cm.data[False][True]
    vandalism_vandalism = evaluation_metric.cm.data[True][True]
    vandalism_regular = evaluation_metric.cm.data[True][False]

    predictor_path = str(pathlib.Path(__file__).parent)

    # print(str(rocauc_metric))

    text_file_path = predictor_path + "/results_testing.txt"
    text_file = open(text_file_path, 'w')

    text_file.write("Overall Accuracy: " + str(evaluation_metric) + "\n")

    text_file.write("Label Regular and Classified as Regular: " + str(regular_regular) + "\n")
    text_file.write("Label Regular and Classified as Vandalism: " + str(regular_vandalism) + "\n")
    text_file.write("Label Vandalism and Classified as Vandalism: " + str(vandalism_vandalism) + "\n")
    text_file.write("Label Vandalism and Classified as Regular: " + str(vandalism_regular) + "\n")

    text_file.write("Number of Regular Editions: " + str(number_of_regular_editions) + "\n")
    text_file.write("Number of Vandalism Editions: " + str(number_of_vandalism_editions) + "\n")

    text_file.write(
        "Percentage Corrected Classified Regular: " + str(regular_regular / number_of_regular_editions) + "\n")
    text_file.write(
        "Percentage Corrected Classified Vandalism: " + str(vandalism_vandalism / number_of_vandalism_editions) + "\n")

    text_file.close()


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

    trained_model_per_predictor = []
    evaluation_metric_per_predictor = []

    for predictor in range(0, ensemble_size):
        model = compose.Pipeline()

        if ml_model == "sparse_lr":
            model = compose.Pipeline(
                preprocessing.StandardScaler() |
                imblearn.RandomSampler(
                    classifier=linear_model.LogisticRegression(optimizer=optim.SGD(.01)),
                    desired_dist={0: .5, 1: .5},
                    sampling_rate=1
                )
            )
        elif ml_model == "rf":
            model = compose.Pipeline(
                preprocessing.StandardScaler() |
                imblearn.RandomSampler(
                    classifier=ensemble.AdaptiveRandomForestClassifier(seed=42, n_models=200),
                    desired_dist={0: .5, 1: .5},
                    sampling_rate=0.01
                )
            )
        elif ml_model == "nn":
            model = compose.Pipeline(
                preprocessing.StandardScaler() |
                imblearn.RandomSampler(
                    classifier=neural_net.MLPRegressor(
                        hidden_dims=(6, 3),
                        activations=(neural_net.activations.ReLU,
                                     neural_net.activations.ReLU,
                                     neural_net.activations.ReLU,
                                     neural_net.activations.Sigmoid),
                        optimizer=optim.SGD(0.01),
                        seed=42
                    ),
                    desired_dist={0: .5, 1: .5},
                    sampling_rate=1
                )
            )

        trained_model_per_predictor.append(model)
        evaluation_metric_per_predictor.append(metrics.Accuracy())

    number_of_regular_editions = 0
    number_of_vandalism_editions = 0

    # Initialize running statistics
    running_statistical_values = {}
    for individual_feature in features:
        # Initial array for the feature. The first position is the current mean and the second, the current N.
        # The third position is the number of true, the fourth position number of false. These are used for the mode.
        running_statistical_values[individual_feature] = [0, 0, 0, 0]

    drift_detector = drift.ADWIN()
    drifts = []
    count = 0

    for input_aux, output_aux in zip(data_input, data_output):
        if count > -1:  # I wanna look now just to the new data. I wanna test how the model behaves.
            input_dict = dict(zip(features, input_aux))  # This is necessary because river works with dictionary
            if output_aux == "reg" or output_aux == 0:
                output_aux = False
                number_of_regular_editions += 1
            else:
                output_aux = True
                number_of_vandalism_editions += 1

            input_dict_no_null, running_statistical_values = UtilDataSetFiles.fill_null(input_dict,
                                                                                        running_statistical_values)

            '''drift_detector.update(output_aux)
            if drift_detector.change_detected:
                # The drift detector indicates after each sample if there is a drift in the data
                print(f'Change detected at index {count}')
                drifts.append(count)
                drift_detector.reset()  # As a best practice, we reset the detector'''

            # just for the regression
            output_aux = int(output_aux)

            for predictor in range(0, ensemble_size):
                estimated_prediction_class = trained_model_per_predictor[predictor].predict_one(input_dict_no_null)

                trained_model_per_predictor[predictor].learn_one(input_dict_no_null, output_aux)

                #evaluation_metric_per_predictor[predictor].update(output_aux, estimated_prediction_class)

        count += 1

    for predictor in range(0, ensemble_size):
        predictor_path = str(pathlib.Path(__file__).parent) + '/predictors/'

        regular_regular = evaluation_metric_per_predictor[predictor].cm.data[False][False]
        regular_vandalism = evaluation_metric_per_predictor[predictor].cm.data[False][True]
        vandalism_vandalism = evaluation_metric_per_predictor[predictor].cm.data[True][True]
        vandalism_regular = evaluation_metric_per_predictor[predictor].cm.data[True][False]

        text_file_path = predictor_path + "metrics_predictor" + str(predictor) + '.txt'
        text_file = open(text_file_path, 'w')

        text_file.write("Overall Accuracy: " + str(evaluation_metric_per_predictor[predictor]) + "\n")

        text_file.write("Label Regular and Classified as Regular: " + str(regular_regular) + "\n")
        text_file.write("Label Regular and Classified as Vandalism: " + str(regular_vandalism) + "\n")
        text_file.write("Label Vandalism and Classified as Vandalism: " + str(vandalism_vandalism) + "\n")
        text_file.write("Label Vandalism and Classified as Regular: " + str(vandalism_regular) + "\n")

        text_file.write("Number of Regular Editions: " + str(number_of_regular_editions) + "\n")
        text_file.write("Number of Vandalism Editions: " + str(number_of_vandalism_editions) + "\n")

        text_file.write("Percentage Corrected Classified Regular: " + str(regular_regular / number_of_regular_editions) + "\n")
        text_file.write("Percentage Corrected Classified Vandalism: " + str(vandalism_vandalism / number_of_vandalism_editions) + "\n")

        text_file.close()

    return trained_model_per_predictor, running_statistical_values


def evaluate_ensemble_with_testing_data(record_trained_models=True, ml_model="rf", number_of_experiments=10,
                                        ensemble_size=12):
    biggest_overall_recall = -1

    for current_experiment in range(0, number_of_experiments):
        dataset_file_path = str(pathlib.Path(__file__).parent) + '/current_testing.csv'
        features, data_input, data_output = get_data_from_csv(dataset_file_path)

        trained_model_per_predictors, running_statistics = train_ensemble(ml_model=ml_model,
                                                                          ensemble_size=ensemble_size)

        number_of_vandalism_editions = 0
        number_of_regular_editions = 0
        regular_regular = 0
        vandalism_vandalism = 0
        regular_vandalism = 0
        vandalism_regular = 0

        for input_aux, output_aux in zip(data_input, data_output):
            input_dict = dict(zip(features, input_aux))

            input_dict_no_null, running_statistical_values = UtilDataSetFiles.fill_null(input_dict, running_statistics)

            editions_vandalism = []
            for model in trained_model_per_predictors:
                data_scaled = model.transform_one(input_dict_no_null)
                editions_vandalism.append(model.predict_one(data_scaled))

            ensemble_result = get_ensemble_result(editions_vandalism)  # True if edition is vandalism

            if output_aux == 1 and ensemble_result:
                number_of_vandalism_editions += 1
                vandalism_vandalism += 1
            elif output_aux == 1 and not ensemble_result:
                number_of_vandalism_editions += 1
                vandalism_regular += 1
            elif output_aux != 1 and not ensemble_result:
                number_of_regular_editions += 1
                regular_regular += 1
            else:
                number_of_regular_editions += 1
                regular_vandalism += 1

        print("Label Regular and Classified as Regular: " + str(regular_regular))
        print("Label Regular and Classified as Vandalism: " + str(regular_vandalism))
        print("Label Vandalism and Classified as Vandalism: " + str(vandalism_vandalism))
        print("Label Vandalism and Classified as Regular: " + str(vandalism_regular))

        print("Number of Regular Editions: " + str(number_of_regular_editions))
        print("Number of Vandalism Editions: " + str(number_of_vandalism_editions))

        corrected_classified_regular = regular_regular / number_of_regular_editions
        corrected_classified_vandalism = vandalism_vandalism / number_of_vandalism_editions
        overall_recall = (corrected_classified_regular + corrected_classified_vandalism) / 2

        print("Percentage Corrected Classified Regular: " + str(corrected_classified_regular))
        print("Percentage Corrected Classified Vandalism: " + str(corrected_classified_vandalism))

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


def evaluate_ensemble_re_label_data(record_trained_models=True, ml_model="sparse_lr", number_of_experiments=10,
                                    ensemble_size=12, training_file_name="/re_labeled.csv",
                                    testing_file_name="/re_label_testing.csv",
                                    training_file_no_longer="",
                                    testing_file_no_longer=""):
    biggest_overall_recall = -1

    for current_experiment in range(0, number_of_experiments):
        dataset_file_path = str(pathlib.Path(__file__).parent) + '/re_training/re_label' + testing_file_name
        features, data_input, data_output = get_data_from_csv_re_label(dataset_file_path)

        trained_model_per_predictors, running_statistics = train_ensemble("/re_training/re_label",
                                                                          training_file_name,
                                                                          True, ml_model=ml_model,
                                                                          ensemble_size=ensemble_size)

        number_of_vandalism_editions = 0
        number_of_regular_editions = 0
        regular_regular = 0
        vandalism_vandalism = 0
        regular_vandalism = 0
        vandalism_regular = 0

        for input_aux, output_aux in zip(data_input, data_output):
            input_dict = dict(zip(features, input_aux))

            input_dict_no_null, running_statistical_values = UtilDataSetFiles.fill_null(input_dict, running_statistics)

            editions_vandalism = []
            for model in trained_model_per_predictors:
                editions_vandalism.append(model.predict_one(input_dict_no_null))

            ensemble_result = get_ensemble_result(editions_vandalism)  # True if edition is vandalism

            if output_aux == 1 and ensemble_result:
                number_of_vandalism_editions += 1
                vandalism_vandalism += 1
            elif output_aux == 1 and not ensemble_result:
                number_of_vandalism_editions += 1
                vandalism_regular += 1
            elif output_aux != 1 and not ensemble_result:
                number_of_regular_editions += 1
                regular_regular += 1
            else:
                number_of_regular_editions += 1
                regular_vandalism += 1

        print("Label Regular and Classified as Regular: " + str(regular_regular))
        print("Label Regular and Classified as Vandalism: " + str(regular_vandalism))
        print("Label Vandalism and Classified as Vandalism: " + str(vandalism_vandalism))
        print("Label Vandalism and Classified as Regular: " + str(vandalism_regular))

        print("Number of Regular Editions: " + str(number_of_regular_editions))
        print("Number of Vandalism Editions: " + str(number_of_vandalism_editions))

        corrected_classified_regular = regular_regular / number_of_regular_editions
        corrected_classified_vandalism = vandalism_vandalism / number_of_vandalism_editions
        overall_recall = (corrected_classified_regular + corrected_classified_vandalism) / 2

        print("Percentage Corrected Classified Regular: " + str(corrected_classified_regular))
        print("Percentage Corrected Classified Vandalism: " + str(corrected_classified_vandalism))

        if record_trained_models and overall_recall > biggest_overall_recall:
            biggest_overall_recall = overall_recall
            if ml_model == "nn":
                save_trained_models(trained_model_per_predictors, "model",
                                    "/experiments/article_datasets/re_training/re_label/trained_nn_models/")
            else:
                save_trained_models(trained_model_per_predictors, "model",
                                    "/experiments/article_datasets/re_training/re_label/trained_rf_models/")
            print("Biggest Overall Recall \n \n \n")

        training_no_longer_file_path = str(pathlib.Path(__file__).parent) + '/re_training/re_label' + training_file_no_longer
        features_no_longer, data_input_no_longer, data_output_no_longer = get_data_from_csv_re_label(training_no_longer_file_path)
        number_of_regular_editions = 0
        regular_regular = 0
        regular_vandalism = 0
        for input_aux, output_aux in zip(data_input_no_longer, data_output_no_longer):
            input_dict = dict(zip(features_no_longer, input_aux))

            input_dict_no_null, running_statistical_values = UtilDataSetFiles.fill_null(input_dict, running_statistics)

            editions_vandalism = []
            for model in trained_model_per_predictors:
                editions_vandalism.append(model.predict_one(input_dict_no_null))

            ensemble_result = get_ensemble_result(editions_vandalism)  # True if edition is vandalism

            if not ensemble_result:
                number_of_regular_editions += 1
                regular_regular += 1
            else:
                number_of_regular_editions += 1
                regular_vandalism += 1

        print("NO LONGER VANDALISM TRAINING")
        print("Label Regular and Classified as Regular: " + str(regular_regular))
        print("Label Regular and Classified as Vandalism: " + str(regular_vandalism))

        print("Number of Regular Editions: " + str(number_of_regular_editions))

        corrected_classified_regular = regular_regular / number_of_regular_editions

        print("Percentage Corrected Classified Regular: " + str(corrected_classified_regular))

        testing_no_longer_file_path = str(
            pathlib.Path(__file__).parent) + '/re_training/re_label' + testing_file_no_longer
        features_test_no_longer, data_input_test_no_longer, data_output_test_no_longer = get_data_from_csv_re_label(
            testing_no_longer_file_path)
        number_of_regular_editions = 0
        regular_regular = 0
        regular_vandalism = 0
        for input_aux, output_aux in zip(data_input_test_no_longer, data_output_test_no_longer):
            input_dict = dict(zip(features_test_no_longer, input_aux))

            input_dict_no_null, running_statistical_values = UtilDataSetFiles.fill_null(input_dict, running_statistics)

            editions_vandalism = []
            for model in trained_model_per_predictors:
                editions_vandalism.append(model.predict_one(input_dict_no_null))

            ensemble_result = get_ensemble_result(editions_vandalism)  # True if edition is vandalism

            if not ensemble_result:
                number_of_regular_editions += 1
                regular_regular += 1
            else:
                number_of_regular_editions += 1
                regular_vandalism += 1

        print("NO LONGER VANDALISM TESTING")
        print("Label Regular and Classified as Regular: " + str(regular_regular))
        print("Label Regular and Classified as Vandalism: " + str(regular_vandalism))

        print("Number of Regular Editions: " + str(number_of_regular_editions))

        corrected_classified_regular = regular_regular / number_of_regular_editions

        print("Percentage Corrected Classified Regular: " + str(corrected_classified_regular))


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

    features = data_set.columns.values
    data_input = data_set.iloc[:, :-1].values  # It gets all rows and columns, except the last column.
    data_output = data_set.iloc[:, number_columns-1]  # It gets the column with the label.

    return features, data_input, data_output


def get_data_from_csv_re_label(dataset_file_path):
    data_set = pd.read_csv(dataset_file_path, na_values="?")
    number_columns = len(data_set.columns)

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


def create_re_label_dataset(directory_to_save_datasets, load_dataset_from_disk=False):

    complete_dataset_standard = pd.DataFrame(np.empty((0, 59)))
    complete_dataset_clean = pd.DataFrame(np.empty((0, 59)))
    testing_dataset_clean = pd.DataFrame(np.empty((0, 58)))
    testing_dataset_standard = pd.DataFrame(np.empty((0, 58)))
    all_balanced_datasets = []

    wtdelayed_hashreverted_repcountry = "WT_DELAYED > 0.40, HASH_REVERTED=1, HIST_REP_COUNTRY > 0.10"  # 94 Editions
    wtdelayed_usereditsmonth_hashreverted = "WT_DELAYED > 0.40, USER_EDITS_MONTH <= 4.00, HASH_REVERTED=1"  # 76 Editions
    wtdelayed_usereditsmonth_usereditstotal = "WT_DELAYED > 0.40, USER_EDITS_MONTH <= 4.00, USER_EDITS_TOTAL <= 7.00"  # 536 Editions
    userseditsmonth_usereditstotal_wtdelayed = "USER_EDITS_MONTH <= 4.00, USER_EDITS_TOTAL <= 7.00, WT_DELAYED <= 0.22"  # 48 Editions

    relevant_features_by_lime = [wtdelayed_usereditsmonth_usereditstotal]

    if load_dataset_from_disk:
        complete_dataset_clean = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/re_training/re_label/",
                                                                         "re_labeled.csv")

        complete_dataset_standard = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/re_training/re_label/",
                                                                            "re_labeled_standard.csv")

        testing_dataset_clean = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/re_training/re_label/",
                                                                        "testing.csv")

        testing_dataset_standard = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/re_training/re_label/",
                                                                           "testing_standard.csv")

        all_balanced_datasets = UtilDataSetFiles.get_data_set_balanced("/experiments/article_datasets/re_training/re_label/balanced/",
                                                                       "balanced_dataset", 12)

    else:
        initial_train_dataset = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/",
                                                                        "current_training.csv")

        re_train_dataset = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/",
                                                                   "current_testing.csv")

        re_train_dataset_relevant_features = UtilDataSetFiles.get_data_set_from_file("/experiments/article_datasets/",
                                                                                     "current_testing_relevant_features.csv",
                                                                                     null_value='nan')
        # current_testing.csv and current_testing_relevant_features have the same data, the difference is that
        # current_testing_relevant_features.csv contain a column with the relevant features found by lime.

        re_train_dataset = re_train_dataset.join(re_train_dataset_relevant_features["RELEVANT_FEATURES"])

        # From the new dataset (re_training), I want to separate further to get test data.
        x_re_train, x_test, y_re_train, y_test = train_test_split(re_train_dataset.drop('LABEL', axis=1),
                                                                  re_train_dataset['LABEL'],
                                                                  test_size=0.2,
                                                                  random_state=42)

        relevant_features_re_train = x_re_train['RELEVANT_FEATURES']
        x_re_train = x_re_train.drop('RELEVANT_FEATURES', axis=1)

        relevant_features_re_train_test = x_test['RELEVANT_FEATURES']
        x_test = x_test.drop('RELEVANT_FEATURES', axis=1)

        x_re_train["LABEL"] = y_re_train
        x_re_train["RELEVANT_FEATURES"] = relevant_features_re_train

        initial_train_dataset["RELEVANT_FEATURES"] = ""

        datasets_to_merge = [initial_train_dataset, x_re_train]
        new_complete_dataset = pd.concat(datasets_to_merge).reset_index(drop=True)

        # Adding the column label to the testing dataset that is going to be saved as a file.
        testing_dataset = x_test
        testing_dataset["LABEL"] = y_test
        testing_dataset["RELEVANT_FEATURES"] = relevant_features_re_train_test

        all_new_editions_vandalism = new_complete_dataset[new_complete_dataset["RELEVANT_FEATURES"].isin(relevant_features_by_lime)]

        all_new_editions_vandalism_test = testing_dataset[testing_dataset["RELEVANT_FEATURES"].isin(relevant_features_by_lime)]

        # Now I calculate the distance between the base edition and all the new editions that are vandalism.
        ids_no_longer_vandalism = []
        dt_no_longer_vandalism = pd.DataFrame(columns=new_complete_dataset.columns)
        for index, edition_to_compare in all_new_editions_vandalism.iterrows():
            new_complete_dataset.loc[index, "LABEL"] = 0

            dt_aux = pd.DataFrame(columns=dt_no_longer_vandalism.columns, data=[new_complete_dataset.loc[index]])
            dt_no_longer_vandalism = pd.concat([dt_no_longer_vandalism, dt_aux])
            ids_no_longer_vandalism.append(index)

        ids_no_longer_vandalism_test = []
        dt_no_longer_vandalism_test = pd.DataFrame(columns=testing_dataset.columns)
        for index, edition_test_to_compare in all_new_editions_vandalism_test.iterrows():
            testing_dataset.loc[index, "LABEL"] = 0

            dt_aux = pd.DataFrame(columns=dt_no_longer_vandalism_test.columns, data=[testing_dataset.loc[index]])
            dt_no_longer_vandalism_test = pd.concat([dt_no_longer_vandalism_test, dt_aux])
            ids_no_longer_vandalism_test.append(index)

        directory_to_save_datasets += "re_label/"

        # Save the data that was relabeled. This is the training data.
        UtilDataSetFiles.save_data_set(new_complete_dataset, directory_to_save_datasets, "re_labeled.csv")
        UtilDataSetFiles.save_data_set(dt_no_longer_vandalism, directory_to_save_datasets, "re_label_no_longer_vandalism.csv")
        dt_ids_no_longer_vandalism = pd.DataFrame(ids_no_longer_vandalism)
        UtilDataSetFiles.save_data_set(dt_ids_no_longer_vandalism, directory_to_save_datasets, "ids_no_longer_vandalism.csv")

        # Save the data that was relabeled. This is the testing data.
        UtilDataSetFiles.save_data_set(testing_dataset, directory_to_save_datasets, "re_label_testing.csv")
        UtilDataSetFiles.save_data_set(dt_no_longer_vandalism_test, directory_to_save_datasets, "re_label_no_longer_vandalism_test.csv")
        dt_ids_no_longer_vandalism_test = pd.DataFrame(ids_no_longer_vandalism_test)
        UtilDataSetFiles.save_data_set(dt_ids_no_longer_vandalism_test, directory_to_save_datasets, "ids_no_longer_vandalism_test.csv")