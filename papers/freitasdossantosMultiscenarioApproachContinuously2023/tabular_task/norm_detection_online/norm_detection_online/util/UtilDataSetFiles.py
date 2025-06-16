import math
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CATEGORICAL_FEATURES = ["USER_IS_IP", "USER_IS_BOT", "USER_BLKED_BEFORE", "USER_BLKED_EVER", "COMM_HAS_SECT",
                        "COMM_VAND", "HIST_USER_HAS_RB", "HASH_REVERTED", "PREV_USER_IS_IP", "PREV_USER_SAME",
                        "NEXT_USER_IS_IP", "NEXT_USER_SAME", "NEXT_COMMENT_VAND"]


def get_data_set_from_file(base_file_path, file_name, null_value="?"):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    data_set = pd.read_csv(file_path, na_values=null_value)

    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    return data_set


def save_data_set(data_set, base_file_path, file_name):
    path_to_save = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name

    pd.DataFrame.to_csv(data_set,
                        path_to_save,
                        index=False)


'''def calculate_running_statistics(feature, feature_running_statistics, new_value):
    if feature in CATEGORICAL_FEATURES:
        current_number_of_true_values = feature_running_statistics[2]
        current_number_of_false_values = feature_running_statistics[3]

        if new_value:
            new_number_of_true_values = current_number_of_true_values + 1  # In position 2 is the number of true values
            feature_running_statistics[2] = new_number_of_true_values
            if new_number_of_true_values > current_number_of_false_values:
                feature_running_statistics[0] = 1  # In position 0 is the mode
        else:
            new_number_of_false_values = current_number_of_false_values + 1  # In position 3 is the number of false values
            feature_running_statistics[3] = new_number_of_false_values
            if new_number_of_false_values > current_number_of_true_values:
                feature_running_statistics[0] = 0
    else:
        current_mean = feature_running_statistics[0]  # In position 0 is the current mean
        current_n = feature_running_statistics[1] + 1  # In position 1 is number of data points for that feature (N).
        current_sum_of_squares = feature_running_statistics[4]

        new_mean = current_mean + (new_value - current_mean) / current_n
        new_sum_of_squares = current_sum_of_squares + (new_value - current_mean) * (new_value - new_mean)
        new_standard_deviation = math.sqrt(new_sum_of_squares / current_n)

        feature_running_statistics[0] = new_mean
        feature_running_statistics[1] = current_n
        feature_running_statistics[4] = new_sum_of_squares
        feature_running_statistics[5] = new_standard_deviation

    return feature_running_statistics
'''


def calculate_running_statistics(feature_running_statistics, new_value, is_categorical_feature):
    if is_categorical_feature:
        current_number_of_true_values = feature_running_statistics[2]
        current_number_of_false_values = feature_running_statistics[3]

        if new_value:
            new_number_of_true_values = current_number_of_true_values + 1  # In position 2 is the number of true values
            feature_running_statistics[2] = new_number_of_true_values
            if new_number_of_true_values > current_number_of_false_values:
                feature_running_statistics[0] = 1  # In position 0 is the mode
        else:
            new_number_of_false_values = current_number_of_false_values + 1  # In position 3 is the number of false values
            feature_running_statistics[3] = new_number_of_false_values
            if new_number_of_false_values > current_number_of_true_values:
                feature_running_statistics[0] = 0
    else:
        current_mean = feature_running_statistics[0]  # In position 0 is the current mean
        current_n = feature_running_statistics[1] + 1  # In position 1 is number of data points for that feature (N).
        current_sum_of_squares = feature_running_statistics[4]

        new_mean = current_mean + (new_value - current_mean) / current_n
        new_sum_of_squares = current_sum_of_squares + (new_value - current_mean) * (new_value - new_mean)
        new_standard_deviation = math.sqrt(new_sum_of_squares / current_n)

        feature_running_statistics[0] = new_mean
        feature_running_statistics[1] = current_n
        feature_running_statistics[4] = new_sum_of_squares
        feature_running_statistics[5] = new_standard_deviation

    return feature_running_statistics


def fill_null(edition, running_statistical_values):
    features = edition.index

    for feature_aux in features:
        value_aux = edition[feature_aux]

        '''if isinstance(value_aux, str):
            value_aux = float(value_aux)
            edition[feature_aux] = value_aux'''

        if math.isnan(value_aux):
            # In position 0 is the running mean or the running mode
            edition[feature_aux] = running_statistical_values[feature_aux][0]
        else:
            #value_aux = int(value_aux)
            new_statistical_value = calculate_running_statistics(feature_aux, running_statistical_values[feature_aux],
                                                                 value_aux)
            running_statistical_values[feature_aux] = new_statistical_value

    return edition, running_statistical_values


def apply_pre_process(edition, running_statistical_values):
    features = edition.index
    edition_standardized = edition.copy()

    for feature_aux in features:
        value_aux = edition[feature_aux]
        is_categorical_feature = True if feature_aux in CATEGORICAL_FEATURES else False

        if math.isnan(value_aux):
            # In position 0 is the running mean or the running mode
            edition[feature_aux] = edition_standardized[feature_aux] = running_statistical_values[feature_aux][0]
        else:
            new_statistical_value = calculate_running_statistics(running_statistical_values[feature_aux], value_aux,
                                                                 is_categorical_feature)
            running_statistical_values[feature_aux] = new_statistical_value

        if not is_categorical_feature:
            edition_standardized[feature_aux] = apply_standardization_single(edition_standardized[feature_aux],
                                                                             feature_aux, running_statistical_values)

    return edition, edition_standardized, running_statistical_values


def fill_null_no_update(data, running_statistical_values):
    for feature_aux in data:
        value_aux = data[feature_aux]

        if isinstance(value_aux, str):
            value_aux = float(value_aux)
            data[feature_aux] = value_aux

        if math.isnan(value_aux):
            running_mean = running_statistical_values[feature_aux][0]  # In position 0 is the running mean or the running mode
            data[feature_aux] = running_mean

    return data


def apply_standardization(edition, running_statistical_values):
    features = edition.index

    for feature_aux in features:
        if feature_aux not in CATEGORICAL_FEATURES:  # We don't apply standardization to categorical data
            value_aux = edition[feature_aux]

            # value_standardized = (x – mean) / standard_deviation
            standard_deviation = running_statistical_values[feature_aux][5]
            if standard_deviation != 0:
                value_standardized = (value_aux - running_statistical_values[feature_aux][0]) / standard_deviation
            else:
                value_standardized = value_aux

            edition[feature_aux] = value_standardized

    return edition


def apply_standardization_single(value, feature, running_statistical_values):
    standard_deviation = running_statistical_values[feature][5]
    if standard_deviation != 0:
        value_standardized = (value - running_statistical_values[feature][0]) / standard_deviation
    else:
        value_standardized = value

    return value_standardized


def update_missing_values_media_mode(dataset, columns_statistical_values):
    for column in columns_statistical_values.keys():
        dataset.loc[dataset[column].isnull(), column] = columns_statistical_values[column][0]

    dataset = dataset.replace(False, 0)
    dataset = dataset.replace("False", 0)
    dataset = dataset.replace(True, 1)
    dataset = dataset.replace("True", 1)
    dataset = dataset.replace("reg", 0)
    dataset = dataset.replace("vand", 1)

    return dataset


def update_missing_values_media_mode_dict(dataset, columns_statistical_values):
    for column in columns_statistical_values.keys():
        dataset.loc[dataset[column].isnull(), [column]] = columns_statistical_values[column][0]

    dataset = dataset.replace(False, 0)
    dataset = dataset.replace("False", 0)
    dataset = dataset.replace(True, 1)
    dataset = dataset.replace("True", 1)
    dataset = dataset.replace("reg", 0)
    dataset = dataset.replace("vand", 1)

    return dataset


def handle_missing_values(data_set, columns_not_altered=[57]):
    columns_to_handle = np.delete(data_set.columns.values, columns_not_altered)

    non_numeric_columns = data_set[columns_to_handle].dtypes[data_set.dtypes == 'bool'].index.values.tolist()
    non_numeric_columns += data_set[columns_to_handle].dtypes[data_set.dtypes == 'object'].index.values.tolist()

    columns_statistical_values = pd.DataFrame(columns=data_set.columns)
    columns_statistical_values = data_set.mean()
    for column in non_numeric_columns:
        column_mode = data_set[column].mode().values[0]
        columns_statistical_values[column] = column_mode
        data_set[column].fillna(column_mode, inplace=True)

    # For the columns that contain "True" or "False", I'm changing by the mode of the column.
    # data_set[non_numeric_columns].fillna(columns_statistical_values[non_numeric_columns], inplace=True)

    # For the numeric columns, I'm changing to the mean.
    data_set.fillna(data_set.mean(), inplace=True)

    data_set = data_set.replace(False, 0)
    data_set = data_set.replace("False", False)
    data_set = data_set.replace(True, 1)
    data_set = data_set.replace("True", True)
    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    return data_set, columns_statistical_values[columns_to_handle]


def apply_standardization_in_dataset(data_set, columns_not_altered=[57]):
    standard_scaling = StandardScaler()

    #name_columns_to_scale = data_set.columns.values[:-1]
    #labels_columns_not_altered = data_set.columns.values[columns_not_altered]
    #name_columns_to_scale = data_set.drop(labels_columns_not_altered, axis=1).columns.values[:]

    name_columns_to_scale = np.delete(data_set.columns.values, columns_not_altered)

    # Don't apply standardization in the label.
    standard_data_set = standard_scaling.fit_transform(data_set[name_columns_to_scale])

    data_set[name_columns_to_scale] = standard_data_set

    return data_set, standard_scaling


def apply_standardization_defined(data_set, standard_scaling, columns_not_altered=[57]):
    name_columns_to_scale = np.delete(data_set.columns.values, columns_not_altered)

    # Don't apply standardization in the label. Don't fit when the standard_scaling is already defined
    standard_data_set = standard_scaling.transform(data_set[name_columns_to_scale])

    data_set[name_columns_to_scale] = standard_data_set

    return data_set


def split_training_test_k_validation(dataset_folders, dataset_columns, base_file_path, training_file_name,
                                     testing_file_name, save_to_file=True, number_of_k_folders=10):
    # TODO: Save to file all these data to make it easier to get it later for training and testing.

    training_folders = []
    testing_folders = []

    current_folder_index = 0
    for current_testing_folder in dataset_folders:
        testing_folders.append(current_testing_folder)

        current_training_dataset = pd.DataFrame(columns=dataset_columns)
        for aux_training_index in range(0, number_of_k_folders):
            if aux_training_index != current_folder_index:
                current_training_dataset = pd.concat([current_training_dataset, dataset_folders[aux_training_index]])

        training_folders.append(current_training_dataset)
        current_folder_index += 1

    return training_folders, testing_folders


def save_pickle_objects(data, path):
    with open(path, "wb") as pickled_file:
        pickle.dump(data, pickled_file)


def load_pickle_objects(path):
    with open(path, 'rb') as f:
        unpickled_object = pickle.load(f)
        return unpickled_object


def get_data_set_balanced(base_file_path, file_name, number_of_files, null_value="?"):
    all_balanced_datasets = []

    for i in range(0, number_of_files):
        file_name_temp = file_name + str(i) + ".csv"
        balanced_temp = get_data_set_from_file(base_file_path, file_name_temp, null_value)

        all_balanced_datasets.append(balanced_temp)

    return all_balanced_datasets
