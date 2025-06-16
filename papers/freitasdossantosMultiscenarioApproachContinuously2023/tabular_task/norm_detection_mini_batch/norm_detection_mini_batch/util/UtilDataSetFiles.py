import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data_set_from_file(base_file_path, file_name, null_value="?"):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    data_set = pd.read_csv(file_path, na_values=null_value)

    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    return data_set


def handle_imbalanced_dataset(data_set, new_file_base_path, new_file_name, save_to_file=False,
                              balanced_folder="/balanced_data/"):
    data_set_regular_editions = data_set[data_set["LABEL"] == 0]
    data_set_vandalism_editions = data_set[data_set["LABEL"] == 1]

    number_of_regular_editions = len(data_set_regular_editions)
    number_of_vandalism_editions = len(data_set_vandalism_editions)

    number_of_groups = np.math.floor(number_of_regular_editions / number_of_vandalism_editions)

    regular_editions_by_group = np.array_split(data_set_regular_editions, number_of_groups)

    all_balanced_datasets = []
    all_columns_statistical_values = []
    for group_index in range(0, number_of_groups):
        if len(regular_editions_by_group[group_index]) > 0:
            regular_editions_to_concat = pd.DataFrame(regular_editions_by_group[group_index],
                                                      columns=data_set_vandalism_editions.columns)
            concatenated_data_sets = [regular_editions_to_concat, data_set_vandalism_editions]

            # This lines concatenates both dataframe and also shuffle them.
            balanced_data_set = pd.concat(concatenated_data_sets).sample(frac=1)

            all_balanced_datasets.append(balanced_data_set)

            # It's empty because we are not considering the statistical values individually by balanced dataset
            columns_statistical_values = pd.DataFrame()
            all_columns_statistical_values.append(columns_statistical_values)

            if save_to_file:
                base_file_path = str(pathlib.Path(__file__).parent.parent.parent)

                path_to_save_balanced_dataset = base_file_path + new_file_base_path + balanced_folder + \
                                                new_file_name + str(group_index) + '.csv'

                path_to_save_statistical_values = base_file_path + new_file_base_path + balanced_folder + new_file_name \
                                                  + str(group_index) + '_statistical_values.csv'

                pd.DataFrame.to_csv(balanced_data_set,
                                    path_to_save_balanced_dataset,
                                    index=False)
                pd.DataFrame.to_csv(columns_statistical_values,
                                    path_to_save_statistical_values,
                                    index=False)

    return all_balanced_datasets, all_columns_statistical_values


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


def update_missing_values_media_mode(dataset, columns_statistical_values):
    for column in columns_statistical_values.index.tolist():
        dataset.loc[dataset[column].isnull(), [column]] = columns_statistical_values[column]

    dataset = dataset.replace(False, 0)
    dataset = dataset.replace("False", 0)
    dataset = dataset.replace(True, 1)
    dataset = dataset.replace("True", 1)
    dataset = dataset.replace("reg", 0)
    dataset = dataset.replace("vand", 1)

    return dataset


def save_data_set(data_set, base_file_path, file_name):
    path_to_save = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name

    pd.DataFrame.to_csv(data_set,
                        path_to_save,
                        index=False)


def get_data_set_balanced(base_file_path, file_name, number_of_files, null_value="?"):
    all_balanced_datasets = []

    for i in range(0, number_of_files):
        file_name_temp = file_name + str(i) + ".csv"
        balanced_temp = get_data_set_from_file(base_file_path, file_name_temp, null_value)

        all_balanced_datasets.append(balanced_temp)

    return all_balanced_datasets


def apply_standardization(data_set, columns_not_altered=[57]):
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