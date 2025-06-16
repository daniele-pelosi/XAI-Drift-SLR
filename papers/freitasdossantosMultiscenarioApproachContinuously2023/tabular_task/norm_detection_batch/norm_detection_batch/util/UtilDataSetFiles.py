import pathlib
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_dataset_from_balanced_datasets(base_file_path, base_file_name, number_of_datasets, file_mode_to_create,
                                          vand_minority_class, contain_testing_dataset):
    """
    Method responsible for building a dataset in the mode the file_mode_to_create defines.
    :param base_file_path: The file structure that defines the path of the datasets.
    :param base_file_name: The base name of the file.
    :param number_of_datasets: The number of datasets we want to get information from.
    :param file_mode_to_create: The different types of file we want to create. 3 options: 1) dataset will all regular
    editions; 2) dataset with all vandalism edition; and 3) complete dataset with regular and vandalism edition.
    :param vand_minority_class: True or False. True if the minority class is Vandalism,
                                False if the minority class is regular.
    :param contain_testing_dataset: True or False. True if the user wants to concat with a testing dataset.
    :return:
    """

    dataset_file_name = "regular_dataset"
    if file_mode_to_create == 2:
        dataset_file_name = "vandalism_dataset"
    elif file_mode_to_create == 3:
        dataset_file_name = "complete_dataset"

    new_dataset_file_path = str(pathlib.Path(__file__).parent.parent) + "/" + dataset_file_name + ".csv"
    datasets_to_extract_file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/" + base_file_path

    new_dataset = pd.DataFrame(np.empty((0, 58)))

    for index_dataset in range(number_of_datasets):
        dataset_file_path = datasets_to_extract_file_path + str(index_dataset) + "/" + base_file_name \
                            + str(index_dataset) + ".csv"
        dataset = pd.read_csv(dataset_file_path)

        if index_dataset == 0:
            new_dataset.columns = list(dataset.columns.values)

        if file_mode_to_create == 1:
            new_dataset = pd.concat([new_dataset, dataset[dataset["LABEL"] == 0]])
        elif file_mode_to_create == 2:
            new_dataset = pd.concat([new_dataset, dataset[dataset["LABEL"] == 1]])
        else:
            if index_dataset == 0:
                new_dataset = dataset
            else:
                if vand_minority_class:
                    # Vandalism Minority, then I will just get the regular data from the other datasets
                    # because getting the vandalism data would be redundant. The other way in the case bellow
                    new_dataset = pd.concat([new_dataset, dataset[dataset["LABEL"] == 0]])
                else:
                    new_dataset = pd.concat([new_dataset, dataset[dataset["LABEL"] == 1]])

    if contain_testing_dataset:
        testing_dataset_file_path = datasets_to_extract_file_path + "testing_dataset.csv"
        testing_dataset = pd.read_csv(testing_dataset_file_path)

        if file_mode_to_create == 1:
            new_dataset = pd.concat([new_dataset, testing_dataset[testing_dataset["LABEL"] == 0]])
        elif file_mode_to_create == 2:
            new_dataset = pd.concat([new_dataset, testing_dataset[testing_dataset["LABEL"] == 1]])
        else:
            new_dataset = pd.concat([new_dataset, testing_dataset])

    new_dataset.sample(frac=1)
    pd.DataFrame.to_csv(new_dataset, new_dataset_file_path, index=False)


def create_re_training_dataset(base_file_path, new_base_file_path, base_file_name, number_of_datasets,
                               contain_testing_dataset, change_label_to, base_file_path_changed_dataset):
    datasets_to_extract_file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/" + base_file_path
    new_dataset_file_path = str(pathlib.Path(__file__).parent.parent.parent) + "/" + new_base_file_path

    changed_dataset_file_name = ""
    if change_label_to == 0:
        changed_dataset_file_name = "no_longer_vandalism.csv"
    elif change_label_to == 1:
        changed_dataset_file_name = "no_longer_regular.csv"

    changed_dataset_file = str(pathlib.Path(__file__).parent.parent.parent) + "/" + base_file_path_changed_dataset \
                           + changed_dataset_file_name
    changed_dataset = pd.read_csv(changed_dataset_file)

    for index_dataset in range(0, number_of_datasets):
        old_dataset_file = datasets_to_extract_file_path + str(index_dataset) + "/" + base_file_name \
                           + str(index_dataset) + ".csv"
        new_dataset_file = new_dataset_file_path + str(index_dataset) + "/" + base_file_name + str(index_dataset) \
                           + ".csv"

        old_dataset = pd.read_csv(old_dataset_file)

        # Here we take out the last columns because we don't want to compare the label (that will be changed for retraining).
        old_dataset.loc[old_dataset.set_index(list(old_dataset.columns.values[:-1])).index
                            .isin(
            changed_dataset.set_index(list(changed_dataset.columns.values[:-1])).index), "LABEL"] = change_label_to

        pd.DataFrame.to_csv(old_dataset, new_dataset_file, index=False)

    if contain_testing_dataset:
        old_testing_dataset_file = datasets_to_extract_file_path + "testing_dataset.csv"
        old_testing_dataset = pd.read_csv(old_testing_dataset_file)

        old_testing_dataset.loc[old_testing_dataset.set_index(list(old_testing_dataset.columns.values[:-1])).index
                                    .isin(
            changed_dataset.set_index(list(changed_dataset.columns.values[:-1])).index), "LABEL"] \
            = change_label_to

        new_testing_dataset_file = new_dataset_file_path + "new_testing_dataset.csv"
        pd.DataFrame.to_csv(old_testing_dataset, new_testing_dataset_file, index=False)


def split_training_test_dataset(complete_data_set, base_file_path, training_file_name, testing_file_name,
                                save_to_file=True, percentage_of_split=0.66):
    data_set_vandalism = complete_data_set[complete_data_set["LABEL"] == 1]
    data_set_regular = complete_data_set[complete_data_set["LABEL"] == 0]

    index_percent_vandalism = int(len(data_set_vandalism) * percentage_of_split)
    training_data_set_vandalism = data_set_vandalism[0:index_percent_vandalism]
    testing_data_set_vandalism = data_set_vandalism[index_percent_vandalism:len(data_set_vandalism)]

    index_percent_regular = int(len(data_set_regular) * percentage_of_split)
    training_data_set_regular = data_set_regular[0:index_percent_regular]
    testing_data_set_regular = data_set_regular[index_percent_regular:len(data_set_regular)]

    data_set_training_concat = [training_data_set_regular, training_data_set_vandalism]
    training_data_set = pd.concat(data_set_training_concat).sample(frac=1)

    data_set_testing_concat = [testing_data_set_regular, testing_data_set_vandalism]
    testing_data_set = pd.concat(data_set_testing_concat).sample(frac=1)

    if save_to_file:
        directory_structure = str(pathlib.Path(__file__).parent.parent.parent)

        path_to_save_training_data_set = directory_structure + base_file_path + training_file_name
        path_to_save_testing_data_set = directory_structure + base_file_path + testing_file_name

        pd.DataFrame.to_csv(training_data_set,
                            path_to_save_training_data_set,
                            index=False)

        pd.DataFrame.to_csv(testing_data_set,
                            path_to_save_testing_data_set,
                            index=False)

    return training_data_set, testing_data_set


def split_dataset_current_future(complete_dataset, base_file_path, current_file_name="current_dataset.csv",
                                 future_file_name="future_dataset.csv", save_to_file=True, percentage_of_split=0.5):

    dataset_vandalism = complete_dataset[complete_dataset["LABEL"] == 1]
    dataset_regular = complete_dataset[complete_dataset["LABEL"] == 0]

    index_percent_vandalism = int(len(dataset_vandalism) * percentage_of_split)
    current_dataset_vandalism = dataset_vandalism[0:index_percent_vandalism]
    future_dataset_vandalism = dataset_vandalism[index_percent_vandalism:len(dataset_vandalism)]

    index_percent_regular = int(len(dataset_regular) * percentage_of_split)
    current_dataset_regular = dataset_regular[0:index_percent_regular]
    future_dataset_regular = dataset_regular[index_percent_regular:len(dataset_regular)]

    current_dataset_concat = [current_dataset_regular, current_dataset_vandalism]
    current_dataset = pd.concat(current_dataset_concat).sample(frac=1)

    future_dataset_concat = [future_dataset_regular, future_dataset_vandalism]
    future_dataset = pd.concat(future_dataset_concat).sample(frac=1)

    if save_to_file:
        directory_structure = str(pathlib.Path(__file__).parent.parent.parent)

        path_to_save_current_dataset = directory_structure + base_file_path + current_file_name
        path_to_save_future_dataset = directory_structure + base_file_path + future_file_name

        pd.DataFrame.to_csv(current_dataset,
                            path_to_save_current_dataset,
                            index=False)

        pd.DataFrame.to_csv(future_dataset,
                            path_to_save_future_dataset,
                            index=False)

    return current_dataset, future_dataset


def split_training_test_dataset_with_file(base_file_path, file_name, training_file_name, testing_file_name,
                                          save_to_file=True):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    data_set = pd.read_csv(file_path, na_values="?")

    testing_data_set, training_data_set = split_training_test_dataset(data_set, base_file_path, training_file_name,
                                                                      testing_file_name, save_to_file)

    return testing_data_set, training_data_set


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
    data_set = data_set.replace("False", 0)
    data_set = data_set.replace(True, 1)
    data_set = data_set.replace("True", 1)
    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    return data_set, columns_statistical_values[columns_to_handle]


def handle_missing_values_with_file(base_file_path, new_base_file_path, file_name, new_file_name, save_to_file=False):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    new_file_path_to_save = str(pathlib.Path(__file__).parent.parent.parent) + new_base_file_path + new_file_name
    data_set = pd.read_csv(file_path, na_values="?")

    data_set, columns_statistical_values = handle_missing_values(data_set)

    if save_to_file:
        pd.DataFrame.to_csv(data_set, new_file_path_to_save, index=False)

    return data_set, columns_statistical_values


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


def handle_imbalanced_dataset(data_set, new_file_base_path, new_file_name, save_to_file=False,
                              balanced_folder="/balanced_data/", initial_number_of_classifiers=12):
    data_set_regular_editions = data_set[data_set["LABEL"] == 0]
    data_set_vandalism_editions = data_set[data_set["LABEL"] == 1]

    number_of_regular_editions = len(data_set_regular_editions)
    number_of_vandalism_editions = len(data_set_vandalism_editions)

    number_of_groups = np.math.ceil(number_of_regular_editions / number_of_vandalism_editions)

    if number_of_groups > initial_number_of_classifiers:
        data_set_vandalism_editions = oversample_minority_class(initial_number_of_classifiers,
                                                                number_of_regular_editions,
                                                                data_set_vandalism_editions)
        number_of_groups = initial_number_of_classifiers

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

            # Removed this code because we don't want to handle the missing values per group of balanced (this is done
            # in the whole training dataset.
            #balanced_data_set, columns_statistical_values = handle_missing_values(balanced_data_set)

            all_balanced_datasets.append(balanced_data_set)

            # It's empty because we are not considering the statistical values individually by balanced dataset
            columns_statistical_values = pd.DataFrame()
            all_columns_statistical_values.append(columns_statistical_values)
            #all_columns_statistical_values.append(columns_statistical_values)

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


def oversample_minority_class(number_of_classifiers, number_of_majority_editions, dataset_minority_editions):
    number_minority_editions = len(dataset_minority_editions)
    desired_number_of_minority = np.math.ceil(number_of_majority_editions / number_of_classifiers)
    number_of_minority_to_oversample = desired_number_of_minority - number_minority_editions

    updated_weight = dataset_minority_editions["WEIGHT"].max()
    dataset_minority_updated_weight = dataset_minority_editions[dataset_minority_editions["WEIGHT"] == updated_weight]
    number_minority_updated_weight = len(dataset_minority_updated_weight)

    if number_of_minority_to_oversample > number_minority_updated_weight:
        editions_oversampled = dataset_minority_updated_weight.sample(n=number_of_minority_to_oversample, replace=True)
    else:
        editions_oversampled = dataset_minority_updated_weight.sample(n=number_of_minority_to_oversample)

    dataset_minority_editions = pd.concat([dataset_minority_editions, editions_oversampled])
    return dataset_minority_editions


def handle_imbalanced_dataset_re_label(data_set, re_labeled_data, re_labeled_indexes, new_file_base_path, new_file_name,
                                       save_to_file=False, balanced_folder="/balanced_data/"):
    '''
    This function should not be used to build the dataset that is going to be used to build the ML model.
    It adds a bias to the balanced dataset.
    '''

    data_set_regular_editions = data_set.loc[data_set["LABEL"] == 0].copy()
    data_set_regular_editions.drop(re_labeled_indexes, inplace=True)

    # Contains all the data that will be present in all balanced dataset. This is necessary to represent in a more
    # consistent way the data that was relabeled.
    data_set_vandalism_editions = data_set[data_set["LABEL"] == 1]
    data_set_vand_relabel_editions = pd.concat([data_set_vandalism_editions, re_labeled_data])

    number_of_original_regular_editions = len(data_set_regular_editions)
    number_of_vandalism_editions = len(data_set_vandalism_editions)
    number_of_relabel_editions = len(re_labeled_data)
    number_of_reg_relabel_editions = number_of_original_regular_editions + number_of_relabel_editions
    number_of_vand_relabel_editions = number_of_vandalism_editions + number_of_relabel_editions

    # This calculation is due to the fact that I want to keep a balanced representation of the vandalism data.
    # This code works, I updated with a more complex calculation.
    # number_of_groups = np.math.floor(number_of_reg_relabel_editions / number_of_vandalism_editions)

    '''number_of_groups = np.math.ceil(number_of_original_regular_editions / number_of_vandalism_editions)
    number_regular_editions_by_group = np.math.floor(number_of_original_regular_editions / number_of_groups)
    number_regular_excluding_relabel = number_regular_editions_by_group - number_of_relabel_editions
    real_number_of_groups = np.math.ceil(number_of_original_regular_editions / number_regular_excluding_relabel)'''

    real_number_of_groups = np.math.ceil(number_of_original_regular_editions / number_of_vandalism_editions)

    regular_editions_by_group = np.array_split(data_set_regular_editions, real_number_of_groups)

    all_balanced_datasets = []
    all_columns_statistical_values = []
    for group_index in range(0, real_number_of_groups):
        if len(regular_editions_by_group[group_index]) > 0:
            regular_editions_to_concat = pd.DataFrame(regular_editions_by_group[group_index],
                                                      columns=data_set.columns)

            # Oversampling vandalism data to match the re_label.
            oversampling_vandalism = data_set_vandalism_editions.sample(n=number_of_relabel_editions)

            # concatenated_data_sets = [regular_editions_to_concat, data_set_vand_relabel_editions]
            concatenated_data_sets = [regular_editions_to_concat, data_set_vand_relabel_editions,
                                      oversampling_vandalism]

            # This lines concatenates both dataframe and also shuffle them.
            balanced_data_set = pd.concat(concatenated_data_sets).sample(frac=1)

            # Removed this code because we don't want to handle the missing values per group of balanced (this is done
            # in the whole training dataset.
            #balanced_data_set, columns_statistical_values = handle_missing_values(balanced_data_set)

            all_balanced_datasets.append(balanced_data_set)

            # It's empty because we are not considering the statistical values individually by balanced dataset
            columns_statistical_values = pd.DataFrame()
            all_columns_statistical_values.append(columns_statistical_values)
            #all_columns_statistical_values.append(columns_statistical_values)

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


def handle_imbalanced_dataset_wrong_classified(data_set, wrong_classified_editions,
                                               new_file_base_path, new_file_name, save_to_file=False,
                                               balanced_folder="/balanced_data/"):

    data_set_regular_editions = data_set.loc[data_set["LABEL"] == 0].copy()

    # Contains all the data that will be present in all balanced dataset. This is necessary to represent in a more
    # consistent way the data that was relabeled.
    data_set_vandalism_editions = data_set[data_set["LABEL"] == 1]

    number_of_original_regular_editions = len(data_set_regular_editions)
    number_of_vandalism_editions = len(data_set_vandalism_editions)

    real_number_of_groups = np.math.ceil(number_of_original_regular_editions / number_of_vandalism_editions)
    regular_editions_by_group = np.array_split(data_set_regular_editions, real_number_of_groups)

    wrong_regular_editions = wrong_classified_editions[wrong_classified_editions["LABEL"] == 0]
    wrong_vandalism_editions = wrong_classified_editions[wrong_classified_editions["LABEL"] == 1]

    wrong_regular_editions_by_group = np.array_split(wrong_regular_editions, real_number_of_groups)
    for group_index in range(0, real_number_of_groups):
        wrong_regular_editions_by_group[group_index] = pd.concat([wrong_regular_editions_by_group[group_index],
                                                                  wrong_vandalism_editions])

    all_balanced_datasets = []
    all_columns_statistical_values = []
    for group_index in range(0, real_number_of_groups):
        if len(regular_editions_by_group[group_index]) > 0:
            regular_editions_to_concat = pd.DataFrame(regular_editions_by_group[group_index],
                                                      columns=data_set.columns)

            # concatenated_data_sets = [regular_editions_to_concat, data_set_vand_relabel_editions]
            concatenated_data_sets = [regular_editions_to_concat,  wrong_regular_editions_by_group[group_index],
                                      data_set_vandalism_editions]

            # This lines concatenates both dataframe and also shuffle them.
            balanced_data_set = pd.concat(concatenated_data_sets).sample(frac=1)

            # Removed this code because we don't want to handle the missing values per group of balanced (this is done
            # in the whole training dataset.
            #balanced_data_set, columns_statistical_values = handle_missing_values(balanced_data_set)

            all_balanced_datasets.append(balanced_data_set)

            # It's empty because we are not considering the statistical values individually by balanced dataset
            columns_statistical_values = pd.DataFrame()
            all_columns_statistical_values.append(columns_statistical_values)
            #all_columns_statistical_values.append(columns_statistical_values)

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


def handle_imbalanced_dataset_with_file(base_file_path, file_name, new_file_base_path, new_file_name,
                                        save_to_file=False):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    data_set = pd.read_csv(file_path, na_values="?")

    all_balanced_datasets, all_columns_statistical_values = handle_imbalanced_dataset(data_set, new_file_base_path,
                                                                                      new_file_name, save_to_file)

    return all_balanced_datasets, all_columns_statistical_values


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


def apply_standardization_balanced(balanced_data_sets):
    standard_data_sets = []
    standard_scalings = []

    for data_set in balanced_data_sets:
        temp_standard_data_set, temp_standard_scaling = apply_standardization(data_set)

        standard_data_sets.append(temp_standard_data_set)
        standard_scalings.append(temp_standard_scaling)

    return standard_data_sets, standard_scalings


def get_data_set_from_file(base_file_path, file_name, null_value="?"):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    data_set = pd.read_csv(file_path, na_values=null_value)

    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    return data_set


def get_data_set_from_file_space(base_file_path, file_name, null_value="?"):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    data_set = pd.read_csv(file_path, na_values=null_value, sep=" ", index_col=False)

    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    return data_set


def get_data_set_balanced(base_file_path, file_name, number_of_files, null_value="?"):
    all_balanced_datasets = []

    for i in range(0, number_of_files):
        file_name_temp = file_name + str(i) + ".csv"
        balanced_temp = get_data_set_from_file(base_file_path, file_name_temp, null_value)

        all_balanced_datasets.append(balanced_temp)

    return all_balanced_datasets


def get_test_data_set_clean(base_file_path, file_name):
    file_path = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name
    data_set = pd.read_csv(file_path, na_values="?")

    data_set = data_set.replace(False, 0)
    data_set = data_set.replace("False", 0)
    data_set = data_set.replace(True, 1)
    data_set = data_set.replace("True", 1)
    data_set = data_set.replace("reg", 0)
    data_set = data_set.replace("vand", 1)

    return data_set


def get_statistical_values_balanced(data_set_columns, all_columns_statistical_values):
    columns_statistical_values = pd.DataFrame(columns=data_set_columns)

    data_set_statistical_values = pd.DataFrame(columns=data_set_columns)
    for statistical_values in all_columns_statistical_values:
        data_set_statistical_values.append(statistical_values, ignore_index=True)

    columns_statistical_values = data_set_statistical_values.mean()
    '''
    for column in non_numeric_columns:
        columns_statistical_values[column] = data_set[column].mode()
    '''

    number_balanced_groups = len(all_columns_statistical_values)

    return columns_statistical_values


def save_data_set(data_set, base_file_path, file_name):
    path_to_save = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name

    pd.DataFrame.to_csv(data_set,
                        path_to_save,
                        index=False)


def save_data_set_with_index(data_set, base_file_path, file_name):
    path_to_save = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name

    pd.DataFrame.to_csv(data_set,
                        path_to_save,
                        index=True)


def build_training_dataset_from_editions(base_file_path, file_name_training, file_name_result):
    # When working with the en_result file, the last space character in the line must be removed.

    file_path_training = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name_training
    file_path_result = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name_result

    data_set_training = pd.read_csv(file_path_training, na_values="null")
    data_set_result = pd.read_csv(file_path_result, na_values="?")

    column_to_search = data_set_training["oldrevisionid"]
    training_data_set = data_set_result[data_set_result["OLDREVID"].isin(column_to_search)].drop_duplicates()

    return training_data_set


def add_label_to_training(base_file_path, training_data_set, file_name_with_labels, file_name_edits):
    file_path_edits = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name_edits
    file_path_labels = str(pathlib.Path(__file__).parent.parent.parent) + base_file_path + file_name_with_labels

    data_set_edits = pd.read_csv(file_path_edits, na_values="null")
    data_set_labels = pd.read_csv(file_path_labels, na_values="?")

    #training_data_set["LABEL"] = []

    old_rev_column_to_search = training_data_set["OLDREVID"]
    ids = data_set_edits["editid"][data_set_edits["oldrevisionid"].isin(old_rev_column_to_search)].drop_duplicates()

    labels = []
    for index, row in training_data_set.iterrows():
        edit_id = data_set_edits["editid"][data_set_edits["oldrevisionid"] == row["OLDREVID"]].values[0]
        label_aux = data_set_labels["class"][data_set_edits["editid"] == edit_id].values[0]
        labels.append(label_aux)

    training_data_set["LABEL"] = labels

    return training_data_set




