import collections
import math
import os
import pathlib
import pickle
import re
from enum import Enum, IntEnum
from pprint import pprint

import numpy as np
import pandas as pd
import spacy
from tensorflow import keras


class LabelDescription(IntEnum):
    REGULAR = 0
    SWEARWORD = 1
    CONTENTREMOVAL = 2
    GRAMATICALMISTAKE = 3
    INSULT = 4
    SEXUAL = 5
    RACISM = 6
    HOMOPHOBIA = 7
    NONNEUTRAL = 8
    WRITINGSTYLE = 9
    FAKENEWS = 10
    DERROGATORYTERMS = 11
    WIKIHYPERSTYLE = 12
    SEXISM = 13
    WRONGCLASSFIED = 14
    DONTEXIST = 15
    TEXTPERSONAL = 16
    NOTRELATED = 17
    VIOLENCE = 18
    RELIGION = 19
    NUMBERING = 20
    ADVERTISING = 21


class ResampleStrategy(IntEnum):
    NoResample = 0
    SimpleResample = 1
    ResampleAugmentedData = 2
    JustAugmentedData = 3
    ResampleAugmentedDataToLimit = 4
    ReplicateMinorities = 5
    ResampleAugmentedDataSpecific = 6
    ResampleUpDownSingle = 7


def create_experiments_folders(save_experiment_directory, number_k_datasets):
    base_directory = str(pathlib.Path(__file__).parent)
    base_experiment_directory = base_directory + save_experiment_directory
    for k in range(0, number_k_datasets):
        folder_to_create = base_experiment_directory + str(k) + "/" + "trained_model/"
        if not os.path.exists(folder_to_create):
            os.mkdir(folder_to_create)


def set_vandalism_label(dataset):
    dataset["VANDALISM"] = dataset["MULTILABEL"].apply(lambda multilabel: 0 if multilabel[0] == 1 else 1)

    return dataset


def create_single_label_outputs(dataset,
                                labels_to_consider=[LabelDescription.SWEARWORD, LabelDescription.SEXUAL,
                                                    LabelDescription.RACISM, LabelDescription.HOMOPHOBIA,
                                                    LabelDescription.DERROGATORYTERMS, LabelDescription.SEXISM]):
    number_of_labels = len(labels_to_consider)
    # Creating the fields on the dataframe for each of the labels considered
    for label in labels_to_consider:
        dataset[label.name] = dataset.apply(lambda _: 0, axis=1)

    for edit_index, edit in dataset.iterrows():
        multi_label = edit["NEWMULTILABEL"]

        for label, current_index in zip(labels_to_consider, range(0, number_of_labels)):
            label_value = multi_label[current_index]
            if label_value:
                dataset.loc[edit_index, label.name] = 1

    return dataset


def create_single_label_outputs_multi(dataset,
                                      labels_to_consider=[LabelDescription.SWEARWORD, LabelDescription.SEXUAL,
                                                          LabelDescription.RACISM, LabelDescription.HOMOPHOBIA,
                                                          LabelDescription.DERROGATORYTERMS, LabelDescription.SEXISM]):
    number_of_labels = len(labels_to_consider)
    all_single_multi_labels = []

    # Creating the fields on the dataframe for each of the labels considered
    for label in labels_to_consider:
        dataset[label.name] = dataset.apply(lambda _: [], axis=1)
        all_single_multi_labels.append([])

    for edit_index, edit in dataset.iterrows():
        multi_label = edit["NEWMULTILABEL"]

        for label, current_index in zip(labels_to_consider, range(0, number_of_labels)):
            label_value = multi_label[current_index]
            new_single_task_multi_label = [1, 0]
            if label_value:
                new_single_task_multi_label[1] = 1

                # If the edit contains only the label of interest as the violation label, then position zero that
                # represents all the other edits is set to false.
                if multi_label.count(1) == 1:
                    new_single_task_multi_label[0] = 0

            all_single_multi_labels[current_index].append(new_single_task_multi_label)

    for label, label_index in zip(labels_to_consider, range(0, number_of_labels)):
        dataset[label.name] = all_single_multi_labels[label_index]

    return dataset


def consider_specific_labels(dataset,
                             labels_to_consider=[LabelDescription.SWEARWORD, LabelDescription.INSULT,
                                                 LabelDescription.SEXUAL, LabelDescription.RACISM,
                                                 LabelDescription.HOMOPHOBIA, LabelDescription.SEXISM]):
    # add new column that contains only the desired specific labels.
    # dataset["NEWMULTILABEL"] = dataset.apply(lambda _: [], axis=1)
    dataset.loc[:, "NEWMULTILABEL"] = dataset.apply(lambda _: [], axis=1)
    all_new_multi_label = []
    number_of_labels = len(labels_to_consider)  # + 1  # + 1 to consider all the other types of violation.
    indexes_to_remove = []

    for edit_index, edit in dataset.iterrows():
        multi_label = edit["MULTILABEL"]
        new_multi_label = [0] * number_of_labels

        label_index = 0
        for label_aux in labels_to_consider:
            new_multi_label[label_index] = multi_label[int(label_aux)]
            label_index += 1

        all_label_descriptions = [item.value for item in LabelDescription]

        # We remove all edits that do not contain any of the labels of interest.
        if new_multi_label == [0] * number_of_labels:
            indexes_to_remove.append(edit_index)
        else:
            all_new_multi_label.append(new_multi_label)

        # If the edit has no other label, then it means the edit is a violation, but it's the non hate speech type.
        # All non hate speech type are grouped together.
        '''if new_multi_label == [0] * number_of_labels:
            new_multi_label.append(1)
        else:
            new_multi_label.append(0)'''

        # We are not removing all edits that do not contain any of the labels of interest, we are just setting them to zero.
        # Because they don't have any of the labels we want, but they are still vandalism, so we need to consider them in the
        # classification. Future work shall handle the above case, in which we remove the edits of no interest.
        # all_new_multi_label.append(new_multi_label)

    dataset.drop(indexes_to_remove, axis=0, inplace=True)
    dataset["NEWMULTILABEL"] = all_new_multi_label

    return dataset


def consider_specific_labels_augmented(dataset,
                                       labels_to_consider=[LabelDescription.SWEARWORD, LabelDescription.INSULT,
                                                           LabelDescription.SEXUAL, LabelDescription.RACISM,
                                                           LabelDescription.HOMOPHOBIA, LabelDescription.SEXISM]):
    # add new column that contains only the desired specific labels.
    # dataset["NEWMULTILABEL"] = dataset.apply(lambda _: [], axis=1)
    dataset.loc[:, "NEWMULTILABEL"] = dataset.apply(lambda _: [], axis=1)
    all_new_multi_label = []
    number_of_labels = len(labels_to_consider)
    indexes_to_remove = []

    for edit_index, edit in dataset.iterrows():
        multi_label = list(map(int, edit["LABEL"].strip('][').split(', ')))
        new_multi_label = [0] * number_of_labels

        label_index = 0
        '''for label_aux in labels_to_consider:
            new_multi_label[label_index] = multi_label[int(label_aux)]
            label_index += 1'''
        new_multi_label[0] = multi_label[0]
        new_multi_label[1] = multi_label[1]
        new_multi_label[2] = multi_label[2]
        new_multi_label[3] = multi_label[3]
        new_multi_label[4] = multi_label[4]
        new_multi_label[5] = multi_label[6]  # we put together insult and derogatory, thus we are not considering derogatory here.

        # We remove all edits that do not contain any of the labels of interest.
        if new_multi_label == [0] * number_of_labels:
            indexes_to_remove.append(edit_index)
        else:
            all_new_multi_label.append(new_multi_label)

    dataset.drop(indexes_to_remove, axis=0, inplace=True)
    dataset["NEWMULTILABEL"] = all_new_multi_label

    return dataset


def consider_specific_labels_single(dataset, labels_to_consider):
    # add new column that contains only the desired specific labels.
    # dataset["NEWMULTILABEL"] = dataset.apply(lambda _: [], axis=1)
    # dataset.loc[:, "NEWMULTILABEL"] = dataset.apply(lambda _: [], axis=1)
    all_new_multi_label = []
    number_of_labels = len(labels_to_consider)
    indexes_to_remove = []

    for edit_index, edit in dataset.iterrows():
        multi_label = edit["NEWMULTILABEL"]
        new_multi_label = [0] * number_of_labels

        label_index = 0
        for label_aux, multi_label_index in zip(multi_label, range(0, 7)):
            if multi_label_index != 1:
                new_multi_label[label_index] = label_aux
                label_index += 1

        # We remove all edits that do not contain any of the labels of interest.
        if new_multi_label == [0] * number_of_labels:
            indexes_to_remove.append(edit_index)
        else:
            all_new_multi_label.append(new_multi_label)

    dataset.drop(indexes_to_remove, axis=0, inplace=True)
    dataset["NEWMULTILABEL"] = all_new_multi_label

    return dataset


def execute_resample(dataset):
    # Start by calculating Imbalance Ratio and get information about the imbalance.
    ir_per_label, index_majority_label = calculate_ir_per_label(dataset)
    # mean_ir = np.mean(ir_per_label)
    number_of_labels = len(ir_per_label)

    for label in range(0, number_of_labels):
        # Multilabel is the position 8.
        current_bag = [edit for edit in dataset.values
                       if edit[8][label] == 1
                       and edit[8][index_majority_label] != 1]

        df_current_bag = pd.DataFrame(current_bag, columns=dataset.columns)

        if ir_per_label[label] >= 2:
            amount_to_replicate = round(ir_per_label[label], 0)
            df_current_bag = df_current_bag.loc[df_current_bag.index.repeat(amount_to_replicate)]
        else:
            balanced_amount_of_edits = round(len(df_current_bag) * ir_per_label[label])
            amount_of_edits_to_add = balanced_amount_of_edits - len(df_current_bag)
            df_current_bag = df_current_bag.sample(n=amount_of_edits_to_add)

        dataset = pd.concat([dataset, df_current_bag])

        # Re-Calculate the imbalance ratio after each re-sampling.
        ir_per_label, index_majority_label = calculate_ir_per_label(dataset)

    # ir_per_label, index_majority_label = calculate_ir_per_label(dataset)
    # mean_ir = np.mean(ir_per_label)

    return dataset.sample(frac=1)


def calculate_ir_per_label(dataset):
    # This function calculates the imbalance ratio per label.

    # Just getting the size of the elements in column NEWMULTILABEL, it indicates how many labels I'm considering.
    number_of_labels = len(dataset.iloc[0]["NEWMULTILABEL"])

    number_of_instances_per_label = [0] * number_of_labels
    for current_label in range(0, number_of_labels):
        # We are counting the amount of times each label contains the value 1. It means that the edit contains certain
        # label.
        number_of_instances_per_label[current_label] = len([multilabel for multilabel in dataset["NEWMULTILABEL"]
                                                            if multilabel[current_label] == 1])

        if number_of_instances_per_label[current_label] == 0:
            number_of_instances_per_label[current_label] = 1

    index_majority_label = np.argmax(number_of_instances_per_label)
    instances_majority_label = number_of_instances_per_label[index_majority_label]

    ir_per_label = list(map(lambda ratio: instances_majority_label / ratio, number_of_instances_per_label))

    return ir_per_label, index_majority_label


def calculate_ir_binary(dataset, violation_label):
    # This function calculates the imbalance ratio for the binary case.
    number_of_instances_per_label = [0, 0]

    number_of_instances_per_label[0] = len(dataset[dataset[violation_label] == 0])
    number_of_instances_per_label[1] = len(dataset[dataset[violation_label] == 1])

    index_majority_label = np.argmax(number_of_instances_per_label)
    ir = number_of_instances_per_label[index_majority_label] / number_of_instances_per_label[not index_majority_label]

    return ir, index_majority_label


def calculate_ir_single_multi(dataset, violation_label):
    number_of_instances_per_label = calculate_instances_single_multi(dataset, violation_label)
    index_majority_label = np.argmax(number_of_instances_per_label)
    ir = number_of_instances_per_label[index_majority_label] / number_of_instances_per_label[not index_majority_label]

    return ir, index_majority_label


def calculate_instances_per_label(dataset, number_of_labels=0):
    if len(dataset) > 0:
        # Just getting the size of the elements in column NEWMULTILABEL, it indicates how many labels I'm considering.
        number_of_labels = len(dataset.iloc[0]["NEWMULTILABEL"])

        number_of_instances_per_label = [0] * number_of_labels
        for current_label in range(0, number_of_labels):
            # We are counting the amount of times each label contains the value 1. It means that the edit contains
            # certain label.
            number_of_instances_per_label[current_label] = len([multilabel for multilabel in dataset["NEWMULTILABEL"]
                                                                if multilabel[current_label] == 1])

        return number_of_instances_per_label
    else:
        return [0] * number_of_labels


def calculate_instances_binary(dataset, violation_label):
    number_of_instances_per_label = [0, 0]

    number_of_instances_per_label[0] = len(dataset[dataset[violation_label] == 0])
    number_of_instances_per_label[1] = len(dataset[dataset[violation_label] == 1])

    return number_of_instances_per_label


def calculate_instances_single_multi(dataset, violation_label):
    # Calculates the number of instances per label for the cases in which we have single models for each label.
    number_of_instances_per_label = [0, 0]

    number_of_instances_per_label[0] = len([multilabel for multilabel in dataset[violation_label]
                                            if multilabel[0] == 1])
    number_of_instances_per_label[1] = len([multilabel for multilabel in dataset[violation_label]
                                            if multilabel[1] == 1])

    return number_of_instances_per_label


def calculate_instances_per_label_binary(dataset):
    vandalism_dataset = dataset[dataset["VANDALISM"] == 1]
    regular_dataset = dataset[dataset["VANDALISM"] == 0]

    number_edits_vandalism = len(vandalism_dataset)
    number_edits_regular = len(regular_dataset)

    return [number_edits_regular, number_edits_vandalism]


def execute_resample_augmented(dataset, experiment_directory):
    # Start by calculating Imbalance Ratio and get information about the imbalance.
    ir_per_label, index_majority_label = calculate_ir_per_label(dataset)
    # mean_ir = np.mean(ir_per_label)
    number_of_labels = len(ir_per_label)

    for label in range(0, number_of_labels):
        # Multilabel is the position 10.
        current_bag = [edit for edit in dataset.values
                       if edit[10][label] == 1
                       and edit[10][index_majority_label] != 1]

        df_current_bag = pd.DataFrame(current_bag, columns=dataset.columns)

        if ir_per_label[label] >= 2:
            amount_to_replicate = round(ir_per_label[label], 0)
            df_current_bag = df_current_bag.loc[df_current_bag.index.repeat(amount_to_replicate)]
        else:
            balanced_amount_of_edits = round(len(df_current_bag) * ir_per_label[label])
            amount_of_edits_to_add = balanced_amount_of_edits - len(df_current_bag)
            df_current_bag = df_current_bag.sample(n=amount_of_edits_to_add)

        augmented_dataset = get_augmented_resample(df_current_bag, experiment_directory)
        dataset = pd.concat([dataset, augmented_dataset])

        # Re-Calculate the imbalance ratio after each re-sampling.
        ir_per_label, index_majority_label = calculate_ir_per_label(dataset)

    # ir_per_label, index_majority_label = calculate_ir_per_label(dataset)
    # mean_ir = np.mean(ir_per_label)

    return dataset.sample(frac=1)


def execute_resample_augmented_to_limit(dataset, experiment_directory):
    class_distribution = calculate_instances_per_label(dataset)
    class_min_value = class_distribution.index(min(class_distribution))
    min_value = class_distribution[class_min_value]

    if min_value < 100:
        # Multilabel is the position 10.
        current_bag = [edit for edit in dataset.values
                       if edit[10][class_min_value] == 1]
        df_current_bag = pd.DataFrame(current_bag, columns=dataset.columns)

        amount_min_class = len(df_current_bag)
        if 100 / amount_min_class:
            amount_to_replicate = 2
            df_current_bag = df_current_bag.loc[df_current_bag.index.repeat(amount_to_replicate)]
        else:
            amount_of_edits_to_add = 100 - amount_min_class
            df_current_bag = df_current_bag.sample(n=amount_of_edits_to_add)

        augmented_dataset = get_augmented_resample(df_current_bag, experiment_directory)
        dataset = pd.concat([dataset, augmented_dataset])

        return execute_resample_augmented_to_limit(dataset, experiment_directory)

    return dataset.sample(frac=1)


def execute_resample_replication(dataset):
    class_distribution = calculate_instances_per_label(dataset)
    class_min_value = class_distribution.index(min(class_distribution))
    min_value = class_distribution[class_min_value]
    '''while min_value == 0:  # TODO: IMPROVE THIS CODE, WE DO THIS TO AVOID PROBLEMS WITH ZERO DIVISION.
        class_distribution[class_min_value] = 1000
        class_min_value = class_distribution.index(min(class_distribution))
        min_value = class_distribution[class_min_value]'''

    if min_value < 50:
        # Multilabel is the position 10.
        current_bag = [edit for edit in dataset.values
                       if edit[10][class_min_value] == 1 and
                       edit[10].count(1) == 1]  # We just want to get those edits that are specific of the replicated type.
        df_current_bag = pd.DataFrame(current_bag, columns=dataset.columns)

        if 100 / min_value > 2:
            amount_to_replicate = 2
            df_current_bag = df_current_bag.loc[df_current_bag.index.repeat(amount_to_replicate)]
        else:
            amount_of_edits_to_add = 100 - min_value
            df_current_bag = df_current_bag.sample(n=amount_of_edits_to_add)

        # augmented_dataset = get_augmented_resample(df_current_bag, experiment_directory)
        # dataset = pd.concat([dataset, augmented_dataset])
        dataset = pd.concat([dataset, df_current_bag])

        return execute_resample_replication(dataset)

    return dataset.sample(frac=1)


def execute_resample_augmented_specific_class(dataset, experiment_directory):
    class_distribution = calculate_instances_per_label(dataset)
    class_min_value = class_distribution.index(min(class_distribution))
    min_value = class_distribution[class_min_value]

    if min_value < 50:
        # Multilabel is the position 10.
        current_bag = [edit for edit in dataset.values
                       if edit[10][class_min_value] == 1 and
                       edit[10].count(1) == 1]  # We just want to get those edits that are specific of the replicated type.
        df_current_bag = pd.DataFrame(current_bag, columns=dataset.columns)

        '''if 50 / min_value > 2:
            amount_to_replicate = 1
            df_current_bag = df_current_bag.loc[df_current_bag.index.repeat(amount_to_replicate)]
        else:
            amount_of_edits_to_add = 100 - min_value
            df_current_bag = df_current_bag.sample(n=amount_of_edits_to_add)'''

        augmented_dataset = get_augmented_resample(df_current_bag, experiment_directory)
        dataset = pd.concat([dataset, augmented_dataset])

        return execute_resample_augmented_specific_class(dataset, experiment_directory)

    return dataset.sample(frac=1)


def execute_resample_augmented_single(dataset, violation_label, labels_to_consider, experiment_directory):
    # Start by calculating Imbalance Ratio and get information about the imbalance.
    ir, index_majority_label = calculate_ir_binary(dataset, violation_label.name)
    index_minority_label = not index_majority_label

    current_bag = dataset[dataset[violation_label.name] == index_minority_label]
    if ir >= 2:
        amount_to_replicate = round(ir, 0)
        current_bag = current_bag.loc[current_bag.index.repeat(amount_to_replicate)]
    else:
        balanced_amount_of_edits = round(len(current_bag) * ir)
        amount_of_edits_to_add = balanced_amount_of_edits - len(current_bag)
        current_bag = current_bag.sample(n=amount_of_edits_to_add)

    augmented_dataset = get_augmented_resample(current_bag, experiment_directory)
    augmented_dataset = merge_badword_insult(augmented_dataset)
    augmented_dataset = merge_similar_violations_augmented(augmented_dataset, LabelDescription.SWEARWORD,
                                                           LabelDescription.INSULT)
    augmented_dataset = consider_specific_labels_single(augmented_dataset, labels_to_consider)
    augmented_dataset = create_single_label_outputs(augmented_dataset, labels_to_consider)

    resampled_dataset = pd.concat([dataset, augmented_dataset])

    # Re-Calculate the imbalance ratio after each re-sampling.
    ir, index_majority_label = calculate_ir_binary(resampled_dataset, violation_label.name)

    return resampled_dataset.sample(frac=1)


def execute_resample_augmented_single_multi(dataset, violation_label, labels_to_consider, experiment_directory):
    # Start by calculating Imbalance Ratio and get information about the imbalance.
    ir, index_majority_label = calculate_ir_single_multi(dataset, violation_label.name)
    index_minority_label = not index_majority_label

    # Here we are getting exclusively those edits that are only part of the violation label.
    violation_label_column = dataset.columns.get_loc(violation_label.name)
    aux_current_bag = [edit for edit in dataset.values
                       if edit[violation_label_column][1]
                       and not edit[violation_label_column][0]]

    current_bag = pd.DataFrame(aux_current_bag, columns=dataset.columns)

    if ir >= 2:
        amount_to_replicate = round(ir, 0)
        current_bag = current_bag.loc[current_bag.index.repeat(amount_to_replicate)]
    else:
        balanced_amount_of_edits = round(len(current_bag) * ir)
        amount_of_edits_to_add = balanced_amount_of_edits - len(current_bag)
        current_bag = current_bag.sample(n=amount_of_edits_to_add)

    augmented_dataset = get_augmented_resample(current_bag, experiment_directory)
    augmented_dataset = merge_badword_insult(augmented_dataset)
    augmented_dataset = merge_similar_violations_augmented(augmented_dataset, LabelDescription.SWEARWORD,
                                                           LabelDescription.INSULT)
    augmented_dataset = consider_specific_labels_single(augmented_dataset, labels_to_consider)
    augmented_dataset = create_single_label_outputs_multi(augmented_dataset, labels_to_consider)

    resampled_dataset = pd.concat([dataset, augmented_dataset])

    # Re-Calculate the imbalance ratio after each re-sampling.
    ir, index_majority_label = calculate_ir_single_multi(dataset, violation_label.name)

    return resampled_dataset.sample(frac=1)


def execute_resampled_augmented_balanced(data_block, violation_label, min_instances_per_label, experiment_directory):
    current_number_of_instances = len(data_block)
    number_of_instances_to_add = min_instances_per_label - current_number_of_instances

    if number_of_instances_to_add > current_number_of_instances:
        data_block_augmented = data_block.loc[data_block.index.repeat(number_of_instances_to_add)]
    else:
        data_block_augmented = data_block.sample(n=number_of_instances_to_add)

    augmented_dataset = get_augmented_resample(data_block_augmented, experiment_directory)
    resampled_dataset = pd.concat([data_block, augmented_dataset]).sample(frac=1)

    return resampled_dataset


def execute_resample_updown_single(dataset, violation_label, labels_to_consider, max_up_percentage=1,
                                   max_down_percentage=0.5):
    # We are getting the data points that belong to the violation_label (the positive class).
    current_violation_bag = dataset[dataset[violation_label.name] == 1]
    number_of_violation_label = len(current_violation_bag)
    number_of_non_violation_label = len(dataset) - number_of_violation_label

    optimal_number_of_violations = number_of_violation_label + round((max_up_percentage * number_of_violation_label), 0)
    if number_of_non_violation_label > optimal_number_of_violations:
        amount_to_add = round(max_up_percentage * number_of_violation_label, 0)
    else:
        if number_of_non_violation_label > number_of_violation_label:
            amount_to_add = number_of_non_violation_label - number_of_violation_label
        else:
            amount_to_add = 0

    if amount_to_add > number_of_violation_label:
        current_violation_bag = current_violation_bag.loc[current_violation_bag.index.repeat(amount_to_add)]
    else:
        current_violation_bag = current_violation_bag.sample(n=amount_to_add)

    resampled_dataset = pd.concat([dataset, current_violation_bag])
    # Getting the amount of data points that are not of the violation_label type.
    non_violation_label_bag = resampled_dataset[resampled_dataset[violation_label.name] == 0]
    number_of_non_violation_label = len(non_violation_label_bag)
    number_of_violation_label = len(resampled_dataset[resampled_dataset[violation_label.name] == 1])

    if number_of_non_violation_label > number_of_violation_label:
        amount_to_down_sample = int(number_of_non_violation_label * max_down_percentage)
        indexes_to_remove = non_violation_label_bag.sample(n=amount_to_down_sample).index
        resampled_dataset.drop(indexes_to_remove, inplace=True)

    return resampled_dataset.sample(frac=1)


def get_augmented_data(dataset, experiment_directory):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = experiment_directory + "augmented_dataset.csv"
    augmented_dataset_path = base_directory + file_path
    augmented_dataset = pd.read_csv(augmented_dataset_path)

    new_data_points = []
    for index, edit in dataset.iterrows():
        url = edit["URL"]
        augmented_data_points = augmented_dataset[augmented_dataset["URL"] == url]
        if len(augmented_data_points) > 1:
            augmented_edit = augmented_data_points.sample(1)
        else:
            augmented_edit = augmented_data_points.sample(1)

        for augmented_index, edit_augmented in augmented_edit.iterrows():
            data_point_to_add = {"NEW_FORMATTED_TEXT": edit_augmented["TEXT"],
                                 "NEWMULTILABEL": list(map(int, edit_augmented["LABEL"].strip('][').split(', '))),
                                 "URL": url}

            new_data_points.append(data_point_to_add)

    dataset = pd.concat([dataset, pd.DataFrame(new_data_points)]).sample(frac=1)

    return dataset


def get_augmented_resample(dataset, experiment_directory):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = experiment_directory + "augmented_dataset.csv"
    augmented_dataset_path = base_directory + file_path
    augmented_dataset = pd.read_csv(augmented_dataset_path)

    # This is necessary since we are changing the base dataset that we are comparing with.
    augmented_dataset = merge_similar_violations_augmented(augmented_dataset, 1, 5)
    augmented_dataset = consider_specific_labels_augmented(augmented_dataset)

    augmented_data_points = []
    for index, edit in dataset.iterrows():
        url = edit["URL"]
        augmented_edit = augmented_dataset[augmented_dataset["URL"] == url].sample(1).iloc[0]

        data_point_to_add = {"NEW_FORMATTED_TEXT": augmented_edit["TEXT"],
                             "NEWMULTILABEL": augmented_edit["NEWMULTILABEL"],
                             "URL": url}

        augmented_data_points.append(data_point_to_add)

    return pd.DataFrame(augmented_data_points)


def save_test(test_results, save_experiment_directory, current_folder, file_name="test_results.pickle"):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = save_experiment_directory + str(current_folder) + "/" + file_name
    complete_path_to_save = base_directory + file_path

    pickle.dump(test_results, open(complete_path_to_save, 'wb'))


def save_model(trained_model, save_experiment_directory, k_validation=0, trained_model_folder_name="trained_model/"):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = save_experiment_directory + str(k_validation) + "/" + trained_model_folder_name
    complete_path_to_save = base_directory + file_path

    keras.models.save_model(trained_model, complete_path_to_save)


def save_models(trained_models, save_experiment_directory, k_validation=0, trained_model_folder_name="trained_model_"):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = save_experiment_directory + str(k_validation) + "/" + trained_model_folder_name

    number_of_models = len(trained_models)
    for model_index in range(0, number_of_models):
        complete_path_to_save = base_directory + file_path + str(model_index) + "/"
        keras.models.save_model(trained_models[model_index], complete_path_to_save)


def create_folder_to_save(complete_path_to_save):
    if not os.path.exists(complete_path_to_save):
        os.mkdir(complete_path_to_save)


def create_balanced_datasets(dataset, initial_number_of_classifiers=12):
    dataset_regular = dataset[dataset["VANDALISM"] == 0]
    dataset_vandalism = dataset[dataset["VANDALISM"] == 1]

    number_of_regular = len(dataset_regular)
    number_of_vandalism = len(dataset_vandalism)

    number_of_classifiers = np.math.ceil(number_of_regular / number_of_vandalism)

    regular_by_classifier = np.array_split(dataset_regular, number_of_classifiers)

    balanced_datasets = []
    for classifier_index in range(0, number_of_classifiers):
        regular_to_concat = pd.DataFrame(regular_by_classifier[classifier_index],
                                         columns=dataset.columns)

        # Putting together the different regular edits with the same vandalism edits.
        # This is done to balance the dataset.
        concatenated_datasets = [regular_to_concat, dataset_vandalism]
        current_balanced_dataset = pd.concat(concatenated_datasets).sample(frac=1)

        balanced_datasets.append(current_balanced_dataset)

    return balanced_datasets


def old_format_edit_text(dataset, file_name, load_from_disk=True):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = base_directory + "/" + file_name + ".pickle"

    if load_from_disk:
        formatted_text_dataset = pickle.load(open(file_path, "rb"))

        # text_to_print = formatted_text_dataset["FORMATTED_TEXT"].apply(lambda text: print(text))
        return formatted_text_dataset
    else:
        spacy_nlp = spacy.load("en_core_web_sm")
        # spacy_nlp.add_pipe("sentencizer")

        # Only getting the parts of the edits that are not "None".
        dataset["FORMATTED_TEXT"] = dataset["TEXT"].apply(lambda parts_of_edit: ' '.join([part for part in
                                                                                          parts_of_edit if part !=
                                                                                          "None"]))

        dataset.drop(dataset[dataset["FORMATTED_TEXT"] == ""].index, inplace=True)

        '''edit = "my fucking best run running do not doing sucking ohisfd w"
        test = [token for token in spacy_nlp(edit).sents]
        lemmas_test = [token.lemma_ for token in test]
    
        dataset["NEW_TEST"] = dataset["FORMATTED_TEXT"].apply(lambda edit:
                                                              len([sent_aux for sent_aux in spacy_nlp(edit).sents]))
    
        dataset["NEW_TEST_2"] = dataset["NEW_TEST"].apply(lambda edit: "ARRRUDFIDFH" if edit == 0 else "")
    
        dataset["NEW_TEST"] = dataset["FORMATTED_TEXT"].apply(lambda edit:
                                                              [sent_aux for sent_aux in spacy_nlp(edit).sents][0].lemma_)'''

        dataset["FORMATTED_TEXT"] = dataset["FORMATTED_TEXT"].apply(lambda edit:
                                                                    ' '.join([token.lemma_ for token in spacy_nlp(edit)
                                                                              ]))
        # if token.text.isalpha() is True

        pickle.dump(dataset, open(file_path, "wb"))

        return dataset


def merge_badword_insult(dataset):
    # The goal is to put together those edits that are insult and bad word. They are better together.
    all_new_multi_label = []
    for index, edit in dataset.iterrows():
        multi_label = edit["NEWMULTILABEL"]
        if multi_label[0] == 1:
            multi_label[1] = 1
        elif multi_label[1] == 1:
            multi_label[0] = 1

        # dataset.loc[index, "NEWMULTILABEL"] = multi_label
        all_new_multi_label.append(multi_label)

    # Temp, removing to see how it works
    # dataset["NEWMULTILABEL"] = all_new_multi_label
    return dataset


def merge_similar_violations(dataset, main_label, secondary_label):
    # Here we are going to put together labels that are similar. Main label is going to receive the edits that contain
    # secondary_label, while secondary_label is not going to be considered in the dataset anymore.
    dataset = dataset.loc[dataset["VANDALISM"] == 1].copy()

    for index, edit in dataset.iterrows():
        multi_label = edit["MULTILABEL"]

        # If the secondary label is true, then we are setting the main_label as the type of vandalism as well.
        if multi_label[int(secondary_label)]:
            multi_label[int(main_label)] = 1

    return dataset


def merge_similar_violations_augmented(dataset, main_label, secondary_label):
    # Here we are going to put together labels that are similar. Main label is going to receive the edits that contain
    # secondary_label, while secondary_label is not going to be considered in the dataset anymore.
    # dataset = dataset.loc[dataset["VANDALISM"] == 1].copy()

    for index, edit in dataset.iterrows():
        multi_label = list(map(int, edit["LABEL"].strip('][').split(', ')))

        # If the secondary label is true, then we are setting the main_label as the type of vandalism as well.
        if multi_label[int(secondary_label)]:
            multi_label[int(main_label)] = 1

    return dataset


def execute_resample_strategy(dataset, resample_strategy, dataset_directory_augmented_data=""):
    if resample_strategy == ResampleStrategy.SimpleResample:
        return execute_resample(dataset)
    elif resample_strategy == ResampleStrategy.ResampleAugmentedData:
        return execute_resample_augmented(dataset, dataset_directory_augmented_data)
    elif resample_strategy == ResampleStrategy.JustAugmentedData:
        return get_augmented_data(dataset, dataset_directory_augmented_data)
    elif resample_strategy == ResampleStrategy.ResampleAugmentedDataToLimit:
        return execute_resample_augmented_to_limit(dataset, dataset_directory_augmented_data)
    elif resample_strategy == ResampleStrategy.ReplicateMinorities:
        return execute_resample_replication(dataset)
    elif resample_strategy == ResampleStrategy.ResampleAugmentedDataSpecific:
        return execute_resample_augmented_specific_class(dataset, dataset_directory_augmented_data)

    # No resample was applied
    return dataset


def save_interpretability_dataset(dataset, directory_to_save):
    pickle.dump(dataset, open(directory_to_save + "training_dataset.pickle", 'wb'))


def load_interpretability_dataset(interpretability_directory):
    dataset_location = interpretability_directory + "training_dataset.pickle"
    return pickle.load(open(dataset_location, "rb"))
