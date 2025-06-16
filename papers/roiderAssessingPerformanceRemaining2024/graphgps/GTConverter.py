import numpy as np
import data_preprocessing
import torch
from torch_geometric.data import Data
import os
import yaml
import pm4py
import joblib
import random
import pickle
import os.path as osp
import pandas as pd

import warnings
warnings.simplefilter('ignore')


# Read user inputs from .yml file
def read_user_inputs(file_path):
    with open(file_path, 'r') as f:
        user_inputs = yaml.safe_load(f)
    return user_inputs


# Get other relevant global information from training data
def get_global_stat_func(log, config, oh_encoders):
    """
    node_class_dict (activity classes), keys: tuple of activity/lifecycle, value: integer representation
    max_case_df: maximum number of same df relationship in one case
    """
    max_case_length = 0
    node_class_dict = {}
    node_class_rep = 0
    # Get node_class_dict, max_case_df
    for case_counter in range(len(log)):
        df_dict = {}  # Dict: all activity-class df relationships and their frequencies
        current_case = log[case_counter]
        case_length = len(current_case)
        if case_length > 1:
            # iterate over all events of the case, collect df information
            for event_counter in range(case_length - 1):
                source_class = (current_case[event_counter]['concept:name'],)
                target_class = (current_case[event_counter + 1]['concept:name'],)
                df_class = (source_class, target_class)
                if df_class in df_dict:
                    df_dict[df_class] += 1
                else:
                    df_dict[df_class] = 1
                if not (source_class in node_class_dict):
                    node_class_dict[source_class] = node_class_rep
                    node_class_rep += 1
            if max((df_dict).values()) > max_case_length:
                max_case_length = max((df_dict).values())

    # Get node and edge dimensions
    node_dim = len(node_class_dict.keys())  # size for node featuers

    # Get info for calculating edge_dim
    cat_cardinality = 0
    for name, encoder in oh_encoders.items():
        if name == config['activity_column']:
            continue
        cat_cardinality += len(encoder.categories_[0])

    num_cardinality = len(config['static_numerical_columns']) + len(config['dynamic_numerical_columns'])

    # edge_dim = attribute_cardinality + case_cardinality + case_num_card + event_num_card + 7
    # +1 for special feature in graph_conversion_func
    edge_dim = cat_cardinality + num_cardinality + 1

    return node_dim, edge_dim, max_case_length


def graph_conversion_func(df,
                          config,
                          oh_encoders,
                          edge_dim,
                          max_case_length,
                          data_list,
                          idx):

    grouped = df.groupby(config['case_id_column'])
    num_cases = len(grouped)
    i = 0
    for case_id, group_df in grouped:

        case_length = len(group_df)
        case_level_feat = np.empty((0,))

        # first categorical attributes
        # One hot encoding of features and directly append to case_level_feat
        for col in config['static_categorical_columns']:
            case_att = str(group_df[col].iloc[0])
            encoder = oh_encoders[col]
            case_att_enc = data_preprocessing.one_hot_encoding(encoder=encoder,
                                                               value_to_encode=case_att)
            case_level_feat = np.append(case_level_feat, case_att_enc)

        # now, numerical attributes
        # We normalized the input features already when preparing the data
        # Missing values are set to 0.
        for col in config['static_numerical_columns']:
            case_level_feat = np.append(case_level_feat, np.array(float(group_df[col].iloc[0])))

        # collect activity classes, timestamps, and all attributes of intrest for each event
        # We don't use the lifecycle:transition attribute.
        collection_lists = [[] for _ in range(len(config['dynamic_categorical_columns']) + len(config['dynamic_numerical_columns']) + 2)]
        remaining_times = list()

        for index, row in group_df.iterrows():
            remaining_times.append(row['remaining_time_normalized'])
            collection_lists[0].append(row[config['activity_column']])
            collection_lists[1].append(row[config['timestamp_column']])
            for attribute_counter in range(2, len(config['dynamic_categorical_columns']) + 2):
                collection_lists[attribute_counter].append(row[config['dynamic_categorical_columns'][attribute_counter - 2]])
            for attribute_counter in range(len(config['dynamic_categorical_columns']) + 2,
                                           len(config['dynamic_categorical_columns']) + len(config['dynamic_numerical_columns']) + 2):
                collection_lists[attribute_counter].append(row[config['dynamic_numerical_columns'][attribute_counter - len(config['dynamic_categorical_columns']) - 2]])


        for prefix_length in range(2, case_length + 1):
            prefix_event_classes = collection_lists[0][:prefix_length]
            prefix_classes = list(set(prefix_event_classes))  # only includes unique classes

            # create target based on the normalization option for user
            # Get normalized target directly from event log
            target_cycle = np.array(remaining_times[prefix_length - 1])
            y = torch.from_numpy(target_cycle).float()

            # collect information about nodes
            # define zero array to collect node features of the graph associated to this prefix
            node_feature = np.zeros((len(prefix_classes), 1), dtype=np.int64)
            # collect node type by iteration over all nodes in the graph.
            for prefix_class in prefix_classes:
                # get index of the relevant prefix class, and update its row in node feature matirx
                encoder = oh_encoders[config['activity_column']]
                transformed_value = encoder.transform(np.asarray(prefix_class).reshape(1, -1)).astype(int)[0][0]
                if np.isnan(transformed_value) or transformed_value < 0:
                    transformed_value = len(encoder.categories_[0])

                node_feature[prefix_classes.index(prefix_class)] = transformed_value
            x = torch.from_numpy(node_feature).long()

            # Compute edge index list.
            # Each item in pair_result: tuple of tuples representing df between two activity classes
            pair_result = list(zip(prefix_event_classes, prefix_event_classes[1:]))
            pair_freq = {}
            for item in pair_result:
                source_index = prefix_classes.index(item[0])
                target_index = prefix_classes.index(item[1])
                if ((source_index, target_index) in pair_freq):
                    pair_freq[(source_index, target_index)] += 1
                else:
                    pair_freq[(source_index, target_index)] = 1
            edges_list = list(pair_freq.keys())
            edge_index = torch.tensor(edges_list, dtype=torch.long)

            # Compute edge attributes
            edge_feature = np.zeros((len(edge_index), edge_dim), dtype=np.float64)  # initialize edge feature matrix
            edge_counter = 0
            for edge in edge_index:
                source_indices = [i for i, x in enumerate(prefix_event_classes) if x == prefix_classes[edge[0]]]
                target_indices = [i for i, x in enumerate(prefix_event_classes) if x == prefix_classes[edge[1]]]
                acceptable_indices = [(x, y) for x in source_indices for y in target_indices if x + 1 == y]
                special_feat = np.empty((0,))  # collect all special features
                # Add edge weights to the special feature vector
                num_occ = len(acceptable_indices) / max_case_length
                special_feat = np.append(special_feat, np.array(num_occ))
                partial_edge_feature = np.append(special_feat, case_level_feat)

                # One-hot encoding for the target of last occurence + numerical event attributes
                for attribute_counter in range(2, len(config['dynamic_categorical_columns']) + 2):
                    encoder = oh_encoders[config['dynamic_categorical_columns'][attribute_counter-2]]
                    attribute_value = np.array(collection_lists[attribute_counter][acceptable_indices[-1][1]]).reshape(-1, 1)[0][0]
                    ohe = data_preprocessing.one_hot_encoding(encoder=encoder,
                                                              value_to_encode=attribute_value)
                    partial_edge_feature = np.append(partial_edge_feature, ohe)

                # Numerical event attributes
                for attribute_counter in range(len(config['dynamic_categorical_columns']) + 2,
                                               len(config['dynamic_categorical_columns']) + len(
                                                   config['dynamic_numerical_columns']) + 2):
                    attribute_value = np.array(collection_lists[attribute_counter][acceptable_indices[-1][1]])
                    partial_edge_feature = np.append(partial_edge_feature, attribute_value)

                edge_feature[edge_counter, :] = partial_edge_feature
                edge_counter += 1

            edge_attr = torch.from_numpy(edge_feature).float()
            graph = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y, cid=case_id,
                         pl=prefix_length)

            data_list.append(graph)
            idx += 1

        i += 1
        print(i, num_cases, end='\r')

    return data_list, idx


def create_graph_dataset(input_dataset_location,
                         graph_dataset_path_raw):
    train_df = pd.read_pickle(os.path.join(input_dataset_location, 'train.pkl'))
    val_df = pd.read_pickle(os.path.join(input_dataset_location, 'val.pkl'))
    test_df = pd.read_pickle(os.path.join(input_dataset_location, 'test.pkl'))
    config = joblib.load(os.path.join(input_dataset_location, 'config.pkl'))
    oh_encoders = joblib.load(os.path.join(input_dataset_location, 'oh_encoders.pkl'))

    train_df = train_df[train_df['remaining_time'] != 0]
    val_df = val_df[val_df['remaining_time'] != 0]
    test_df = test_df[test_df['remaining_time'] != 0]

    train_df[config['timestamp_column']] = pd.to_datetime(train_df[config['timestamp_column']], format='mixed',
                                                          infer_datetime_format=True)
    train_df = train_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

    val_df[config['timestamp_column']] = pd.to_datetime(val_df[config['timestamp_column']], format='mixed',
                                                        infer_datetime_format=True)
    val_df = val_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

    test_df[config['timestamp_column']] = pd.to_datetime(test_df[config['timestamp_column']], format='mixed',
                                                         infer_datetime_format=True)
    test_df = test_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

    train_df[config['case_id_column']] = train_df[config['case_id_column']].astype(str)
    node_dim, edge_dim, max_case_length = get_global_stat_func(log=pm4py.convert_to_event_log(train_df),
                                                               config=config,
                                                               oh_encoders=oh_encoders)

    idx = 0  # index for graphs
    data_list = []  # a list to collect all Pytorch geometric data objects.
    data_list, idx = graph_conversion_func(df=train_df,
                                           config=config,
                                           oh_encoders=oh_encoders,
                                           edge_dim=edge_dim,
                                           max_case_length=max_case_length,
                                           data_list=data_list,
                                           idx=idx)
    last_train_idx = idx
    data_list, idx = graph_conversion_func(df=val_df,
                                           config=config,
                                           oh_encoders=oh_encoders,
                                           edge_dim=edge_dim,
                                           max_case_length=max_case_length,
                                           data_list=data_list,
                                           idx=idx)
    last_val_idx = idx
    data_list, idx = graph_conversion_func(df=test_df,
                                           config=config,
                                           oh_encoders=oh_encoders,
                                           edge_dim=edge_dim,
                                           max_case_length=max_case_length,
                                           data_list=data_list,
                                           idx=idx)

    indices = list(range(len(data_list)))
    # data split based on the global graph list
    train_indices = indices[:last_train_idx]
    val_indices = indices[last_train_idx:last_val_idx]
    test_indices = indices[last_val_idx:]
    data_train = [data_list[i] for i in train_indices]
    data_val = [data_list[i] for i in val_indices]
    data_test = [data_list[i] for i in test_indices]

    # shuffle the data in each split, to avoid order affect training process
    random.shuffle(data_train)
    random.shuffle(data_val)
    random.shuffle(data_test)
    # Save the training, validation, and test datasets
    file_save_list = [data_train, data_val, data_test]
    if not os.path.exists(graph_dataset_path_raw):
        os.makedirs(graph_dataset_path_raw)

    output_address_list = ['train.pickle', 'val.pickle', 'test.pickle']
    for address_counter in range(len(output_address_list)):
        save_address = osp.join(graph_dataset_path_raw, output_address_list[address_counter])
        save_flie = open(save_address, "wb")
        pickle.dump(file_save_list[address_counter], save_flie)
        save_flie.close()
