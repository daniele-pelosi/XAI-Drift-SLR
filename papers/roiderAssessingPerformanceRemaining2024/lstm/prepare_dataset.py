import pandas as pd
import joblib
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from tqdm import tqdm
from itertools import chain
import h5py
import torch.nn.functional as F


def convert_case_ids(df, config):
    """
    Convert case ids to numerical format. Str is not supported by h5py by default.

    Parameters
    ----------
    df: pd.DataFrame
    config: dict
        Dataset configuration.

    Returns
    -------
    pd.DataFrame
    """
    groups = list()
    current_case_id = 1
    if isinstance(df[config['case_id_column']].iloc[0], str):
        # Need to create int type for case id to be compatible with hdf5 storage
        for index, group_df in df.groupby(config['case_id_column']):
            group_df[config['case_id_column']] = current_case_id
            groups.append(group_df)
            current_case_id += 1

        return pd.concat(groups)
    return df


def prepare_dataset(input_data_filepath,
                    output_data_filepath):
    """
    Prepare datasets for training, validation and testing.

    Parameters
    ----------
    input_data_filepath: str
    output_data_filepath: str

    Returns
    -------
    None
    """

    if not Path(output_data_filepath).exists():
        Path(output_data_filepath).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_pickle(input_data_filepath + '/train.pkl')
    val_df = pd.read_pickle(input_data_filepath + '/val.pkl')
    test_df = pd.read_pickle(input_data_filepath + '/test.pkl')

    # Remove 0 labels
    train_df = train_df[train_df['remaining_time'] != 0]
    val_df = val_df[val_df['remaining_time'] != 0]
    test_df = test_df[test_df['remaining_time'] != 0]

    config = joblib.load(input_data_filepath + '/config.pkl')
    oh_encoders = joblib.load(input_data_filepath + '/oh_encoders.pkl')
    # label column is already normalized!

    # Convert case id's to numerical format. Str is not supported by h5py by default.
    train_df = convert_case_ids(df=train_df, config=config)
    val_df = convert_case_ids(df=val_df, config=config)
    test_df = convert_case_ids(df=test_df, config=config)

    train_df[config['timestamp_column']] = pd.to_datetime(train_df[config['timestamp_column']], format='mixed',
                                                    infer_datetime_format=True)
    train_df = train_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

    val_df[config['timestamp_column']] = pd.to_datetime(val_df[config['timestamp_column']], format='mixed',
                                                    infer_datetime_format=True)
    val_df = val_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

    test_df[config['timestamp_column']] = pd.to_datetime(test_df[config['timestamp_column']], format='mixed',
                                                    infer_datetime_format=True)
    test_df = test_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])


    hdf5_filepath_train = output_data_filepath + '/train.hdf5'
    hdf5_filepath_test = output_data_filepath + '/test.hdf5'

    maxlen = generate_set(df=train_df,
                          hdf5_filepath=hdf5_filepath_train,
                          data_name='train_X',
                          labels_name='train_y',
                          cases_name='train_cases',
                          data_columns_name='train_columns',
                          config=config,
                          maxlen=None,
                          oh_encoders=oh_encoders)

    generate_set(df=val_df,
                 hdf5_filepath=hdf5_filepath_train,
                 data_name='validation_X',
                 labels_name='validation_y',
                 cases_name='validation_cases',
                 data_columns_name='validation_columns',
                 config=config,
                 maxlen=maxlen,
                 oh_encoders=oh_encoders)

    generate_set(df=test_df,
                 hdf5_filepath=hdf5_filepath_test,
                 data_name='test_X',
                 labels_name='test_y',
                 cases_name='test_cases',
                 data_columns_name='test_columns',
                 config=config,
                 maxlen=maxlen,
                 oh_encoders=oh_encoders)


def get_categorical_column_names(oh_encoders: dict,
                                 cols: list):
    col_names = list()
    for col in cols:
        num_idx = len(oh_encoders[col].categories_[0])
        col_names.extend([col + '||' + str(i) for i in range(num_idx)])
    return col_names


def get_column_names(config: dict,
                     oh_encoders: dict):
    cols = list()
    cols.extend(get_categorical_column_names(oh_encoders=oh_encoders,
                                             cols=config['dynamic_categorical_columns']))
    cols.extend(get_categorical_column_names(oh_encoders=oh_encoders,
                                             cols=config['static_categorical_columns']))
    cols.extend(config['dynamic_numerical_columns'])
    cols.extend(config['static_numerical_columns'])
    cols.append(config['activity_column'])
    return cols


def buildOHE(index: int,
             n: int):
    """
    Get one hot encoding.

    Parameters
    ----------
    index: int
        The index to be encoded as 1.
    n: int
        Total number of values present. Determines the length of the encoded vector.

    Returns
    -------
    list
        One hot encoding as a list.

    """
    L=[0.]*n
    L[index]=1.
    return L


def generate_batch(index: int,
                   group_df: pd.DataFrame,
                   oh_encoders: dict,
                   config: dict,
                   maxlen: int):

    categorical_columns = list()
    categorical_columns.extend(config['dynamic_categorical_columns'])
    categorical_columns.extend(config['static_categorical_columns'])
    categorical_columns.append(config['activity_column'])

    group_df[categorical_columns] = group_df[categorical_columns].astype(str)

    numerical_columns = list()
    numerical_columns.extend(config['dynamic_numerical_columns'])
    numerical_columns.extend(config['static_numerical_columns'])

    data_large = list()
    labels = list()
    cases = list()

    length = len(group_df)
    start = 2 # Start from the second event for fair comparison with graph based method
    labels.append(group_df['remaining_time_normalized'].values.tolist()[1:])
    cases.extend([index for i in range(len(group_df))])
    for i in range(start, length + 1):
        df_tmp = group_df[:i]

        if len(df_tmp) > maxlen:
            df_tmp = df_tmp[-maxlen:]

        data_trace = list()
        for index_, row in df_tmp.iterrows():
            data_row = list()
            for col in categorical_columns:
                idx = oh_encoders[col].transform(np.asarray(row[col]).reshape(1, -1)).astype(int)[0][0]

                if np.isnan(idx) or idx < 0:
                    data_row.extend([0] * len(oh_encoders[col].categories_[0]))
                else:
                    data_row.extend(buildOHE(int(idx), len(oh_encoders[col].categories_[0])))

            for col in numerical_columns:
                data_row.append(row[col])

            data_trace.append(data_row)

        data_large.append(data_trace)

    return data_large, labels, cases


def generate_set(df,
                 config,
                 hdf5_filepath,
                 oh_encoders,
                 data_columns_name,
                 data_name,
                 labels_name,
                 cases_name,
                 maxlen=None):
    """

    Parameters
    ----------
    df: pd.DataFrame
    config: dict
    hdf5_filepath: str
    oh_encoders: dict
    data_columns_name: str
        Name of the columns key in the hdf5 file.
    data_name: str
        Name of input features key in the hdf5 file.
    labels_name: str
           Name of the labels key in the hdf5 file.
    cases_name: str
           Name of the cases key in the hdf5 file.
    maxlen: int
        Maximum length of the traces. If None, the maximum length of the traces is used.

    Returns
    -------

    """
    grouped = df.groupby(by=config['case_id_column'])

    if maxlen is None:
        maxlen = grouped.size().max()

    data_index = 0
    labels_index = 0
    cases_index = 0

    if not Path(hdf5_filepath).exists():
        mode = 'w'
    else:
        mode = 'r+'

    with h5py.File(hdf5_filepath, mode) as f:
        pbar = tqdm(total=len(grouped))
        for index, group in grouped:
            data, labels, cases = generate_batch(group_df=group,
                                                 index=index,
                                                 oh_encoders=oh_encoders,
                                                 config=config,
                                                 maxlen=maxlen)

            data_list = list()
            for d in data:
                data_list.append(torch.from_numpy(np.asarray(d)))

            data_list[0] = F.pad(data_list[0], pad=(0, 0, 0, maxlen - data_list[0].shape[0]))
            data = pad_sequence(data_list, batch_first=True).numpy()

            labels = np.asarray(list(chain.from_iterable(labels)))

            if data_index == 0:

                dynamic_cols= get_column_names(config=config,
                                               oh_encoders=oh_encoders)

                # Store column names
                data_cols = f.create_dataset(data_columns_name, data=dynamic_cols)

                data_h5 = f.create_dataset(data_name, (len(df) - len(grouped), maxlen, data.shape[2]), dtype='float32')
                labels_h5 = f.create_dataset(labels_name, (len(df) - len(grouped),), dtype='float32')
                cases_h5 = f.create_dataset(cases_name, (len(df),))
                keys = [key for key in f.keys()]

            data_h5[data_index: data_index + len(data)] = data
            data_index += len(data)
            labels_h5[labels_index: labels_index + len(labels)] = labels
            labels_index += len(labels)
            cases_h5[cases_index: cases_index + len(cases)] = cases
            cases_index += len(cases)

            pbar.update(1)
        pbar.close()

    return maxlen
