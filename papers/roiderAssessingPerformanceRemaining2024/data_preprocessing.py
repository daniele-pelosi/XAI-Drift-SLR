import pandas as pd
import feature_calculation
from configs.datasets import dataset_config
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from pathlib import Path
import joblib
from copy import deepcopy


def prepare_data(dataset_name,
                 dataset_location,
                 output_location):
    """
    Prepare the data for the given dataset.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset.
    dataset_location: str
        Location of the dataset.
    output_location: str
        Location where the prepared dataset should be stored.

    Returns
    -------
    None
    """

    df = pd.read_csv(dataset_location + '/' + dataset_name + '.csv')
    config = getattr(dataset_config, dataset_name)

    df, config = calculate_additional_features(df=df,
                                               config=config)

    # Remove events with remaining time == 0
    df = df[df['remaining_time'] != 0]

    # Split data
    train_df, val_df, test_df = data_split(df=df,
                                           config=config)

    # Remove events from validation and test set which do not occur in training set
    # Keep only cases which have at least two events with remaining time > 0
    for index, group in train_df.groupby(by=config['case_id_column']):
        if len(group) < 2:
            train_df = train_df[train_df[config['case_id_column']] != index]
    unique_events = train_df[config['activity_column']].unique()

    val_df = val_df[val_df[config['activity_column']].isin(unique_events)]
    for index, group in val_df.groupby(by=config['case_id_column']):
        if len(group) < 2:
            val_df = val_df[val_df[config['case_id_column']] != index]

    test_df = test_df[test_df[config['activity_column']].isin(unique_events)]
    for index, group in test_df.groupby(by=config['case_id_column']):
        if len(group) < 2:
            test_df = test_df[test_df[config['case_id_column']] != index]

    # Replace categorical outliers
    categorical_cols = list()
    categorical_cols.extend(deepcopy(config['static_categorical_columns']))
    categorical_cols.extend(deepcopy(config['dynamic_categorical_columns']))
    for col in categorical_cols:
        train_df, to_replace = replace_categorical_outliers(df=train_df,
                                                            case_id_column=config['case_id_column'],
                                                            col=col,
                                                            perc_threshold=0.02,
                                                            to_replace=None)
        val, to_replace = replace_categorical_outliers(df=val_df,
                                                       case_id_column=config['case_id_column'],
                                                       col=col,
                                                       perc_threshold=0.02,
                                                       to_replace=to_replace)
        test_df, to_replace = replace_categorical_outliers(df=test_df,
                                                           case_id_column=config['case_id_column'],
                                                           col=col,
                                                           perc_threshold=0.02,
                                                           to_replace=to_replace)
    # Encoding for categorical values and activities
    ohe_encoders = dict()
    for col in categorical_cols:
        train_df[col] = train_df[col].astype(str)
        ohe_encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value',
                                           unknown_value=np.nan).fit(train_df[col].to_numpy().reshape(-1, 1))
    train_df[config['activity_column']] = train_df[config['activity_column']].astype(str)
    ohe_encoders[config['activity_column']] = OrdinalEncoder(handle_unknown='use_encoded_value',
                                                             unknown_value=np.nan).fit(train_df[config['activity_column']].to_numpy().reshape(-1, 1))

    # Normalize numerical columns
    for col in config['static_numerical_columns']:
        train_df[col] = train_df[col].astype(float)
        max_value = train_df[col].abs().max()
        if max_value != 0:
            train_df[col] = train_df[col] / max_value
            val_df[col] = val_df[col] / max_value
            test_df[col] = test_df[col] / max_value
    for col in config['dynamic_numerical_columns']:
        train_df[col] = train_df[col].astype(float)
        max_value = train_df[col].abs().max()
        if max_value != 0:
            train_df[col] = train_df[col] / max_value
            val_df[col] = val_df[col] / max_value
            test_df[col] = test_df[col] / max_value

    # Normalize target label
    max_remtime = train_df['remaining_time'].max()
    train_df['remaining_time_normalized'] = train_df['remaining_time'] / max_remtime
    val_df['remaining_time_normalized'] = val_df['remaining_time'] / max_remtime
    test_df['remaining_time_normalized'] = test_df['remaining_time'] / max_remtime

    results_location = output_location + '/' + dataset_name + '/'
    Path(results_location).mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(results_location + '/train.pkl')
    val_df.to_pickle(results_location + '/val.pkl')
    test_df.to_pickle(results_location + '/test.pkl')
    joblib.dump(ohe_encoders, results_location + '/oh_encoders.pkl')
    joblib.dump(config, results_location + '/config.pkl')

    train_df[config['case_id_column']] = train_df[config['case_id_column']].astype(str)
    val_df[config['case_id_column']] = val_df[config['case_id_column']].astype(str)
    test_df[config['case_id_column']] = test_df[config['case_id_column']].astype(str)


def replace_categorical_outliers(df, case_id_column, col, perc_threshold, to_replace=None):
    """
    Replace values which occur in less than perc_threshold of cases with __OTHER__
    If to_replace is provided, only values in this list will be replaced
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the column where categorical outliers should be replaced
    case_id_column: str
        Name of the column containing the case ids.
    col: str
        Name of the column where categorical outliers should be replaced.
    perc_threshold: float
        Threshold for the percentage of cases a value should occur in to be replaced.
    to_replace: list
        List of values that should be replaced. If None, the function will determine the values that should be replaced
        based on the perc_threshold.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with replaced values
    to_replace: list
        List of values that were replaced

    """

    if to_replace is None:
        to_replace = list()
        unique_values = df[col].unique()
        counts = {u: 0 for u in unique_values}
        grouped = df.groupby(by=case_id_column)
        for index, group in grouped:
            unique_group_values = group[col].unique()
            for u in unique_group_values:
                counts[u] += 1

        num_cases = len(grouped)
        for key, count in counts.items():
            if count / num_cases < perc_threshold:
                to_replace.append(key)

    for value in to_replace:
        print('Replacing', col, value, 'with __OTHER__')
        df[col] = df[col].replace(value, '__OTHER__')
    return df, to_replace



def calculate_additional_features(df,
                                  config):
    """
    Calculate additional features for the dataset. Features include: day of the week, remaining time, time since midnight,
    time since sunday, time since last event, time since process started,
    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing the data for which additional features should be calculated.
    config: dict
        Configuration dictionary. keys: timestamp_column, case_id_column, dynamic_categorical_columns, dynamic_numerical_columns.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with additional features added.
    dict
        Config with features added.

    """

    # Prepare dataframe for calculation of additional features
    df[config['timestamp_column']] = pd.to_datetime(df[config['timestamp_column']], format='mixed',
                                                    infer_datetime_format=True)
    df = df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

    # Calculate target label
    df = feature_calculation.remaining_case_time(df=df,
                                                 timestamp_column=config['timestamp_column'],
                                                 case_id_column=config['case_id_column'],
                                                 new_column_name='remaining_time')

    # Calculate additional features
    df = feature_calculation.day_of_week(df=df,
                                         timestamp_column=config['timestamp_column'],
                                         new_column_name='day_of_week')
    config['dynamic_categorical_columns'].append('day_of_week')

    df = feature_calculation.time_since_midnight(df=df,
                                                 timestamp_column=config['timestamp_column'],
                                                 new_column_name='time_since_midnight')
    config['dynamic_numerical_columns'].append('time_since_midnight')

    df = feature_calculation.time_since_sunday(df=df,
                                               timestamp_column=config['timestamp_column'],
                                               new_column_name='time_since_sunday')
    config['dynamic_numerical_columns'].append('time_since_sunday')

    df = feature_calculation.time_since_last_event(df=df,
                                                   timestamp_column=config['timestamp_column'],
                                                   new_column_name='time_since_last_event',
                                                   case_id_column=config['case_id_column'])
    config['dynamic_numerical_columns'].append('time_since_last_event')

    df = feature_calculation.time_since_process_start(df=df,
                                                      timestamp_column=config['timestamp_column'],
                                                      new_column_name='time_since_start',
                                                      case_id_column=config['case_id_column'])
    config['dynamic_numerical_columns'].append('time_since_start')

    df[config['dynamic_numerical_columns']] = df[config['dynamic_numerical_columns']].fillna(0.)
    df[config['static_numerical_columns']] = df[config['static_numerical_columns']].fillna(0.)
    df[config['dynamic_categorical_columns']] = df[config['dynamic_categorical_columns']].fillna('__NO_VALUE_PROVIDED__')
    df[config['static_categorical_columns']] = df[config['static_categorical_columns']].fillna('__NO_VALUE_PROVIDED__')

    for cat_col in config['dynamic_categorical_columns']:
        df[cat_col] = df[cat_col].astype(str).astype(pd.CategoricalDtype())

    return df, config


def data_split(df,
               config):
    """
    Split the data into training, validation and test set.
    Parameters
    ----------
    df: pd.DataFrame
        Event log which should be split.
    config: dict
        Configuration dictionary. keys: case_id_column, timestamp_column.

    Returns
    -------
    pd.DataFrame
        Training set
    pd.DataFrame
        Validation set
    pd.DataFrame
        Test set

    """
    cases_sorted = df.groupby(by=config['case_id_column'])[
        config['timestamp_column']].min().sort_values().index.tolist()
    num_cases = len(cases_sorted)
    num_train_cases = int(num_cases * 0.6)
    num_val_cases = int(num_cases * 0.2)
    training_cases = cases_sorted[:num_train_cases]
    validation_cases = cases_sorted[num_train_cases:num_train_cases + num_val_cases]
    test_cases = cases_sorted[num_train_cases + num_val_cases:]

    train_df = df[df[config['case_id_column']].isin(training_cases)].sort_values(
        by=[config['case_id_column'], config['timestamp_column']])
    validation_df = df[df[config['case_id_column']].isin(validation_cases)].sort_values(
        by=[config['case_id_column'], config['timestamp_column']])
    test_df = df[df[config['case_id_column']].isin(test_cases)].sort_values(
        by=[config['case_id_column'], config['timestamp_column']])

    return train_df, validation_df, test_df


def one_hot_encoding(encoder,
                     value_to_encode):
    """
    One hot encode a value.
    Parameters
    ----------
    encoder: sklearn.preprocessing.OneHotEncoder
        Encoder to use for one hot encoding.
    value_to_encode: str
        Value to encode.

    Returns
    -------
    list
        One hot encoded feature vector.

    """
    idx = encoder.transform(np.asarray(str(value_to_encode)).reshape(1, -1)).astype(int)[0][0]
    if np.isnan(idx) or idx < 0:
        return [0] * len(encoder.categories_[0])
    else:
        L=[0] * len(encoder.categories_[0])
        L[idx]=1
        return L
