import pandas as pd
from cml import bucketing, encoding
from sklearn.pipeline import Pipeline, FeatureUnion
from cml.ClassifierWrapper import ClassifierWrapper
import numpy as np


def generate_prefix_data(df,
                         min_length,
                         config,
                         max_length=None):
    """
    Generate prefix data (each possible prefix becomes a trace)

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with event log data.
    min_length: int
        Minimum length for prefixes to create.
    config: dict
        Configuration dictionary.
    max_length: int
        Maximum length for prefixes to create.

    Returns
    -------
    pd.DataFrame
        DataFrame with prefix data.

    """
    grouped = df.groupby(by=config['case_id_column'])
    if max_length is None:
        max_length = grouped.size().max()
    df['case_length'] = grouped[config['activity_column']].transform(len)
    dt_prefixes = df[df['case_length'] >= min_length].groupby(config['case_id_column']).head(min_length)
    dfs_to_concat = list()
    dfs_to_concat.append(dt_prefixes)
    for nr_events in range(min_length + 1, max_length + 1):
        tmp = df[df['case_length'] >= nr_events].groupby(config['case_id_column']).head(nr_events)
        tmp[config['case_id_column']] = tmp[config['case_id_column']].apply(lambda x: "%s__--__%s" % (x, nr_events))
        dfs_to_concat.append(tmp)
    dt_prefixes = pd.concat(dfs_to_concat, axis=0)
    dt_prefixes['case_length'] = dt_prefixes.groupby(config['case_id_column'])[config['activity_column']].transform(len)
    return dt_prefixes


def fit_bucketing(dt_train_prefixes,
                  bucketing_method,
                  config,
                  n_clusters,
                  random_state):
    """
    Fit a bucketing method.

    Parameters
    ----------
    dt_train_prefixes: pd.DataFrame
    bucketing_method: str
    config: dict
    n_clusters: int
    random_state: int

    Returns
    -------
    object
        The fitted bucketer.
    """

    # Code taken from Verenich paper
    if bucketing_method == "state":
        bucket_encoding = "last"
    else:
        bucket_encoding = "agg"

    bucketing_args = {}
    bucketing_args['case_id_col'] = config['case_id_column']
    bucketing_args['encoding_method'] = bucket_encoding

    categorical_columns = list()
    categorical_columns.extend(config['static_categorical_columns'])
    categorical_columns.extend(config['dynamic_categorical_columns'])

    numerical_columns = list()
    numerical_columns.extend(config['static_numerical_columns'])
    numerical_columns.extend(config['dynamic_numerical_columns'])

    bucketing_args['cat_cols'] = categorical_columns
    bucketing_args['num_cols'] = numerical_columns
    bucketing_args['n_clusters'] = n_clusters
    bucketing_args['random_state'] = random_state

    bucketer = bucketing.get_bucketer(bucketing_method, **bucketing_args)
    bucketer = bucketer.fit(dt_train_prefixes)

    return bucketer


def get_indexes(data,
                config):
    """
    Get indexes of cases.

    Parameters
    ----------
    data: pd.DataFrame
    config: dict

    Returns
    -------
    pd.Index
        Index for each case id.
    """
    return data.groupby(config['case_id_column']).first().index


def get_relevant_data_by_indexes(data, indexes, config):
    """
    Get relevant data by indexes.

    Parameters
    ----------
    data: pd.DataFrame
    indexes: pd.Index
    config: dict

    Returns
    -------
    pd.DataFrame

    """
    return data[data[config['case_id_column']].isin(indexes)]


def get_label(data,
              label_col,
              config):
    """
    Get label for each prefix.

    Parameters
    ----------
    data: pd.DataFrame
    label_col: str
    config: dict

    Returns
    -------
    pd.Series

    """
    data = data[data['prefix_len'] > 1]
    data = data.sort_values(by=[config['case_id_column'],
                                config['timestamp_column']]).groupby(config['case_id_column']).last()[label_col]
    return data


def fit_pipelines(dt_train_prefixes,
                  bucketer,
                  config,
                  encoding_methods,
                  encoding_args,
                  cls_method,
                  cls_args):
    """
    Fit pipelines.

    Parameters
    ----------
    dt_train_prefixes: pd.DataFrame
    bucketer: object
    config: dict
    encoding_methods: list
    encoding_args: dict
    cls_method: object
    cls_args: dict

    Returns
    -------
    dict
        Key: int, Value: object. Key indicated the bucket, value the pipeline.

    """
    bucket_assignments_train = bucketer.predict(dt_train_prefixes)
    pipelines = {}
    for bucket in set(bucket_assignments_train):
        relevant_cases_bucket = get_indexes(data=dt_train_prefixes,
                                            config=config)[bucket_assignments_train == bucket]
        dt_train_bucket = get_relevant_data_by_indexes(data=dt_train_prefixes,
                                                       indexes=relevant_cases_bucket,
                                                       config=config)  # one row per event
        train_y = get_label(data=dt_train_bucket,
                            label_col='remaining_time',
                            config=config)

        feature_combiner = FeatureUnion(
            [(method,
              encoding.get_encoder(method, **encoding_args)) for method in encoding_methods])

        pipelines[bucket] = Pipeline(
            [('encoder', feature_combiner),
             ('cls', ClassifierWrapper(cls_method(**cls_args)))])

        pipelines[bucket].fit(dt_train_bucket, train_y)

        return pipelines

def predict_data(dt_prefixes,  # Prefixes to predict
                 pipelines,
                 train_df,  # DataFrame with training data. Needed to calculate mean of remaining time.
                 max_prefix_length,
                 min_prefix_length,
                 bucketer,
                 config,
                 bucketing_method
                 ):
    """
    Predict data.

    Parameters
    ----------
    dt_prefixes: pd.DataFrame
    pipelines: dict
    train_df: pd.DataFrame
    max_prefix_length: int
    min_prefix_length: int
    bucketer: object
    config: dict
    bucketing_method: str

    Returns
    -------
    pd.DataFrame
    """

    # if the bucketing is prefix-length-based, then predict for each prefix length separately, otherwise predict all prefixes together
    max_evaluation_prefix_length = max_prefix_length if bucketing_method == "prefix" else min_prefix_length
    prefix_lengths = dt_prefixes.groupby(config['case_id_column']).size()

    preds = []
    y = []
    y_ = []

    for nr_events in range(min_prefix_length, max_evaluation_prefix_length + 1):

        if bucketing_method == "prefix":
            # select only prefixes that are of length nr_events
            relevant_cases_nr_events = prefix_lengths[prefix_lengths == nr_events].index

            if len(relevant_cases_nr_events) == 0:
                break

            dt_nr_events = get_relevant_data_by_indexes(data=dt_prefixes,
                                                        indexes=relevant_cases_nr_events,
                                                        config=config)
            del relevant_cases_nr_events
        else:
            # evaluate on all prefixes
            dt_nr_events = dt_prefixes.copy()

        # get predicted cluster for each test case
        bucket_assignments = bucketer.predict(dt_nr_events)

        # use appropriate classifier for each bucket of test cases
        # for evaluation, collect predictions from different buckets together
        for bucket in set(bucket_assignments):

            relevant_cases_bucket = get_indexes(data=dt_nr_events,
                                                config=config)[bucket_assignments == bucket]
            dt_bucket = get_relevant_data_by_indexes(data=dt_nr_events,
                                                     indexes=relevant_cases_bucket,
                                                     config=config)  # one row per event

            if len(relevant_cases_bucket) == 0:
                continue

            elif bucket not in pipelines:
                # use mean remaining time (in training set) as prediction
                preds_bucket = np.array([np.mean(train_df["FEAT_REMAINING_TIME"])] * len(relevant_cases_bucket))

            else:
                # make actual predictions
                preds_bucket = pipelines[bucket].predict_proba(dt_bucket)

            preds_bucket = preds_bucket.clip(min=0)  # if remaining time is predicted to be negative, make it zero
            preds.extend(preds_bucket)

            # extract actual label values
            y_bucket = get_label(data=dt_bucket, label_col='remaining_time', config=config)  # one row per case
            y.append(y_bucket)
            y_.extend(y_bucket)

    df = pd.DataFrame(pd.concat(y, axis=0))
    df.columns = ['labels']
    df['preds'] = preds

    return df


def set_case_prefix_identifier(df,
                               config):
    """
    Set case prefix identifier.

    Parameters
    ----------
    df: pd.DataFrame
    config: dict

    Returns
    -------
    pd.DataFrame
    """
    df['prefix_len'] = df.groupby(config['case_id_column']).cumcount() + 1
    df['case_prefix_identifier'] = df[config['case_id_column']].astype(str) + '_' + \
                                                df['prefix_len'].astype(str)
    df['case_prefix_identifier'] = np.where(
        df['case_prefix_identifier'].str[-2:] == '_1',
        df[config['case_id_column']],
        df['case_prefix_identifier'])
    return df


def train(train_df,
          val_df,
          config,
          bucketing_method,
          random_state,
          encoding_methods,
          encoding_args,
          cls_method,
          cls_args):
    """
    Train model.

    Parameters
    ----------
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    config: dict
    bucketing_method: str
    random_state: int
    encoding_methods: list
    encoding_args: dict
    cls_method: sklearn.base.BaseEstimator
    cls_args: dict

    Returns
    -------
    pd.DataFrame
        Prediction on validation set.

    """
    train_df = set_case_prefix_identifier(df=train_df, config=config)
    val_df = set_case_prefix_identifier(df=val_df, config=config)

    max_length = train_df.groupby(by=config['case_id_column']).size().max()

    dt_train_prefixes = generate_prefix_data(df=train_df,
                                             min_length=2,
                                             max_length=max_length,
                                             config=config)

    bucketer = fit_bucketing(dt_train_prefixes=dt_train_prefixes,
                             bucketing_method=bucketing_method,
                             config=config,
                             n_clusters=None,
                             random_state=random_state)

    pipelines = fit_pipelines(dt_train_prefixes=dt_train_prefixes,
                              bucketer=bucketer,
                              config=config,
                              encoding_methods=encoding_methods,
                              encoding_args=encoding_args,
                              cls_method=cls_method,
                              cls_args=cls_args)

    # Predict validation data
    max_length_val = val_df.groupby(by=config['case_id_column']).size().max()
    dt_validation_prefixes = generate_prefix_data(df=val_df,
                                                  min_length=2,
                                                  max_length=max_length,
                                                  config=config)
    validation_preds_df = predict_data(dt_prefixes=dt_validation_prefixes,
                                       pipelines=pipelines,
                                       train_df=train_df,
                                       max_prefix_length=max_length_val,
                                       min_prefix_length=1,
                                       bucketer=bucketer,
                                       bucketing_method=bucketing_method,
                                       config=config)

    dt_validation_prefixes = dt_validation_prefixes[dt_validation_prefixes['prefix_len'] > 1]
    grouped = dt_validation_prefixes.sort_values(by=[config['case_id_column'],
                                                     config['timestamp_column']]).groupby(config['case_id_column'])
    validation_preds_df['case:concept:name'] = grouped.last()['case_prefix_identifier']
    validation_preds_df['concept:name'] = grouped.last()[config['activity_column']]
    validation_preds_df['time:timestamp'] = grouped.last()[config['timestamp_column']]

    return validation_preds_df
