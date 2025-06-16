from typing import Optional, Any
import numpy as np
import pandas as pd


def calculate_unix_timestamp(df: pd.DataFrame,
                             timestamp_column: str,
                             new_column_name: str):
    """
    Calculates the unix timestamp for a given column in a pandas DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        The pandas dataframe which contains a columns for which the unix timestamp is needed.
    timestamp_column: str
        The column for which a unix timestamp should be calculated.
    new_column_name: str
        This column will be added and contains the unix timestamps.

    Returns
    -------
    pd.DataFrame#
        DataFrame with an additional column which contains the unix timestamps.

    """
    try:
        df[new_column_name] = (df[timestamp_column] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
    except TypeError:
        df[new_column_name] = (df[timestamp_column] - pd.Timestamp(
            "1970-01-01").tz_localize(df[timestamp_column].dt.tz)) / pd.Timedelta('1s')
    return df


def day_of_week(df: pd.DataFrame,
                timestamp_column: Optional[str] = None,
                new_column_name: str = 'DayOfWeek',
                nan_fill_value: Any = 7.):
    """
    Set the day of the week.

    Parameters
    ----------
    df: pd.Dataframe
        Dataframe which contains timestamps for which the day of the week should be extracted.
    timestamp_column: str
        The column for which to extract the day of the week.
    new_column_name: str
        Column name to be added which contains the day of the week in integer format.
    nan_fill_value: Any
        Value to be provided if day of week cannot be determined.
    Returns
    -------
    pd.DataFrame
        Dataframe containing a column with the day of the week.

    """
    df[timestamp_column] = df[timestamp_column].replace({'0': pd.NaT}) # Implies that missing values are encoded as 0.
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='mixed', infer_datetime_format=True)
    df[new_column_name] = df[timestamp_column].dt.dayofweek
    df[new_column_name] = df[new_column_name].fillna(nan_fill_value)
    return df


def remaining_case_time(df: pd.DataFrame,
                        timestamp_column: str,
                        case_id_column: str,
                        new_column_name: str):
    """
    Add a new column to a Dataframe which indicates the remaining time of a case in seconds. Events for a case are expected
    to be sorted by timestamp already.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to which the remaining time for each event will be added as a new column.
    timestamp_column: str
        The column which is used to calculate the remaining time.
    case_id_column: str
        Column which indicates the case id.
    new_column_name: str
        Name of the column to be added for indicating the remaining time.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional column that indicates the remaining time.

    """
    df = calculate_unix_timestamp(df=df,
                                  new_column_name='__UnixTimestamp__',
                                  timestamp_column=timestamp_column)

    df['__Timeshifted__'] = df.groupby(by=case_id_column)['__UnixTimestamp__'].shift()
    df['__Timeshifted__'] = np.where(df['__Timeshifted__'].isna(),
                                     df['__UnixTimestamp__'],
                                     df['__Timeshifted__'])
    df['__TimeToPreviousActivity__'] = df['__UnixTimestamp__'] - df[
        '__Timeshifted__']

    df['__NextActivityTimeLabel__'] = df.groupby(by=case_id_column)[
        '__TimeToPreviousActivity__'].shift(-1)
    df['__NextActivityTimeLabel__'] = df['__NextActivityTimeLabel__'].fillna(0)

    new_dfs = []
    for index, group in df.groupby(by=case_id_column):
        group[new_column_name] = group.loc[::-1, '__NextActivityTimeLabel__'].cumsum()[::-1]
        new_dfs.append(group)
    df = pd.concat(new_dfs)

    # Remove intermediate columns
    df = df.drop('__UnixTimestamp__', axis=1)
    df = df.drop('__Timeshifted__', axis=1)
    df = df.drop('__TimeToPreviousActivity__', axis=1)
    df = df.drop('__NextActivityTimeLabel__', axis=1)

    return df


def time_since_midnight(df: pd.DataFrame,
                        timestamp_column: str,
                        new_column_name: str):
    """
    Calculate the time since midnight in seconds for a timestamp column of a Dataframe.

    Parameters
    ----------
    df: pd.Dataframe
        Input dataframe to which an additional column indicating the time since midnight is added.
    timestamp_column: str
        The column which is used to calculate the time since midnight.
    new_column_name: str
        The name of the column to be added for indicating the time since midnight.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional column that indicates the time since midnight.

    """
    df[timestamp_column] = df[timestamp_column].replace({'0': pd.NaT}) # Implies that missing values are encoded as 0.
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='mixed', infer_datetime_format=True)
    df[new_column_name] = (df[timestamp_column] - df[timestamp_column].dt.normalize()) / pd.Timedelta('1 second')
    return df


def time_since_sunday(df: pd.DataFrame,
                      timestamp_column: str,
                      new_column_name: str):
    """
    Add a new column to a Dataframe to indicate the time since sunday midnight in seconds given a timestamp column.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to which the time since sunday is added as a new column.
    timestamp_column: str
        Column of the input dataframe which is used to calculate the time since sundy.
    new_column_name: str
        Column to be added which contains the time since sunday.

    Returns
    -------
    pd.DataFrame
        Input Dataframe with an additional column that indicates the time since sunday.

    """
    df[timestamp_column] = df[timestamp_column].replace({'0': pd.NaT}) # Implies that missing values are encoded as 0.
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format='mixed', infer_datetime_format=True)

    df['__TIME_PAYLOAD_DayOfWeek__'] = df[timestamp_column].dt.dayofweek

    df['__TIME_PAYLOAD_TimeSinceMidnight__'] = (df[timestamp_column] - df[
        timestamp_column].dt.normalize()) / pd.Timedelta(
        '1 second')

    df[new_column_name] = df['__TIME_PAYLOAD_TimeSinceMidnight__'] + (
            df['__TIME_PAYLOAD_DayOfWeek__'] * 3600 * 24)

    # Clean columns
    df = df.drop(['__TIME_PAYLOAD_TimeSinceMidnight__',
                  '__TIME_PAYLOAD_DayOfWeek__'], axis=1)

    return df


def time_since_last_event(df: pd.DataFrame,
                          timestamp_column: str,
                          new_column_name: str,
                          case_id_column: str):
    """
    Add the time since the last event of a case to a dataframe in seconds.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to which to add a new column that indicates the time since the previous event.
    timestamp_column: str
        Column which is used to calculate the time difference to the previous event.
    new_column_name: str
        Name of the column to be added to the input dataframe.
    case_id_column: str
        Column indicating the case id.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional column that indicates the time to the previous event.

    """
    df = calculate_unix_timestamp(df=df,
                                  new_column_name='__UnixTimestamp__',
                                  timestamp_column=timestamp_column)

    df['__Timeshifted__'] = df.groupby(by=case_id_column)[
        '__UnixTimestamp__'].shift()
    df['__Timeshifted__'] = np.where(df['__Timeshifted__'].isna(),
                                     df['__UnixTimestamp__'],
                                     df['__Timeshifted__'])
    df[new_column_name] = df['__UnixTimestamp__'] - df[
        '__Timeshifted__']

    df = df.drop('__UnixTimestamp__', axis=1)
    df = df.drop('__Timeshifted__', axis=1)

    return df


def time_since_process_start(df: pd.DataFrame,
                             timestamp_column: str,
                             new_column_name: str,
                             case_id_column: str):
    """
    Add a new column to a dataframe to indicate the time that has passed since a case started in seconds.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe to which the time since the process start is added.
    timestamp_column: str
        The name of the column that is used to calculate the time since the case started.
    new_column_name: str
        Name of the column to be added.
    case_id_column: str
        Name of the column that indicates the case id.

    Returns
    -------
    pd.DataFrame
        Input dataframe with an additional column which indicates the time in seconds since the process started.

    """
    dfs = []
    for case_id, group_df in df.groupby(by=case_id_column):
        group_df = calculate_unix_timestamp(df=group_df,
                                            new_column_name='__UnixTimestamp__',
                                            timestamp_column=timestamp_column)
        group_df[new_column_name] = group_df['__UnixTimestamp__'] - group_df['__UnixTimestamp__'].min()
        dfs.append(group_df)
    return pd.concat(dfs)
