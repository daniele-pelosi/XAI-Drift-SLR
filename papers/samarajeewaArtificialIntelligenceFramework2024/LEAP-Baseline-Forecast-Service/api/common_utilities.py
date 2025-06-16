import pyodbc
from dynaconf import settings
import numpy as np
import pandas as pd
import logging as log

def get_db_connection():
    server = settings.get('DB_SERVER', '')
    database = settings.get('DB_NAME', '')
    username = settings.get('DB_USERNAME', '')
    password = settings.get('DB_PASSWORD', '')

    connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database \
        + ';UID=' + username + ';PWD=' + password
    return pyodbc.connect(connection_string)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def results_to_data_frame(building_id, model_id, date_key, time_key, pred):
    dump_csv = pd.DataFrame()
    dump_csv['DateKey'] = date_key
    dump_csv['TimeKey'] = time_key
    dump_csv['Baseline'] = pred
    dump_csv['MeterKey'] = building_id
    dump_csv['ModelId'] = model_id
    return dump_csv

def get_file_name(building_ids, baseline_type, start_date, end_date):
    if len(building_ids) > 1:
        file_name = f'all_buildings_forecast_{start_date}_{end_date}.csv'
    else:
        file_name = f'{building_ids[0]}_forecast_{start_date}_{end_date}.csv'

    file_name = f'{baseline_type}_{file_name}'

    return file_name

def get_attribute(message, attribute_name):
    """
    Get the value of the attribute from the message dictionary or return None if not present.
    """
    try:
        return message[attribute_name]
    except KeyError:
        log.warning("%s attribute not found in the message. Setting it to None.", attribute_name)
        return None
