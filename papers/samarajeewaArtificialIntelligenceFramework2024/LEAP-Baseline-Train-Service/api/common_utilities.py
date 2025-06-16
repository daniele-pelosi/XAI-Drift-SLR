import pyodbc
from dynaconf import settings
import numpy as np


# Commented out IPython magic to ensure Python compatibility.
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

def results_to_data_frame(building_id, timestamp, old_pred, new_pred, actual):
    dump_csv = pd.DataFrame()
    dump_csv['Timestamp'] = timestamp
    dump_csv['Old Baseline'] = old_pred
    dump_csv['New Baseline'] = new_pred
    dump_csv['Actual'] = actual
    dump_csv['MeterKey'] = building_id
    return dump_csv
