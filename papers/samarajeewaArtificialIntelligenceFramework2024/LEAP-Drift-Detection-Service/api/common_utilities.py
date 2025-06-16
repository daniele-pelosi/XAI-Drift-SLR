import pyodbc
from dynaconf import settings
import numpy as np
import pandas as pd

# Commented out IPython magic to ensure Python compatibility.
def get_db_connection():
    server = settings.get('DB_SERVER', '')
    database = settings.get('DB_NAME', '')
    username = settings.get('DB_USERNAME', '')
    password = settings.get('DB_PASSWORD', '')

    connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database \
        + ';UID=' + username + ';PWD=' + password
    return pyodbc.connect(connection_string)


def load_data(meter_key, predict_date):
    # load 6 months data
    connection = get_db_connection()

    daily_baseline_data = pd.read_sql("SELECT DateKey, SUM(Prediction) AS Prediction "
                                      "FROM BuildingBaseline2 "
                                      "WHERE MeterKey = %s "
                                      "AND DateKey BETWEEN CONVERT(VARCHAR(8), DATEADD(MONTH, -6, CONVERT(DATE, '%s', 112)), 112) AND '%s'"
                                      "GROUP BY DateKey ORDER BY DateKey"
                                      % (meter_key, predict_date, predict_date), connection)

    daily_actual_data = pd.read_sql("SELECT DateKey, SUM(MeterReading) AS Consumption FROM dbo.MeterReadings "
                                    "WHERE MeterReading > 0 "
                                    "AND (IsSuspicious = 0 OR IsSuspicious IS NULL) "
                                    "AND MeterKey = %s AND DateKey BETWEEN CONVERT(VARCHAR(8), DATEADD(MONTH, -6, CONVERT(DATE, '%s', 112)), 112) AND '%s'"
                                    "GROUP BY DateKey ORDER BY DateKey"
                                    % (meter_key, predict_date, predict_date), connection)

    combined_data = pd.merge(daily_baseline_data, daily_actual_data,
                             how="inner", left_on="DateKey", right_on="DateKey").fillna(method='ffill')

    combined_data["Error"] = np.abs(combined_data["Prediction"] - combined_data["Consumption"])

    return combined_data
