from datetime import datetime
import numpy as np
import ruptures as rpt
import pandas as pd

from common_utilities import get_db_connection, load_data


def get_model_id(meter_key):
    connection = get_db_connection()

    model_details = pd.read_sql(
        "SELECT TOP 1 ModelId FROM dbo.Model WHERE MeterKey = %s AND Type='BASELINE' ORDER BY CreatedDate DESC"
        % meter_key, connection)

    connection.close()
    return int(model_details['ModelId'].values[0])


def get_change_points(meter_key, model_id, end_date_key):
    connection = get_db_connection()

    end_date_key = str(end_date_key)

    change_points = pd.read_sql(
        "SELECT DateKey FROM dbo.BaselineDriftData "
        "WHERE MeterKey = %s AND ModelId = %s "
        "AND DateKey BETWEEN CONVERT(VARCHAR(8), DATEADD(MONTH, -6, CONVERT(DATE, '%s', 112)), 112) AND %s "
        "ORDER BY DateKey DESC"
        % (meter_key, model_id, end_date_key, end_date_key), connection)

    connection.close()

    return list(change_points['DateKey'].values)


def find_change_points(data_df, change_points):
    drifted_change_points_set = set()
    non_drifted_change_points_set = set()

    data = data_df['Error'].values
    data[data == 0] = np.maximum.accumulate(data)[data == 0]

    # Run PELT algorithm on the data
    algo = rpt.Pelt(model="rbf").fit(data)
    result = algo.predict(pen=5)
    if result[-1] == len(data):
        del result[-1]

    i = len(data)

    for r in result:
        if i - r < 15 or i - r > 30:
            # avoid predicting older drifts than 30 days and less than 15 days
            continue

        date_key = int(data_df.loc[r]['DateKey'])

        # dates within a range +-10 days are considered same
        if not any(
                abs(datetime.strptime(str(cp), '%Y%m%d') - datetime.strptime(str(date_key), '%Y%m%d')).days <= 10 for cp
                in change_points):

            drifted = np.mean(data[r - (i - r):r]) < np.mean(data[r:i]) or np.mean(data) <= np.mean(data[r:i])

            change_points.append(date_key)

            if drifted:
                drifted_change_points_set.add(date_key)
            else:
                non_drifted_change_points_set.add(date_key)

    return list(drifted_change_points_set), list(non_drifted_change_points_set)


def add_drifted_change_points(meter_key, model_id, change_points):
    connection = get_db_connection()

    for date_key in change_points:
        sql_query = "INSERT INTO dbo.BaselineDriftData (MeterKey, ModelId, DateKey, CreatedAt, ResumeAt) " \
                    "VALUES (?, ?, ?, ?, ?)"
        connection.execute(sql_query, meter_key, model_id, date_key, datetime.now(), datetime.now())

    connection.commit()
    connection.close()


def add_non_drifted_change_points(meter_key, model_id, change_points):
    connection = get_db_connection()

    for date_key in change_points:
        sql_query = "INSERT INTO dbo.BaselineDriftData (MeterKey, ModelId, DateKey, CreatedAt, ResumeAt, Drifted) " \
                    "VALUES (?, ?, ?, ?, ?, 0)"
        connection.execute(sql_query, meter_key, model_id, date_key, datetime.now(), datetime.now())

    connection.commit()
    connection.close()


def initiate_drift_detection(meter_key, end_date_key):

    # avoid incremental meters
    if meter_key == 116 or meter_key == 111 or meter_key == 635:
        return

    new_change_points = None

    data_df = load_data(meter_key, end_date_key)

    if len(data_df) != 0:
        model_id = get_model_id(meter_key)

        current_change_points = get_change_points(meter_key, model_id, end_date_key)

        drifted_change_points, non_drifted_change_points = find_change_points(data_df, current_change_points)

        add_drifted_change_points(meter_key, model_id, drifted_change_points)
        add_non_drifted_change_points(meter_key, model_id, non_drifted_change_points)

    return new_change_points
