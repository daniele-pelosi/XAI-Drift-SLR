import pyodbc
from dynaconf import settings
from datetime import datetime
import logging

log = logging.getLogger()
log.setLevel(logging.INFO)


def get_db_connection():
    server = settings.get('DB_SERVER', '')
    database = settings.get('DB_NAME', '')
    username = settings.get('DB_USERNAME', '')
    password = settings.get('DB_PASSWORD', '')

    connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database \
                        + ';UID=' + username + ';PWD=' + password
    return pyodbc.connect(connection_string)


def insert_model_train_process(building_id, model_type, start_date, end_date):
    log.info(
        f"Scheduling a model train process for building: {building_id} of type: {model_type} "
        f"from {start_date} to {end_date}")

    connection = get_db_connection()

    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO [dbo].[ModelTrainProcess] ([MeterKey],[Type],[TrainStartDate],[TrainEndDate],[CreatedAt]) "
            "VALUES (?,?,?,?,?)", building_id, model_type, start_date, end_date, datetime.now())

        connection.commit()
    except pyodbc.Error as e:
        connection.rollback()
        log.error(
            "Error while adding model train info to the database for building " + str(building_id))
        raise e
