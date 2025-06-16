import pyodbc
from dynaconf import settings
from datetime import datetime
import pandas as pd

def get_db_connection():
    server = settings.get('DB_SERVER', '')
    database = settings.get('DB_NAME', '')
    username = settings.get('DB_USERNAME', '')
    password = settings.get('DB_PASSWORD', '')

    connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database \
        + ';UID=' + username + ';PWD=' + password
    return pyodbc.connect(connection_string)


def retrieve_pending_model_train_processes(building_ids):
    conn = get_db_connection()

    # Get current date
    current_date = int(datetime.now().strftime('%Y%m%d'))

    building_ids_str = [str(id) for id in building_ids]

    # Query to retrieve pending processes
    query = f" SELECT * FROM [dbo].[ModelTrainProcess] " \
            f"WHERE [Finished] = 0 AND [TrainEndDate] < {current_date} " \
            f"AND [MeterKey] IN ({', '.join(building_ids_str)})"

    # Execute the query and read results into a DataFrame
    pending_processes_df = pd.read_sql(query, conn)

    # Close the connection
    conn.close()

    return pending_processes_df

def mark_train_process_as_finished(process_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # SQL statement to update Finished column
    query = "UPDATE [dbo].[ModelTrainProcess] SET [Finished] = 1 WHERE [ProcessId] = ?"

    try:
        # Execute the update query
        cursor.execute(query, (process_id,))
        # Commit the transaction
        conn.commit()
        print(f"Process with ProcessId {process_id} marked as finished.")
    except pyodbc.Error as e:
        # Handle error
        print(f"Error occurred: {e}")
        # Rollback the transaction
        conn.rollback()
    finally:
        # Close cursor and connection
        cursor.close()
        conn.close()
