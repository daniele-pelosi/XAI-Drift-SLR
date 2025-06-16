import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from joblib import load
from dynaconf import settings
import tempfile

import logging
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.info("test")


def get_event_dict(connection, model_id):
    # Reading the events
    events_df = pd.read_sql("SELECT de.DateKey, de.Timestamp, de.EventId, de.EventName FROM dbo.DimEvent de "
                            "INNER JOIN dbo.ModelEventMappings mem ON de.EventId=mem.EventId WHERE mem.ModelId = %s"
                            % model_id, connection)

    event_dictionary = dict()

    for index, row in events_df.iterrows():
        event_date = datetime.strptime(str(row['DateKey']), '%Y%m%d')

        event_dictionary[row['EventName']] = (
            row['Timestamp'], event_date, row['EventId'], row['DateKey'])

    return event_dictionary


def get_test_data(connection, building_id, startDate, endDate, event_dictionary, resolution, alg):
    date = pd.read_sql("SELECT * FROM dbo.DimDate WHERE DateKey >= %s AND DateKey <= %s" % (startDate, endDate),
                       connection)
    time = pd.read_sql("SELECT * FROM dbo.DimTime", connection)

    st = "SELECT MeterKey,CampusKey FROM [Leap].[dbo].[DimMeter] WHERE MeterKey=%s" % building_id
    building_details = pd.read_sql(st, connection)

    if building_details.empty != True:
        campus_key = int(building_details['CampusKey'].iloc[0])
    else:
        campus_key = 1

    reading_temperature = pd.read_sql("""SELECT[DateKey]
              ,[TimeKey]
              ,[ApparentTemperature]
              ,[AirTemperature]
              ,[DewPointTemperature]
              ,[RelativeHumidity]
          	FROM [Leap].[dbo].[vwClimate] WHERE CampusKey= %s AND DateKey >= %s AND DateKey <= %s"""
                                      % (campus_key, startDate, endDate), connection)

    reading_temperature['Timestamp'] = pd.to_datetime(
        reading_temperature.DateKey.astype(
            str) + " " + reading_temperature['TimeKey'].astype(str).str.zfill(6),
        format="%Y%m%d %H%M%S")

    temp_range = pd.date_range(reading_temperature['Timestamp'].dt.date.min(axis=0),
                               reading_temperature['Timestamp'].max(axis=0),
                               freq='15min',
                               name="Timestamp").to_frame().reset_index(drop=True)
    temp_out = pd.merge(temp_range, reading_temperature, how="left", left_on="Timestamp", right_on="Timestamp").fillna(
        method='ffill')
    temp_out.TimeKey = temp_out.Timestamp.dt.strftime("%H%M%S").astype(int)
    temp_out.DateKey = temp_out.Timestamp.dt.strftime("%Y%m%d").astype(int)
    reading_temperature = temp_out.drop(columns=["Timestamp"])

    date_data = date.drop_duplicates(subset='DateKey', keep='first')
    date_data = date_data.drop(columns=['Date', 'DaySuffix', 'WeekDayName', 'HolidayText', 'DayOfYear',
                                        'ISOWeekOfYear', 'MonthName', 'QuarterName', 'MMYYYY', 'MonthYear',
                                        'FirstDayOfMonth', 'LastDayOfMonth', 'FirstDayOfQuarter', 'LastDayOfQuarter',
                                        'FirstDayOfYear', 'LastDayOfYear', 'FirstDayOfNextMonth', 'FirstDayOfNextYear',
                                        'IsSemester', 'IsExamPeriod', 'CalendarSignificance',
                                        'HasCalendarSignificance'])
    date_data["IsWeekend"] = date_data["IsWeekend"].astype(int)
    date_data["IsHoliday"] = date_data["IsHoliday"].astype(int)

    # preprocess the time data
    time_data = time.drop_duplicates(subset='TimeKey', keep='first')
    time_data = time_data.drop(
        columns=['Hour24ShortString', 'Hour24FullString', 'Hour24MinString', 'Hour12', 'Hour12ShortString',
                 'Hour12MinString',
                 'Hour12FullString', 'AmPmString', 'MinuteCode', 'MinuteShortString', 'MinuteFullString24',
                 'MinuteFullString12', 'HalfHourShortString', 'HalfHourCode',
                 'HalfHourFullString12', 'SecondShortString', 'Second', 'FullTimeString12',
                 'FullTime'])

    # merge temperature data with date and time data
    all_data = pd.merge(left=reading_temperature, right=date_data,
                        how='left', left_on='DateKey', right_on='DateKey')
    all_data = pd.merge(left=all_data, right=time_data,
                        how='left', left_on='TimeKey', right_on='TimeKey')

    # assign training data to all before visualization
    data_training_all = all_data
    data_training_all.sort_values(by=['DateKey', 'TimeKey'], inplace=True)

    CDDBASE = 20
    HDDBASE = 18

    log.info("processing building " + str(building_id) +
             " " + resolution + " " + alg)

    data_training = data_training_all
    data_training = data_training.replace(r'^\s*$', np.nan, regex=True)
    data_training.fillna(method='bfill', inplace=True)
    data_training = data_training.dropna()

    if len(data_training) == 0:
        # return
        raise Exception("No training data!")

    data_training["Timestamp"] = pd.to_datetime(
        (data_training["DateKey"].astype(str) +
         data_training["FullTimeString24"].astype(str)),
        format='%Y%m%d%H:%M:%S')
    data_training["Time"] = data_training["Timestamp"]
    data_training = data_training.set_index('Time')

    # add cdd and hdd to dataframe
    data_training["ApparentTemperature"] = pd.to_numeric(
        data_training["ApparentTemperature"])
    data_training["cdd"] = np.where(data_training["ApparentTemperature"] > CDDBASE,
                                    data_training["ApparentTemperature"] - CDDBASE, 0)
    data_training["hdd"] = np.where(data_training["ApparentTemperature"] < HDDBASE,
                                    HDDBASE - data_training["ApparentTemperature"], 0)

    # add CDD and HDD to dataframe
    dd = data_training["cdd"].resample("D").mean()
    data_training["CDD"] = -1
    for d in dd.index:
        data_training["CDD"] = np.where((data_training.index.date == d.date()) | (data_training["CDD"] < 0),
                                        dd[dd.index == d].values[0],
                                        data_training["CDD"])

    dd = data_training["hdd"].resample("D").mean()
    data_training["HDD"] = -1
    for d in dd.index:
        data_training["HDD"] = np.where((data_training.index.date == d.date()) | (data_training["HDD"] < 0),
                                        dd[dd.index == d].values[0],
                                        data_training["HDD"])

    data_training = data_training.set_index('Timestamp')
    data_training["RelativeHumidity"] = pd.to_numeric(
        data_training["RelativeHumidity"])
    data_training["HDD"] = pd.to_numeric(data_training["HDD"])
    data_training["CDD"] = pd.to_numeric(data_training["CDD"])

    events = []
    for key, value in event_dictionary.items():
        (timestamp, event_date, event_id, datekey) = event_dictionary[key]
        data_training[key] = 0
        data_training[key] = np.where(
            (data_training["DateKey"] >= datekey), 1, 0)
        events.append(key)

    if len(events) > 1:
        events = sorted(events)

    data_training = data_training.sample(frac=1)

    categorial = ['IsWeekend', 'IsHoliday', 'HalfHour'] + events
    nvariables = ["ApparentTemperature", "RelativeHumidity", "HDD", "CDD",
                  'Weekday', 'Hour24', 'Minute']

    log.info("Model will be generated for the building " +
             str(building_id) + 'included features')
    log.info("Categorical variables: %s", categorial)
    log.info("Continuous variables: %s", nvariables)

    log.debug(data_training)
    data_training.sort_values(by=['DateKey', 'TimeKey'], inplace=True)
    X_train_ndata = data_training[nvariables].values
    X_train_cdata = data_training[categorial].values
    if len(nvariables) == 1:
        X_train_ndata = X_train_ndata.reshape(X_train_ndata.shape[0], 1)

    X_train = np.hstack((X_train_cdata, X_train_ndata))
    X_train = X_train.astype(float)

    return X_train, data_training['DateKey'].values, data_training['TimeKey'].values


def load_dump_model(model_file_location):
    model_params = load(model_file_location)

    if len(model_params) == 5:
        reg, finalscore, r2, mse, mape = model_params
        start_date = ""
        end_date = "20200310"
    else:
        reg, finalscore, r2, mse, mape, start_date, end_date = model_params

    return reg, finalscore, r2, mse, mape, start_date, end_date


def get_model_ids(connection, building_id, baseline_type):
    if baseline_type == 'BASELINE':
        model_details = pd.read_sql(
            "SELECT TOP 1 ModelId FROM dbo.Model WHERE MeterKey = %s AND Type='BASELINE' ORDER BY CreatedDate DESC"
            % building_id, connection)
        return model_details['ModelId'].values
    elif baseline_type == 'M_AND_V':
        model_details = pd.read_sql(
            "SELECT ModelId FROM dbo.Model WHERE MeterKey = %s AND Type='M_AND_V' ORDER BY CreatedDate ASC" % building_id, connection)
        return model_details['ModelId'].values

    return []

def download_model(connection, model_id):
    model_details = pd.read_sql("SELECT * FROM dbo.Model WHERE ModelId = %s" % model_id, connection)

    if len(model_details) == 0:
        raise Exception(f'Model Details not found for model: {model_id}')

    model_name = model_details['ModelName'].iloc[0]

    dirname = os.path.dirname(__file__)
    model_store_location = tempfile.gettempdir()
    model_store_location = os.path.join(dirname, model_store_location)
    model_file_location = os.path.join(model_store_location, model_name)

    reg = None
    if not os.path.exists(model_file_location):
        log.info("Model path does not exist")
        # if model isn't exist in the local repository, then download from the blob storage
        try:
            log.debug("Download blob path does not exist")

            account_key = settings.get('AZURE_ACCOUNT_KEY', '')

            blob_connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapmodels;' \
                'AccountKey=' + account_key + 'EndpointSuffix=core.windows.net'

            blob_service_client = BlobServiceClient.from_connection_string(
                blob_connection_str)

            container_name = 'models'

            full_path_to_file = os.path.join(
                model_store_location, model_name)

            log.info("\nDownloading blob to " + full_path_to_file)

            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=model_name)

            with open(full_path_to_file, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            try:
                reg, finalscore, r2, mse, mape, start_date, end_date = load_dump_model(
                    model_file_location)
            except:
                log.info("Model can't be loaded from the file location " + model_file_location +
                         ' for the model ' + str(model_id))
                pass
        except Exception as ex:
            log.error(
                'Exception occurred while downloading the model: ' + str(model_id))
            log.error(ex)
    else:
        log.info("\tLoad existing models ...")
        reg, finalscore, r2, mse, mape, start_date, end_date = load_dump_model(model_file_location)

    return reg, model_name, start_date, end_date


def upload_forecast_data(file_name, file_path):
    try:
        account_key = settings.get('AZURE_ACCOUNT_KEY_BASELINE', '')
        blob_connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapbaselinedata;' \
            'AccountKey=' + account_key + 'EndpointSuffix=core.windows.net'
        blob_service_client = BlobServiceClient.from_connection_string(
            blob_connection_str)
        container_name = 'baselines'

        try:
            # Attempt to create container
            blob_service_client.create_container(container_name)
        # Catch exception and print error
        except ResourceExistsError as error:
            log.debug("Container exist, hence ignore")
            log.debug(error)

        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name,
                                                          blob=file_name)
        log.info("\nUploading to Azure Storage as blob:\n\t" + file_name)

        # Upload the created file
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    except Exception as ex:
        log.error(
            'Exception occurred while uploading to blob storage, file: ' + file_name)
        raise ex
