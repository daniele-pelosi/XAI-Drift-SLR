import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from joblib import dump
from dynaconf import settings
import tempfile
from constants import ModelType

import logging
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.info("test")


def get_event_dict(connection, building_id, model_type, end_date):
    # Reading the events
    event_dictionary = dict()
    train_end_date = datetime.strptime(str(end_date), '%Y%m%d')

    events_df = pd.DataFrame()

    if model_type == ModelType.BASELINE.value:
        events_df = pd.read_sql("SELECT * FROM dbo.DimEvent WHERE IsBaseline=1 AND MeterKey = %s " % building_id,
                                connection)
    elif model_type == ModelType.M_AND_V.value:
        events_df = pd.read_sql("SELECT * FROM dbo.DimEvent WHERE Flagged=1 AND MeterKey = %s " % building_id,
                                connection)

    for index, row in events_df.iterrows():
        event_date = datetime.strptime(str(row['DateKey']), '%Y%m%d')
        if event_date <= (train_end_date - timedelta(days=90)):
            event_dictionary[row['EventName']] = (
                row['Timestamp'], event_date, row['EventId'], row['DateKey'])

    return event_dictionary


def get_train_data(connection, building_id, startDate, endDate, event_dictionary, resolution, alg):
    date = pd.read_sql("SELECT * FROM dbo.DimDate WHERE DateKey >= %s AND DateKey <= %s" % (startDate, endDate),
                       connection)
    time = pd.read_sql("SELECT * FROM dbo.DimTime", connection)

    reading = pd.read_sql(
        "SELECT * FROM dbo.MeterReadings WHERE DateKey >= %s AND DateKey <= %s AND MeterReading > 0 AND "
        "(IsSuspicious = 0 OR IsSuspicious IS NULL) and MeterKey = %s" % (
            startDate, endDate, building_id),
        connection)

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

    # preprocess the meter reading data
    reading_data = reading.drop(
        columns=['Source', 'IsSuspicious', 'AirTemperature', 'ApparentTemperature', 'DewPointTemperature',
                 'RelativeHumidity'])
    reading_data = reading_data.drop_duplicates(
        subset=['MeterKey', 'DateKey', 'TimeKey'], keep='first')

    # merge meter reading and date data
    reading_date = pd.merge(left=reading_data, right=date_data,
                            how='left', left_on='DateKey', right_on='DateKey')
    # merge meter reading, date with time data
    all_data = pd.merge(left=reading_date, right=time_data,
                        how='left', left_on='TimeKey', right_on='TimeKey')
    all_data = pd.merge(left=all_data, right=reading_temperature, how='left', left_on=['DateKey', 'TimeKey'],
                        right_on=['DateKey', 'TimeKey'])

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

    y_train = data_training["MeterReading"].values
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_train = y_train.astype(float)
    y_train = y_train.reshape(y_train.shape[0], )

    return X_train, y_train


def get_model_next_version(connection, building_id, model_type, event_dictionary):
    model_next_version = 'v1'

    model_details = pd.DataFrame()

    if model_type == ModelType.BASELINE.value:
        model_details = pd.read_sql(
            "SELECT TOP 1 * FROM dbo.Model WHERE MeterKey = %s AND Type='BASELINE' ORDER BY CreatedDate DESC" % building_id,
            connection)
    elif model_type == ModelType.M_AND_V.value:
        model_details = pd.read_sql(
            "SELECT TOP 1 * FROM dbo.Model WHERE MeterKey = %s AND Type='M_AND_V' ORDER BY CreatedDate DESC" % building_id,
            connection)

    # check whether any recent event which require retraining along with force train parameter to train model again
    outdated = True
    if not model_details.empty:
        model_timestamp = model_details['CreatedDate'].iloc[0]
        model_version = model_details['ModelVersion'].iloc[0]
        model_identifier = model_details['ModelId'].iloc[0]
        outdated = False

        if model_type == ModelType.BASELINE.value:
            for key, value in event_dictionary.items():
                (event_timestamp, event_date, event_id,
                 datekey) = event_dictionary[key]
                if model_timestamp < event_timestamp:
                    outdated = True
                    model_next_version = 'v' + str(int(model_version.split('v')[1]) + 1)
                    break

        elif model_type == ModelType.M_AND_V.value:
            model_event_details = pd.read_sql(
                "SELECT event.EventName, event.DateKey, event.EventId FROM dbo.DimEvent event "
                "INNER JOIN dbo.ModelEventMappings mapping "
                "ON event.EventId=mapping.EventId WHERE mapping.ModelId = %s" % model_identifier,
                connection)

            for key, value in event_dictionary.items():
                found = False

                for index, row in model_event_details.iterrows():
                    if key == row['EventName']:
                        found = True

                if found is False:
                    outdated = True
                    model_next_version = 'v' + str(int(model_version.split('v')[1]) + 1)
                    log.info("Model mark for the next version " + str(building_id))
                    break

    return outdated, model_next_version


def get_model_path(building_id, alg, resolution, type, model_next_version, start_date, end_date):
    model_store_name = "{}_{}_{}_{}_{}_{}_{}.joblib".format(building_id,
                                                            alg, resolution, type, model_next_version, start_date, end_date)
    # dirname = os.path.dirname(__file__)
    dirname = os.path.abspath('')

    model_store_location = tempfile.gettempdir()
    model_store_location = os.path.join(dirname, model_store_location)
    upload_file_location = os.path.join(model_store_location, model_store_name)

    return model_store_name, upload_file_location


def upload_model(building_id,
                 reg, finalscore, r2, mse, mape,
                 upload_file_location, model_store_name, start_date, end_date):
    dump((reg, finalscore, r2, mse, mape, start_date, end_date), upload_file_location)

    try:
        account_key = settings.get('AZURE_ACCOUNT_KEY', '')
        blob_connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapmodels;' \
            'AccountKey=' + account_key + 'EndpointSuffix=core.windows.net'
        blob_service_client = BlobServiceClient.from_connection_string(
            blob_connection_str)
        container_name = 'models'

        try:
            # Attempt to create container
            blob_service_client.create_container(container_name)
        # Catch exception and print error
        except ResourceExistsError as error:
            log.debug("Container exist, hence ignore")
            log.debug(error)

        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name,
                                                          blob=model_store_name)
        log.info("\nUploading to Azure Storage as blob:\n\t" + model_store_name)

        # Upload the created file
        with open(upload_file_location, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    except Exception as ex:
        log.error(
            'Exception occurred while uploading to blob storage for building ' + str(building_id))
        raise ex
