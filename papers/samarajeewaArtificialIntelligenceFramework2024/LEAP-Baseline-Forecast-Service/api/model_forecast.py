from model_utilities import get_event_dict, get_test_data, download_model, upload_forecast_data, get_model_ids
from common_utilities import get_db_connection, results_to_data_frame, get_file_name
import pandas as pd
import logging as log
import tempfile
import numpy as np
from model_utilities import download_model
from datetime import datetime
import pyodbc
from pandas.tseries.offsets import MonthEnd, MonthBegin

def model_predict(building_id=None, baseline_type=None, start_date=None, end_date=None):
    connection = get_db_connection()
    results_df = pd.DataFrame()

    is_empty = True

    if building_id is None:
        # Get meter keys
        st = "SELECT DISTINCT MeterKey, Meter FROM [Leap].[dbo].[DimMeter]  where IsActive=1"
        meters = pd.read_sql(st, connection)

        building_ids = meters['MeterKey'].values
    else:
        building_ids = [building_id]

    log.info("Building IDs: %s, Forecast start date: %s, Forecast end date: %s", str(
        building_ids), start_date, end_date)

    for building_id in building_ids:

        try:
            model_ids = get_model_ids(connection, building_id, baseline_type)

            if len(model_ids) == 0:
                log.info(f'Skipping for building: {building_id}: No models found!')
                break

            resolution = '15m'
            alg = 'XGBRegressor'

            for model_id in model_ids:
                reg, model_name, _, _ = download_model(connection, model_id)

                event_dictionary = get_event_dict(connection, model_id)

                X_train, date_key, time_key = get_test_data(
                    connection, building_id, start_date, end_date, event_dictionary, resolution, alg)

                Y_pred = reg.predict(X_train)

                results_df = pd.concat([results_df,
                                        results_to_data_frame(building_id, model_id, date_key, time_key, Y_pred)])

                is_empty = False
        except Exception as ex:
            log.error(f'Skipping for building: {building_id} with error:')
            log.error(ex)

    file_name = get_file_name(building_ids, baseline_type, start_date, end_date)

    if not is_empty:
        file_path = f'{tempfile.gettempdir()}/{file_name}'
        results_df.to_csv(file_path, index=False)

        upload_forecast_data(file_name, file_path)
    else:
        log.info("Empty file created!")


def get_events(connection, model_id, building_id):
    events_require_calculation = []
    model_events = []

    event_details = pd.read_sql(
        "SELECT event.EventName, event.DateKey, event.EventId FROM dbo.DimEvent event INNER JOIN dbo.ModelEventMappings mapping "
        "ON event.EventId=mapping.EventId WHERE mapping.ModelId = %s" % model_id,
        connection)

    for index, row in event_details.iterrows():
        event_name = row['EventName']
        event_id = row['EventId']
        measures_df = pd.read_sql("SELECT * FROM dbo.MeterEventSummary WHERE MeterKey = %s AND EventId = %s "
                                  % (building_id, event_id),
                                  connection)

        model_events.append(event_name)
        if measures_df.empty:
            events_require_calculation.append((event_id, event_name))

    return model_events, events_require_calculation

def applyRate(x):
    PEAK_RATE = 0.1166
    OFF_PEAK_RATE = 0.076284
    if (x >= 0 and x < 70000) or (x > 230000):
        return OFF_PEAK_RATE
    else:
        return PEAK_RATE

def month_to_integer(x):
    datetime_object = datetime.strptime(x, "%b")
    return datetime_object.month


def add_month_start(dt_series):
    year = dt_series.dt.year.iloc[0]
    month = dt_series.dt.month.iloc[0]
    return pd.to_datetime(f'{year}-{month:02d}-01')


def add_month_end(dt_series):
    year = dt_series.dt.year.iloc[0]
    month = dt_series.dt.month.iloc[0]
    last_day_of_month = pd.to_datetime(f'{year}-{month:02d}-{dt_series.dt.days_in_month.iloc[0]}')
    return last_day_of_month

def get_data(connection, building_id, start_date, end_date):

    CDDBASE = 20
    HDDBASE = 18

    st = "SELECT MeterKey,CampusKey FROM [Leap].[dbo].[DimMeter] WHERE MeterKey=%s" % building_id
    building_details = pd.read_sql(st, connection)

    if building_details.empty != True:
        campus_key = int(building_details['CampusKey'].iloc[0])
    else:
        campus_key = 1


    date = pd.read_sql("SELECT * FROM dbo.DimDate WHERE DateKey >= %d and DateKey <= %d" % (start_date, end_date),
                       connection)
    time = pd.read_sql("SELECT * FROM dbo.DimTime", connection)

    start_date_key = int(str(start_date)[4:])
    end_date_key = int(str(end_date)[4:])

    reading = pd.read_sql("""SELECT[DateKey]
                      ,[TimeKey]
                      ,[AirTemp] AS [ApparentTemperature]
                      ,[AirTemp] AS [AirTemperature]
                      ,[DewpointTemp] AS [DewPointTemperature]
                      ,[RelativeHumidity]
                  	FROM [Leap].[dbo].[TMYWeatherData] where DateKey >= 101 and DateKey <= 1231 and CampusKey=%s""" % campus_key,
                          connection)

    reading.loc[(reading['DateKey'] >= start_date_key) & (reading['DateKey'] <= 1231), 'TempDateKey'] = \
        str(start_date)[:4] + reading['DateKey'].astype(str).str.zfill(4)

    reading.loc[(reading['DateKey'] >= 101) & (reading['DateKey'] <= end_date_key), 'TempDateKey'] = \
        str(end_date)[:4] + reading['DateKey'].astype(str).str.zfill(4)

    reading['DateKey'] = reading['TempDateKey'].astype(int)

    reading.drop(columns=['TempDateKey'], inplace=True)

    reading['Timestamp'] = pd.to_datetime(
        reading.DateKey.astype(str) + " " + reading['TimeKey'].astype(str).str.zfill(6), format="%Y%m%d %H%M%S")
    temp = pd.date_range(reading['Timestamp'].min(axis=0), reading['Timestamp'].max(axis=0), freq='15min',
                         name="Timestamp").to_frame().reset_index(drop=True)
    temp2 = pd.merge(temp, reading, how="left", left_on="Timestamp", right_on="Timestamp").fillna(method='ffill')
    temp2.TimeKey = temp2.Timestamp.dt.strftime("%H%M%S").astype(int)
    temp2.DateKey = temp2.Timestamp.dt.strftime("%Y%m%d").astype(int)
    reading = temp2.drop(columns=["Timestamp"])

    date_data = date.drop_duplicates(subset='DateKey', keep='first')
    date_data = date_data.drop(columns=['Date', 'DaySuffix', 'WeekDayName', 'HolidayText', 'DayOfYear',
                                        'ISOWeekOfYear', 'MonthName', 'QuarterName', 'MMYYYY', 'MonthYear',
                                        'FirstDayOfMonth', 'LastDayOfMonth', 'FirstDayOfQuarter',
                                        'LastDayOfQuarter',
                                        'FirstDayOfYear', 'LastDayOfYear', 'FirstDayOfNextMonth',
                                        'FirstDayOfNextYear',
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
    reading_data = reading.drop(columns=['AirTemperature'])
    reading_data = reading_data.drop_duplicates(subset=['DateKey', 'TimeKey'], keep='first')

    # merge meter reading and date data
    reading_date = pd.merge(left=reading_data, right=date_data, how='left', left_on='DateKey', right_on='DateKey')
    # merge meter reading, date with time data
    all_data = pd.merge(left=reading_date, right=time_data, how='left', left_on='TimeKey', right_on='TimeKey')

    all_data.sort_values(['DateKey', 'TimeKey'], ascending=[True, True], inplace=True)

    # fill na in weather with backfill
    all_data.ApparentTemperature.fillna(method='ffill', inplace=True)
    all_data.DewPointTemperature.fillna(method='ffill', inplace=True)
    all_data.RelativeHumidity.fillna(method='ffill', inplace=True)

    data_training = all_data

    data_training = data_training.replace(r'^\s*$', np.nan, regex=True)
    data_training = data_training.dropna()
    print("reading " + str(len(data_training)) + " rows")
    if len(data_training) == 0:
        return

    data_training["Timestamp"] = pd.to_datetime(
        (data_training["DateKey"].astype(str) + data_training["FullTimeString24"].astype(str)),
        format='%Y%m%d%H:%M:%S')
    data_training["Time"] = data_training["Timestamp"]
    data_training = data_training.set_index('Time')

    # add cdd and hdd to dataframe
    data_training["ApparentTemperature"] = pd.to_numeric(data_training["ApparentTemperature"])

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
                                        data_training["CDD"])  # dd[d]["sum"]

    dd = data_training["hdd"].resample("D").mean()
    data_training["HDD"] = -1
    for d in dd.index:
        data_training["HDD"] = np.where((data_training.index.date == d.date()) | (data_training["HDD"] < 0),
                                        dd[dd.index == d].values[0],
                                        data_training["HDD"])  # dd[d]["sum"]

    data_training = data_training.set_index('Timestamp')

    data_training["DewPointTemperature"] = pd.to_numeric(data_training["DewPointTemperature"])
    data_training["RelativeHumidity"] = pd.to_numeric(data_training["RelativeHumidity"])
    data_training["HDD"] = pd.to_numeric(data_training["HDD"])
    data_training["CDD"] = pd.to_numeric(data_training["CDD"])

    return data_training

def populate_savings(building_id, model_id, start_date, end_date):
    connection = get_db_connection()
    model_events, events_require_calculation = get_events(connection, model_id, building_id)

    if len(events_require_calculation) == 0:
        log.info("No events require calculation!")
        return

    data_training = get_data(connection, building_id, start_date, end_date)

    for (event_id, event_name) in events_require_calculation:
        events = []

        for model_event in model_events:
            log.info("Adding events " + model_event)
            data_training[model_event] = 1
            events.append(model_event)

        if len(events) > 1:
            events = sorted(events)

        reg, _, _, _ = download_model(connection, model_id)

        categorial = ['IsWeekend', 'IsHoliday', 'HalfHour'] + events
        nvariables = ["ApparentTemperature", "RelativeHumidity", "HDD", "CDD",
                      'Weekday', 'Hour24', 'Minute']

        log.info("Categorical variables: %s", categorial)
        log.info("Continous variables: %s", nvariables)

        X_train_ndata = data_training[nvariables].values
        X_train_cdata = data_training[categorial].values
        X_train = np.hstack((X_train_cdata, X_train_ndata))
        X_train = X_train.astype(float)
        forecast = reg.predict(X_train)
        data_training["Forecast_WithEvent"] = forecast

        data_training[event_name] = 0
        X_train_ndata = data_training[nvariables].values
        X_train_cdata = data_training[categorial].values
        X_train = np.hstack((X_train_cdata, X_train_ndata))
        X_train = X_train.astype(float)
        forecast = reg.predict(X_train)

        data_training["Forecast_WithoutEvent"] = forecast
        building_prediction = pd.DataFrame(
            columns=['BuildingNumber', 'Timestamp', 'Forecast_WithEvent', "Forecast_WithoutEvent"])

        building_prediction["Timestamp"] = data_training.index
        building_prediction["BuildingNumber"] = model_id
        building_prediction["Forecast_WithEvent"] = data_training.Forecast_WithEvent.values
        building_prediction["Forecast_WithoutEvent"] = data_training.Forecast_WithoutEvent.values
        building_prediction['TimeKey'] = building_prediction.Timestamp.dt.strftime("%H%M%S").astype(int)
        building_prediction['DateKey'] = building_prediction.Timestamp.dt.strftime("%Y%m%d").astype(int)

        building_prediction["rate"] = building_prediction.TimeKey.apply(applyRate)
        building_prediction[
            "Forecast_WithEventCost"] = building_prediction.Forecast_WithEvent * building_prediction.rate
        building_prediction[
            "Forecast_WithoutEventCost"] = building_prediction.Forecast_WithoutEvent * building_prediction.rate

        aggregated_by_date = building_prediction.groupby("DateKey", as_index=False).agg(
            {
                'Forecast_WithEvent': sum,  # get the count of networks
                'Forecast_WithoutEvent': sum,  # get the count of networks
                'Forecast_WithEventCost': sum,  # get the count of networks
                'Forecast_WithoutEventCost': sum  # get the count of networks
            })
        aggregated_by_date['DateKey'] = pd.to_datetime(aggregated_by_date['DateKey'].astype(str), format='%Y%m%d')
        aggregated_by_date['DateTime'] = pd.to_datetime(aggregated_by_date['DateKey'], format='%Y-%m-%d')
        aggregated_by_date.set_index('DateKey', inplace=True)

        df_month = aggregated_by_date.groupby(aggregated_by_date['DateTime'].dt.strftime('%b'))[[
            'Forecast_WithEvent', 'Forecast_WithoutEvent', 'Forecast_WithEventCost', 'Forecast_WithoutEventCost']] \
            .sum()


        df_month['month_number'] = df_month.index.to_series().apply(month_to_integer)
        df_month['month'] = df_month.index
        df_month['month_start'] = aggregated_by_date.groupby(aggregated_by_date['DateTime'].dt.strftime('%b'))[
            'DateTime'].agg(add_month_start)
        df_month['month_end'] = aggregated_by_date.groupby(aggregated_by_date['DateTime'].dt.strftime('%b'))[
            'DateTime'].agg(add_month_end)
        df_month['energy_saved'] = df_month['Forecast_WithoutEvent'] - df_month['Forecast_WithEvent']
        df_month['cost_saved'] = df_month['Forecast_WithoutEventCost'] - df_month['Forecast_WithEventCost']

        df_month.set_index("month_number", inplace=True)
        df_month.sort_index()

        user = 'leap_admin'
        for index, row in df_month.iterrows():
            cursor = connection.cursor()
            try:
                cursor.execute(
                    "INSERT INTO [dbo].[MeterEventSummary]([EventId],[MeterKey],[StartDate],[EndDate]"
                    ",[ProjectedEnergyConsumption], [ActualEnergyConsumption], [AvoidedConsumption], "
                    "[AvoidedMonetaryValue], [CreatedBy], [CreatedDate], [ModelID]) values (?,?,?,?,?,?,?,?,?,?,?)",
                    event_id, building_id,
                    row['month_start'], row['month_end'], row['Forecast_WithoutEvent'], row['Forecast_WithEvent'],
                    row['energy_saved'], row['cost_saved'], user, datetime.now(), model_id)

                cursor.commit()
                cursor.close()
            except pyodbc.Error as e:
                log.error("Error while adding model info to the database for building " + str(building_id))
                log.error(e)
