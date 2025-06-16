import logging
from model_utilities import get_event_dict, get_model_next_version, get_model_path, upload_model, get_train_data
from common_utilities import get_db_connection, mean_absolute_percentage_error
from sklearn.metrics import make_scorer
from math import sqrt
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score
import os
import os.path
from xgboost import XGBRegressor
import numpy as np
import pyodbc
import warnings
from constants import ModelType

warnings.filterwarnings("ignore")
# import logging as log


# log.basicConfig(stream=sys.stdout, level=log.DEBUG)

log = logging.getLogger()
log.setLevel(logging.INFO)
logging.info("test")


def get_model():
    return XGBRegressor(max_depth=9, learning_rate=0.1, n_estimators=120,
                        objective="reg:squarederror", booster='gbtree',
                        n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                        subsample=1, colsample_bytree=0.8, colsample_bylevel=1,
                        reg_alpha=0.0, reg_lambda=1, scale_pos_weight=1,
                        base_score=0.5, random_state=0, seed=None, missing=np.nan,
                        importance_type="weight")


def update_model_info(connection, building_id, model_type, event_dictionary, user,
                      cv_rmse, cv_r2, cv_mape,
                      model_store_name, model_next_version, start_date, end_date):

    model_description = 'Model generated for building ' + str(building_id) + '. with version ' \
        + model_next_version + \
        ' trained on date range [' + \
        str(start_date) + ', ' + str(end_date) + ']'
    
    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO [dbo].[Model]([ModelName],[ModelDescription],"
            "[ModelVersion],[MeterKey],[CreatedBy], [Type], [CV_RMSE], [CV_R2], [CV_MAPE]) values (?,?,?,?,?,?,?,?,?)",
            model_store_name,
            model_description,
            model_next_version, building_id, user, model_type, cv_rmse, cv_r2, cv_mape)
        cursor.execute("SELECT @@IDENTITY AS ID;")
        model_id = int(cursor.fetchone()[0])

        # insert model event mappings
        for key, value in event_dictionary.items():
            (event_timestamp, event_date, event_id,
             datekey) = event_dictionary[key]
            cursor.execute("INSERT INTO  [dbo].[ModelEventMappings] ([ModelId], [EventId])"
                           "values (?, ?)", model_id, event_id)
            
        connection.commit()
    except pyodbc.Error as e:
        connection.rollback()
        log.error(
            "Error while adding model info to the database for building " + str(building_id))
        raise e
    return model_id


def train(reg, X_train, y_train):
    Kfold = 10

    mape_scorer = make_scorer(
        mean_absolute_percentage_error, greater_is_better=False)

    scorings = {'cv_r2': 'r2',
                'cv_neg_mse': 'neg_mean_squared_error', 'cv_mape': mape_scorer}

    scores = cross_validate(reg, X_train, y_train,
                            cv=Kfold, scoring=scorings, n_jobs=1)

    cv_rmse = sqrt(abs(np.mean(scores['test_cv_neg_mse'])))
    cv_mape = np.mean(abs(scores['test_cv_mape']))
    cv_r2 = np.mean(scores['test_cv_r2'])

    log.info("CV_R2 %s" % cv_r2)
    log.info("CV_RMSE %s" % cv_rmse)
    log.info("CV_MAPE %s" % cv_mape)

    reg.fit(X_train, y_train)
    finalscore = reg.score(X_train, y_train)
    y_pred = reg.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    mape = mean_absolute_percentage_error(y_train, y_pred)

    log.info("R2 %s" % r2)
    log.info("MSE %s" % mse)
    log.info("MAPE %s" % mape)

    return y_pred, finalscore, r2, mse, mape, cv_rmse, cv_r2, cv_mape


def train_model(building_id, model_type, start_date, end_date):
    connection = get_db_connection()

    user = 'leap_admin'
    resolution = '15m'
    alg = 'XGBRegressor'

    event_dictionary = get_event_dict(connection, building_id, model_type, end_date)

    if model_type == ModelType.M_AND_V.value and len(event_dictionary) == 0:
        raise Exception("Skipping M & V training. No events flagged to perform M & V.")

    latest_event_date_key = max(event_dictionary.values(), key=lambda x: x[3])[3]

    outdated, model_next_version = get_model_next_version(connection,
                                                          building_id,
                                                          model_type,
                                                          event_dictionary)

    if model_type == ModelType.M_AND_V.value and not outdated:
        raise Exception(f'M & V Model is upto date for the building {building_id}')

    X_train, y_train = get_train_data(
        connection, building_id, start_date, end_date, event_dictionary, resolution, alg)

    reg = get_model()

    model_store_name, upload_file_location = get_model_path(building_id, alg,
                                                            resolution, model_type,
                                                            model_next_version, start_date, end_date)

    if not os.path.exists(upload_file_location):

        y_pred, finalscore, r2, mse, mape, cv_rmse, cv_r2, cv_mape = train(
            reg, X_train, y_train)

        try:
            upload_model(building_id, reg, finalscore, r2, mse, mape,
                         upload_file_location, model_store_name, start_date, end_date)

            model_id = update_model_info(connection, building_id, model_type, event_dictionary, user,
                                         cv_rmse, cv_r2, cv_mape,
                                         model_store_name, model_next_version, start_date, end_date)
            
            os.remove(upload_file_location)

        except Exception as ex:
            raise(ex)
    else:
        raise Exception("Model already exists in the path!")

    return model_id, latest_event_date_key, model_store_name, model_next_version, X_train, y_train, y_pred, reg
