from model_trainer import train_model
from common_utilities import get_db_connection
import pandas as pd

import logging
log = logging.getLogger()
log.setLevel(logging.INFO)
logging.info("test")


def model_train(building_id=None, model_type=None, train_start_date=None, train_end_date=None):
    errors = []

    if building_id is None:
        # Get meter keys
        connection = get_db_connection()
        st = "SELECT DISTINCT MeterKey, Meter FROM [Leap].[dbo].[DimMeter]  where IsActive=1"
        meters = pd.read_sql(st, connection)

        building_ids = meters['MeterKey'].values  # [685, 700, 715]
    else:
        building_ids = [int(building_id)]

    if train_start_date is None:
        train_start_date = 20220101

    if train_end_date is None:
        train_end_date = 20221031

    model_ids = []

    log.info("Building IDs: %s, model_type: %s, Train start date: %s, Train end date: %s", str(
        building_ids), model_type, train_start_date, train_end_date)

    for building_id in building_ids:

        try:
            model_id, latest_event_date_key, _, _, _, _, _, _ = train_model(building_id, model_type, train_start_date, train_end_date)
            model_ids.append((building_id, model_id, latest_event_date_key))

        except Exception as e:
            log.error(e)
            err = "Skipping Building " + str(building_id)
            log.error(err)
            errors.append(err)

    return errors, model_ids
