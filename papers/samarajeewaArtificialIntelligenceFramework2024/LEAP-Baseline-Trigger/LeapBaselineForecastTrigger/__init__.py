from datetime import datetime, timezone, timedelta
import pytz
import logging

import azure.functions as func

from common_utilities import get_db_connection, retrieve_pending_model_train_processes, mark_train_process_as_finished
import pandas as pd
from queue_handler import send_baseline_trigger_message, send_drift_detection_trigger_message, \
    send_baseline_train_message


def main(leapbaselineforecasttrigger: func.TimerRequest) -> None:
    utc_timestamp = datetime.utcnow().replace(
        tzinfo=timezone.utc).isoformat()

    if leapbaselineforecasttrigger.past_due:
        logging.info('The timer is past due!')

    logging.info(
        '...LEAP Baseline timer trigger function ran at %s', utc_timestamp)

    try:
        connection = get_db_connection()

        # Get meter keys
        st = "SELECT DISTINCT MeterKey, Meter FROM [Leap].[dbo].[DimMeter]  where IsActive=1 "
        meters = pd.read_sql(st, connection)

        building_ids = meters['MeterKey'].values

        # Generate baslines for yesterday data (at 1 am today)
        yesterday = (datetime.now(pytz.timezone(
            'Australia/Melbourne')) - timedelta(1)).strftime("%Y%m%d")
        start_date = yesterday
        end_date = yesterday

        logging.info(f'Triggering baselines for {yesterday}')

        for building_id in building_ids:
            send_baseline_trigger_message(
                int(building_id), "BASELINE", start_date, end_date)

            send_baseline_trigger_message(
                int(building_id), "M_AND_V", start_date, end_date)

            send_drift_detection_trigger_message(
                int(building_id), end_date)

        logging.info("Retrieve pending model train processes")
        pending_processes = retrieve_pending_model_train_processes(building_ids)

        if (len(pending_processes) == 0):
            logging.info("No pending processes!")

        for process in pending_processes:
            send_baseline_train_message(process['MeterKey'], process['Type'], process['TrainStartDate'],
                                        process['TrainEndDate'])
            mark_train_process_as_finished(process['ProcessId'])

    except Exception as ex:
        logging.error(ex)
