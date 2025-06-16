import logging as log
import json

import azure.functions as func
from baseline_train import model_train
from queue_handler import send_m_and_v_baseline_trigger_message
from datetime import datetime, timedelta


def main(msg: func.QueueMessage) -> None:
    try:
        log.info('Processing a queue item: %s', msg.get_body().decode('utf-8'))

        message = json.loads(msg.get_body().decode('utf-8'))

        building_id = message['building_id']
        model_type = message['model_type']
        start_date = message['start_date']
        end_date = message['end_date']

        if start_date is None or end_date is None or model_type is None:
            log.error(f"[building_id/start_date/end_date/model_type] not found!")
        else:
            errors, model_ids = model_train(building_id, model_type, start_date, end_date)

            if len(errors) == 0:
                for (building_id, model_id, latest_event_date_key) in model_ids:
                    log.info(f'Model for MeterKey:{building_id} Trained Successfully!')

                    forecast_end_date = int(
                        (datetime.strptime(str(latest_event_date_key), "%Y%m%d") + timedelta(days=365)).strftime(
                            "%Y%m%d"))
                    send_m_and_v_baseline_trigger_message(building_id, model_id, latest_event_date_key,
                                                          forecast_end_date)
            else:
                log.error(f'Model Training for MeterKey:{building_id}  failed with {str(errors)}')

    except Exception as ex:
        log.error(ex)
