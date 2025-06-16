import logging as log
import json

import azure.functions as func
from drift_detection import initiate_drift_detection


def main(msg: func.QueueMessage) -> None:
    try:
        log.info('Processing a queue item: %s', msg.get_body().decode('utf-8'))

        message = json.loads(msg.get_body().decode('utf-8'))

        building_id = message['building_id']
        end_date_key = message['end_date']

        if building_id is None or end_date_key is None:
            log.error(f"[building_id/end_date] not found!")
        else:
            initiate_drift_detection(int(building_id), end_date_key)
            log.info(f'Drift Detection was Successful for Building = {building_id}! for date: {end_date_key}')
    except Exception as ex:
        log.error(ex)