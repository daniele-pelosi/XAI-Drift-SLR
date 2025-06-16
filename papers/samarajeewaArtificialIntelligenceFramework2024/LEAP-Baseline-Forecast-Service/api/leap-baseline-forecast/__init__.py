import logging as log
import json

import azure.functions as func
from model_forecast import model_predict, populate_savings
from constants import ModelType
from common_utilities import get_attribute


def main(msg: func.QueueMessage) -> None:
    try:
        log.info('Processing a queue item: %s', msg.get_body().decode('utf-8'))

        message = json.loads(msg.get_body().decode('utf-8'))

        building_id = get_attribute(message, 'building_id')
        start_date = get_attribute(message, 'start_date')
        end_date = get_attribute(message, 'end_date')
        baseline_type = get_attribute(message, 'baseline_type')
        savings_forecast = get_attribute(message, 'savings_forecast')
        model_id = get_attribute(message, 'model_id')

        if building_id is None or start_date is None or end_date is None or baseline_type is None:
            log.error(f"[building_id/start_date/end_date/baseline_type] not found!")
        else:
            if savings_forecast is not None and baseline_type == ModelType.M_AND_V.value and model_id is not None:
                log.info("Starting savings forecast")
                populate_savings(building_id, model_id, start_date, end_date)
                log.info(
                    f'Savings Forecast Successful for building: {building_id}, Model ID: {model_id}, Type: {baseline_type}!')
            else:
                log.info("Starting energy forecast")
                model_predict(building_id, baseline_type, start_date, end_date)
                log.info(f'Model Forecast Successful for building: {building_id}, Type: {baseline_type}!')
    except Exception as ex:
        log.error(ex)
