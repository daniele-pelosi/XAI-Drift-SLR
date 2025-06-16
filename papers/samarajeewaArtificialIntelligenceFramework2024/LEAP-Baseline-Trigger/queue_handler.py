from azure.storage.queue import QueueClient, BinaryBase64EncodePolicy
from dynaconf import settings
import logging as log
import json


def send_baseline_trigger_message(building_id, baseline_type, start_date, end_date):
    account_key = settings.get('AZURE_ACCOUNT_KEY', '')
    connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapbaselineforecast;' \
                     'AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'

    queue_name = settings.get('BASELINE_FORECAST_QUEUE_NAME', '')

    queue = QueueClient.from_connection_string(
        conn_str=connection_str, queue_name=queue_name, message_encode_policy=BinaryBase64EncodePolicy()
    )

    msg = {"building_id": building_id, "baseline_type": baseline_type, "start_date": start_date, "end_date": end_date}

    queue.send_message(json.dumps(msg).encode('utf-8'))

    log.info(
        f'Message ["building_id": {building_id}, "baseline_type": {baseline_type}, "start_date": {start_date}, "end_date": {end_date}] sent to queue: {queue_name}')


def send_drift_detection_trigger_message(building_id, end_date):
    account_key = settings.get('AZURE_ACCOUNT_KEY', '')
    connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapbaselineforecast;' \
                     'AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'

    queue_name = settings.get('DRIFT_DETECTION_QUEUE_NAME', '')

    queue = QueueClient.from_connection_string(
        conn_str=connection_str, queue_name=queue_name, message_encode_policy=BinaryBase64EncodePolicy()
    )

    msg = {"building_id": building_id, "end_date": end_date}

    queue.send_message(json.dumps(msg).encode('utf-8'))

    log.info(f'Message ["building_id": {building_id}, "end_date": {end_date}] sent to queue: {queue_name}')

def send_baseline_train_message(building_id, model_type, start_date, end_date):
    account_key = settings.get('AZURE_ACCOUNT_KEY_BASELINE_TRAIN', '')
    connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapbaselinetrain;' \
                     'AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'

    queue_name = settings.get('BASELINE_TRAIN_QUEUE_NAME', '')

    queue = QueueClient.from_connection_string(
        conn_str=connection_str, queue_name=queue_name, message_encode_policy=BinaryBase64EncodePolicy()
    )

    msg = {"building_id": building_id, "model_type": model_type, "start_date": start_date, "end_date": end_date}

    queue.send_message(json.dumps(msg).encode('utf-8'))

    log.info(f'Message ["building_id": {building_id}, "model_type": {model_type}, "start_date": {start_date},'
             f' "end_date": {end_date}] sent to queue: {queue_name}')
