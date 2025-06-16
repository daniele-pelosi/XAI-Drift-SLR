from azure.storage.queue import QueueClient, BinaryBase64EncodePolicy
from dynaconf import settings
import logging as log
import json


def send_baseline_train_message(building_id, model_type, start_date, end_date):
    account_key = settings.get('AZURE_ACCOUNT_KEY', '')
    connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapbaselinetrain;' \
                     'AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'

    queue_name = settings.get('QUEUE_NAME', '')

    queue = QueueClient.from_connection_string(
        conn_str=connection_str, queue_name=queue_name, message_encode_policy=BinaryBase64EncodePolicy()
    )

    msg = {"building_id": building_id, "model_type": model_type, "start_date": start_date, "end_date": end_date}

    queue.send_message(json.dumps(msg).encode('utf-8'))

    log.info(f'Message ["building_id": {building_id}, "model_type": {model_type}, "start_date": {start_date},'
             f' "end_date": {end_date}] sent to queue: {queue_name}')
