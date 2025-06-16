from azure.storage.queue import QueueClient, BinaryBase64EncodePolicy
from dynaconf import settings
import logging as log
import json


def send_m_and_v_baseline_trigger_message(building_id, model_id, start_date, end_date):
    account_key = settings.get('AZURE_ACCOUNT_KEY_FORECAST', '')
    connection_str = 'DefaultEndpointsProtocol=https;AccountName=leapbaselineforecast;' \
                     'AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'

    queue_name = settings.get('BASELINE_FORECAST_QUEUE_NAME', '')

    queue = QueueClient.from_connection_string(
        conn_str=connection_str, queue_name=queue_name, message_encode_policy=BinaryBase64EncodePolicy()
    )

    msg = {"building_id": building_id, "baseline_type": "M_AND_V", "model_id": model_id, "savings_forecast": True,
           "start_date": start_date, "end_date": end_date}

    log.info(msg)

    queue.send_message(json.dumps(msg).encode('utf-8'))

    log.info(f'Message [{msg}] sent to queue: {queue_name}')
