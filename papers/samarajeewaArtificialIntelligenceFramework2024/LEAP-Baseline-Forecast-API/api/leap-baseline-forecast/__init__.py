import azure.functions as func

from queue_handler import send_baseline_forecast_message

import logging
log = logging.getLogger()
log.setLevel(logging.INFO)


def main(req: func.HttpRequest) -> func.HttpResponse:

    try:
        building_ids = req.params.get('building_ids')
        start_date = req.params.get('start_date')
        end_date = req.params.get('end_date')
        baseline_type = req.params.get('baseline_type')

        if building_ids is None or start_date is None or end_date is None or baseline_type is None:
            return func.HttpResponse(f"Please provide building_ids, baseline_type, start_date and end_date!",
                                     status_code=400)

        building_ids = [building_id.strip()
                        for building_id in building_ids.split(',')]

        if len(building_ids) == 0:
            return func.HttpResponse(f"Please provide building_ids seprated by commas!", status_code=400)

        for building_id in building_ids:
            send_baseline_forecast_message(building_id, baseline_type, start_date, end_date)

        return func.HttpResponse(f'Model Forecasting Triggered Successfully for Buildings = {building_ids}!', status_code=200)

    except Exception as ex:
        log.error(ex)

        return func.HttpResponse(f"Model Forecasting failed!", status_code=500)
