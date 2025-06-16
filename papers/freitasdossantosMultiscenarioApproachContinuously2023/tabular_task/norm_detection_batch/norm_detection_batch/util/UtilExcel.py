import pandas as pd
import pathlib

'''
GENERAL COMMENTS ABOUT CONFIGURATION
    It's necessary to install xlrd for the Excel to work. Pip install
'''


def get_feature_description(feature_name):
    excel_file_path = str(pathlib.Path(__file__).parent.parent) + '/features_information/FeaturesDescriptions.xls'
    excel_file = pd.read_excel(excel_file_path, sheet_name='Features and descriptions')
    excel_column = pd.DataFrame(excel_file, columns=['FEATURE_NAME', 'FEATURE_DESCRIPTION'])

    # I had to put this here, because sometimes the name comes with the day of the week in the end. Ex: TIME_DOW_FRI.
    if "TIME_DOW" in feature_name:
        feature_name = "TIME_DOW"

    mask = excel_column['FEATURE_NAME'] == feature_name
    feature_description = excel_column[mask].values.tolist()[0][1]

    return feature_name + " - " + feature_description

