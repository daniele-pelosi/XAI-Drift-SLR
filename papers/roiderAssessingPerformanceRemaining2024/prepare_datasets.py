import sys
from graphgps import GTConverter
from lstm import prepare_dataset as lstm_prep
import data_preprocessing

if __name__ == '__main__':

    dataset_name = sys.argv[1]
    input_dataset_location = sys.argv[2]
    output_dataset_location = sys.argv[3]

    # Perform data split and prepare features
    data_preprocessing.prepare_data(dataset_name=dataset_name,
                                    dataset_location=input_dataset_location,
                                    output_location=output_dataset_location)

    # Prepare dataest for GraphGPS
    GTConverter.create_graph_dataset(input_dataset_location=output_dataset_location + '/' + dataset_name + '/',
                                     graph_dataset_path_raw=output_dataset_location + '/' + dataset_name + '/graph_dataset/raw/')

    # Prepare dataset for LSTM
    lstm_prep.prepare_dataset(input_data_filepath=output_dataset_location + '/' + dataset_name + '/',
                              output_data_filepath=output_dataset_location + '/' + dataset_name + '/dalstm/data/')

    print('Completed Preprocessing.')
