import pandas as pd
from lstm.dataloader import H5Dataset
from lstm.model import DALSTMModel
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_dataloader(input_data_location, batch_size=1):
    """
    Get dataloader for the test set.

    Parameters
    ----------
    input_data_location: str
    batch_size: int

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = H5Dataset(input_data_location + "/test.hdf5",
                              x_name="test_X",
                              y_name="test_y")

    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)


def predict(pretrained_model_filepath,
            input_dataset_location,
            n_neurons,
            n_layers,
            normalization_factor,
            dropout=0.2):
    """

    Parameters
    ----------
    pretrained_model_filepath: str
    input_dataset_location: str
    n_neurons: int
    n_layers: int
    normalization_factor: float
    dropout: float
        Not utilized for predicting the test set.

    Returns
    -------
    pd.DataFrame containing predictions and corresponding ground truth labels.
    """

    device_name = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    dataloader = get_dataloader(input_data_location=input_dataset_location)

    model = DALSTMModel(input_size=dataloader.dataset.get_feature_dim(),
                        hidden_size=n_neurons,
                        n_layers=n_layers,
                        max_len=dataloader.dataset.get_num_timesteps(),
                        dropout=dropout).to(device)
    checkpoint = torch.load(pretrained_model_filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            pred = model(inputs)
            y_true.extend(targets.tolist())
            y_pred.extend(pred.tolist())


    data = {'preds': y_pred, 'labels': y_true}
    res_df = pd.DataFrame(data=data)
    res_df['preds'] = res_df['preds'] * normalization_factor
    res_df['labels'] = res_df['labels'] * normalization_factor
    return res_df
