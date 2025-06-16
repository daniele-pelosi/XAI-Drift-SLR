from torch import nn
from torch.nn import ModuleList
from lstm.custom_lstm import LSTM


class DALSTMModel(nn.Module):
    def __init__(self, input_size=None,
                 hidden_size=None,
                 n_layers=None,
                 max_len=None,
                 dropout=0.2):
        """
        Constructor.

        Parameters
        ----------
        input_size: int
            Number of dimensions per timestep.
        hidden_size: int
            Number of neurons in LSTM layer weights.
        n_layers: int
            Number of LSTM layers.
        max_len: int
            Number of timesteps.
        dropout: float
        """
        super(DALSTMModel, self).__init__()

        self.n_layers = n_layers
        self.lstm_layers = ModuleList()# list()
        self.batchnorm_layers = ModuleList()# list()

        for i in range(n_layers):

            if i == 0:
                inp_size = input_size
            else:
                inp_size = hidden_size

            # batch first by default
            self.lstm_layers.append(LSTM(input_size=inp_size,
                                         hidden_size=hidden_size,
                                         num_layers=1,
                                         recurrent_dropout=dropout))
            self.batchnorm_layers.append(nn.BatchNorm1d(max_len))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float()  # if tensors are saved in a different format
        for i in range(self.n_layers):
            x, (hidden_state, cell_state) = self.lstm_layers[i](x)
            x = self.batchnorm_layers[i](x)
        yhat = self.output_layer(x[:, -1, :])  # only the last one in the sequence
        return yhat.squeeze(dim=1)
