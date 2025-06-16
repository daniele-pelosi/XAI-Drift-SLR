from torch import nn
from lstm.weight_drop import WeightDrop
"""
Adapted from https://github.com/mourga/variational-lstm/blob/master/rnn_module.py
and https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
"""


class LSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 recurrent_dropout=0.
                 ):
        """

        Parameters
        ----------
        input_size: int
            Number of dimensions per timestep.
        hidden_size: int
            Number of neurons in LSTM layer weights.
        num_layers: int
            Number of LSTM layers.
        recurrent_dropout: float
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_dropout = recurrent_dropout

        self.rnns = list()
        for l in range(num_layers):
            if l == 0:
                inp_size = self.input_size
            else:
                inp_size = self.hidden_size
            self.rnns.append(nn.LSTM(input_size=inp_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=1,
                                     batch_first=True))

        # Dropout to recurrent layers (matrices weight_hh AND weight_ih of each layer of the RNN)
        if self.recurrent_dropout > 0.:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0', 'weight_ih_l0'],
                                    dropout=self.recurrent_dropout) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p.data)
            elif 'bias_ih' in name:
                p.data.fill_(0)
                # Set forget-gate bias to 1
                n = p.size(0)
                p.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                p.data.fill_(0)

    def forward(self, x, hidden=None, lengths=None, return_h=False):
        for l, rnn in enumerate(self.rnns):
            x, _ = rnn(x)
        return x, _
