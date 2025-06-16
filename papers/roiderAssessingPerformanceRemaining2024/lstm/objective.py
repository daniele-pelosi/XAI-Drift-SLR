import optuna
from lstm import train
import yaml


class Objective:
    """
    Objective that defines the procedure for one optuna trial.
    """
    def __init__(self,
                 cfg_file,
                 input_data_location,
                 output_location,
                 device,
                 seed,
                 search_space,
                 max_training_epochs
                 ):
        """
        Constructor for the Objective class.

        Parameters
        ----------
        cfg_file: str
        input_data_location: str
        output_location: str
        device: str
        seed: int
        search_space: dict
        max_training_epochs: int
        """
        self.input_data_location = input_data_location
        self.output_location = output_location
        self.device = device
        self.seed = seed
        self.search_space = search_space
        self.max_training_epochs = max_training_epochs
        self.cfg_file = cfg_file

        with open(cfg_file, 'r') as file:
            cfg = yaml.safe_load(file)

        self.optimizer_type = str(cfg['optim']['optimizer'])
        self.weight_decay = float(cfg['optim']['weight_decay'])
        self.base_lr = float(cfg['optim']['lr'])
        self.eps = float(cfg['optim']['eps'])
        self.early_stop_patience = int(cfg['train']['early_stop_patience'])
        self.early_stop_min_delta = float(cfg['train']['early_stop_min_delta'])
        self.batch_size = int(cfg['train']['batch_size'])
        self.dropout = float(cfg['model']['dropout'])

    def __call__(self,
                 trial: optuna.trial.Trial):
        n_neurons = trial.suggest_categorical('n_neurons', self.search_space['n_neurons'])
        n_layers = trial.suggest_categorical('n_layers', self.search_space['n_layers'])

        val_loss, val_nmae = train.train(batch_size=self.batch_size,
                                         input_data_location=self.input_data_location + '/dalstm/data/',
                                         output_location=self.output_location + '/trial_{}'.format(trial.number) + '/',
                                         device=self.device,
                                         seed=self.seed,
                                         max_training_epochs=self.max_training_epochs,
                                         n_neurons=n_neurons,
                                         n_layers=n_layers,
                                         dropout=self.dropout,
                                         optimizer_type=self.optimizer_type,
                                         base_lr=self.base_lr,
                                         eps=self.eps,
                                         weight_decay=self.weight_decay,
                                         early_stop_patience=self.early_stop_patience,
                                         early_stop_min_delta=self.early_stop_min_delta)

        trial.set_user_attr("best_val_nmae", val_nmae)
        return val_loss
