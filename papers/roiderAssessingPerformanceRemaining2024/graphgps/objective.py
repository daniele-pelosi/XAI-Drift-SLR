from graphgps import train_graphgps
import optuna
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
import os
import joblib
import torch
from types import SimpleNamespace
from pathlib import Path


class Objective:
    """
    Objective that defines the procedure for one optuna trial.
    """
    def __init__(self,
                 cfg_file,
                 output_location,
                 input_location,
                 search_space,
                 seed,
                 num_training_epochs):
        self.cfg_file = cfg_file
        self.output_location = output_location
        self.input_location = input_location
        self.search_space = search_space
        self.seed = seed
        self.num_training_epochs = num_training_epochs


    def __call__(self,
                 trial: optuna.trial.Trial):

        args = SimpleNamespace(cfg_file=self.cfg_file,
                               repeat=1,
                               mark_done=False,
                               opts=['seed', self.seed]
                               )

        set_cfg(cfg)
        cfg.set_new_allowed(True)
        load_cfg(cfg, args)
        cfg.out_dir = os.path.join(self.output_location)
        config = joblib.load(self.input_location + 'config.pkl')
        oh_encoder = joblib.load(self.input_location + 'oh_encoders.pkl')
        unique_activities_train = len(oh_encoder[config['activity_column']].categories_[0])
        cfg.dataset.node_encoder_num_types = unique_activities_train + 1
        tr = joblib.load(self.input_location + '/graph_dataset/raw/train.pickle')
        cfg.two_layer_linear_edge_encoder.in_dim = int(tr[0]['edge_attr'].shape[1])
        cfg.optim.max_epoch = self.num_training_epochs
        cfg.dataset.dir = self.input_location + '/graph_dataset/'

        # Hyperparameters
        cfg.posenc_LapPE.dim_pe = trial.suggest_categorical('posenc_LapPE.dim_pe', self.search_space['posenc_LapPE.dim_pe'])
        times_func_end = trial.suggest_categorical('posenc_RWSE.kernel.times_func_end', self.search_space['posenc_RWSE.kernel.times_func_end'])
        cfg.posenc_RWSE.kernel.times_func = 'range(1, ' + str(times_func_end) + ')'
        dump_cfg(cfg)

        # Set Pytorch environment
        torch.set_num_threads(cfg.num_threads)
        run_id = 'trial_' + str(trial.number)

        if not Path(os.path.join(cfg.out_dir, str(run_id))).exists():
            Path.mkdir(Path(os.path.join(cfg.out_dir, str(run_id))))

        cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
        cfg.seed = self.seed
        cfg.run_id = run_id

        val_loss = train_graphgps.train(cfg=cfg)
        return val_loss
