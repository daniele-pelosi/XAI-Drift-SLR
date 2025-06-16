import datetime
import logging
import graphgps  # noqa, register custom modules
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule
from torch_geometric.graphgym.train import train as train_gg
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything
from graphgps.logger import create_logger
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
import pandas as pd


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period)


def train(cfg):

    set_printing()
    seed_everything(cfg.seed)
    auto_select_device()
    logging.info(f"    Starting now: {datetime.datetime.now()}")
    # Set machine learning pipeline
    loaders = create_loader()
    loggers = create_logger()
    model = create_model()

    optimizer = create_optimizer(model.parameters(),
                                 new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)

    # Start training
    if cfg.train.mode == 'standard':
        if cfg.wandb.use:
            logging.warning("[W] WandB logging is not supported with the "
                            "default train.mode, set it to `custom`")
        datamodule = GraphGymDataModule()
        train_gg(model, datamodule, logger=True)
    else:
        train_dict[cfg.train.mode](loggers,
                                   loaders,
                                   model,
                                   optimizer,
                                   scheduler)

    # Return best validation loss
    df = pd.read_json(cfg.out_dir + '/' + cfg.run_id + '/val/stats.json',
                      lines=True)
    return float(df['loss'].min())
