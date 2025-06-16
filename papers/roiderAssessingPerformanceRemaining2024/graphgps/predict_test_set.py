import torch
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.model_builder import create_model
import pandas as pd
import numpy as np
from torch_geometric.graphgym.loader import create_loader
from types import SimpleNamespace
import joblib


def predict(pretrained_model_filepath,
            normalization_factor,
            cfg_location,
            best_trial,
            data_location):
    args = SimpleNamespace(cfg_file=cfg_location,
                           repeat=1,
                           mark_done=False,
                           opts=[]
                           )

    cfg.set_new_allowed(True)
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, args)
    config = joblib.load(data_location + 'config.pkl')
    oh_encoder = joblib.load(data_location + 'oh_encoders.pkl')
    unique_activities_train = len(oh_encoder[config['activity_column']].categories_[0])
    cfg.dataset.node_encoder_num_types = unique_activities_train + 1
    tr = joblib.load(data_location + '/graph_dataset/raw/train.pickle')
    cfg.two_layer_linear_edge_encoder.in_dim = int(tr[0]['edge_attr'].shape[1])
    cfg.train.batch_size = 1
    cfg.train.eval_period = 1
    cfg.train.enable_ckpt = True
    cfg.train.ckpt_best = True
    cfg.dataset.dir = data_location + '/graph_dataset/'

    cfg.posenc_LapPE.dim_pe = int(best_trial['params_posenc_LapPE.dim_pe'])
    times_func_end = int(best_trial['params_posenc_RWSE.kernel.times_func_end'])
    cfg.posenc_RWSE.kernel.times_func = 'range(1, ' + str(times_func_end) + ')'
    cfg.accelerator = 'cuda:0'
    dump_cfg(cfg)

    loaders = create_loader()
    dataloader = loaders[2]
    model = create_model()
    loaded_checkpoint = torch.load(pretrained_model_filepath)
    model_state_dict = loaded_checkpoint['model_state']
    # update the model parameters by those obtained from check point file.
    model.load_state_dict(model_state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set the device
    model.to(device)  # move the model
    model.eval()  # Set the model to evaluation mode
    # set empty lists to create the final dataframe for each fold (run)
    num_node_list, num_edge_list, real_reamining_times, predictions = [], [], [], []

    print('batch size:', cfg.train.batch_size)

    with torch.no_grad():
        for each_graph in dataloader:
            each_graph.to(device)  # move the test example to device
            graph_transformer_prediction = model(each_graph)  # get prediction of the model
            predictions.append(float(np.array(graph_transformer_prediction[0].cpu())))  # tuple of value & device
            num_node_list.append(each_graph.x.shape[0])  # get number of nodes
            num_edge_list.append(each_graph.edge_attr.shape[0])  # get number of edges
            real_reamining_times.append(np.array(each_graph.y[0].cpu()))  # get real reamining time

    aggregated_graph_info = {'num_node': num_node_list,
                             'num_edge': num_edge_list,
                             'real_cycle_time': real_reamining_times,
                             'predicted_cycle_time': predictions}
    prediction_dataframe = pd.DataFrame(aggregated_graph_info)  # convert to dataframe
    prediction_dataframe['preds'] = prediction_dataframe['predicted_cycle_time'] * normalization_factor
    prediction_dataframe['labels'] = prediction_dataframe['real_cycle_time'] * normalization_factor
    return prediction_dataframe
