import torch
import torch.nn as nn
from configs import qm9_cfg as ds_cfg

config = {
    'log_path': "./log",
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'seed': 1,
    'train_size':0.8,
    'val_size':0.1,
    'test_size':0.1,
    'test_in_train':True,

    'atom_types': ['H', 'N', 'F', 'C', 'O'],

    'dataset_cfg':ds_cfg,

    'grad_clip': None,
    'weight_decay': 0.01,
    'dropout': 0.1, # for self-attention only

    'degree_max': 2,
    'attention_heads': 8,
    'cutoff_radius': 5.0,

    'activation': nn.SiLU(),
    'weight_init': nn.init.xavier_uniform_,
    'bias_init': nn.init.zeros_,
    'combine_heads': False,
}

if 'dataset_cfg' in config.keys() and config['dataset_cfg']:
    config.update(config['dataset_cfg'].config)