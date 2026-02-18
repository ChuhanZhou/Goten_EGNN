import torch
import torch.nn as nn
from configs import qm9_cfg as ds_cfg

config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'seed': 1,

    'atom_types': ['H', 'N', 'F', 'C', 'O'],

    'dataset_cfg':ds_cfg,

    'weight_decay': 0.01,
    'dropout': 0.1,

    'degree_max': 2,
    'attention_heads': 8,
    'cutoff_radius': 5.0,

    'activation': nn.ReLU(),
}

if 'dataset_cfg' in config.keys() and config['dataset_cfg']:
    config.update(config['dataset_cfg'].config)