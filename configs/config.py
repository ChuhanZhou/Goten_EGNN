import torch
import torch.nn as nn
from configs import qm9_cfg as ds_cfg

config = {
    'atom_mass_path': "./dataset/PubChemElements_all.json", # download from https://pubchem.ncbi.nlm.nih.gov/ptable/atomic-mass/
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
    'stop_patience': None,

    'degree_max': 2,
    'high_degree_sizes':None,
    'attention_heads': 8,
    'cutoff_radius': 5.0,

    'activation': nn.SiLU(),
    'weight_init': nn.init.xavier_uniform_,
    'bias_init': nn.init.zeros_,
    'combine_heads': False,
    'vec_rej': False,
}

config['high_degree_sizes'] = [2*i+1 for i in range(1,config['degree_max']+1)]

if 'dataset_cfg' in config.keys() and config['dataset_cfg']:
    config.update(config['dataset_cfg'].config)