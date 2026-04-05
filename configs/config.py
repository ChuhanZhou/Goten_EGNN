import torch
import torch.nn as nn
from configs import qm9_cfg,qm9_s_cfg,molecule3d_cfg

config = {
    'title': None,
    'atom_mass_path': "./dataset/PubChemElements_all.json", # download from https://pubchem.ncbi.nlm.nih.gov/ptable/atomic-mass/
    'log_path': "./log",
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'seed': 1,
    'train_size':0.8,
    'val_size':0.1,
    'test_size':0.1,
    'dataset_random_split':True,
    'dataset_set_split':False,
    'test_in_train':True,

    'model_type':None,
    'atom_types': ['H', 'C', 'N', 'O', 'F'],

    'grad_clip': None,
    'batch_size': 32,
    'epochs': 1000,
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
    #'vec_rej': False,

    'predict_label': None,
}

config['high_degree_sizes'] = [2*i+1 for i in range(1,config['degree_max']+1)]

def update_model_cfg(cfg_type):
    config['model_type'] = cfg_type
    match cfg_type:
        case 'qm9':
            config.update(qm9_cfg.config)
        case 'qm9_s':
            config.update(qm9_s_cfg.config)
        case 'molecule3d':
            config.update(molecule3d_cfg.config)
        case _:
            print('unknown model config: {}'.format(cfg_type))