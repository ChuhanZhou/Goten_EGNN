from tool import md22_loader

import torch.nn as nn

config = {
    'title':"md22",
    'dataset_path':"./dataset/md22",
    'data_loader':md22_loader.Loader(),
    'train_size':0,
    'val_size':1/3,
    'test_size':0,
    'split_key':None,
    'preprocess': False,
    'test_in_train':False,

    'atom_types': ['H', 'C', 'N', 'O'],

    'warmup': 1000,
    'lr_max': 1e-4,
    'lr_decay': 0.8,
    'lr_patience': 30,
    'loss_func':nn.MSELoss(),
    'loss_weights':[0.05,0.95],
    'grad_clip': 5.0,
    'batch_size': 4,
    'epochs': 3000,
    #'weight_decay': 0.01,
    #'dropout': 0.1,
    'stop_patience': 1000, # early stop

    'node_dim': 384, # d_ne
    'edge_dim': 384, # d_ed
    'edge_ref_dim': 768, # d_xpd

    'layer_num':8,
    'rbf_num':32,

    'predict_label': "e&f",
    'mol_type': "tetrapeptide"
}

config['split_key'] = config['mol_type']