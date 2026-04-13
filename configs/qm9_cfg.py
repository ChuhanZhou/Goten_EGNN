from tool import qm9_loader

import torch.nn as nn

config = {
    'title':"qm9_B",
    'dataset_path':"./dataset/qm9",
    'data_loader':qm9_loader.Loader(),
    'train_size':110000,
    'val_size':10000,
    'test_size':10831,
    'preprocess': True,
    'test_in_train':True,

    'atom_types': ['H', 'C', 'N', 'O', 'F'],

    'warmup': 10000,
    'lr_max': 1e-4,
    'lr_decay': 0.8,
    'lr_patience': 15,
    'loss_func':nn.MSELoss(),
    'grad_clip': 10.0,
    'batch_size': 32,
    'epochs': 1000,
    #'weight_decay': 0.01,
    #'dropout': 0.1,
    'stop_patience': 150, # early stop

    'node_dim': 256, # d_ne
    'edge_dim': 256, # d_ed
    'edge_ref_dim': 256, # d_xpd

    'layer_num':6,
    'rbf_num':64,

    'predict_label': "alpha"
}