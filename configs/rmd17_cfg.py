from tool import rmd17_loader

import torch.nn as nn

config = {
    'title':"rmd17_B",
    'dataset_path':"./dataset/rmd17",
    'data_loader':rmd17_loader.Loader(),
    'train_size':950,
    'val_size':50,
    'test_size':1000,
    'split_key':1,
    'preprocess': True,
    'test_in_train':True,

    'atom_types': ['H', 'C', 'N', 'O'],

    'warmup': 1000,
    'lr_max': 2e-4,
    'lr_decay': 0.8,
    'lr_patience': 30,
    'loss_func':nn.MSELoss(),
    'loss_weights':[0.05,0.95],
    'grad_clip': 10.0,
    'batch_size': 4,
    'epochs': 3000,
    #'weight_decay': 0.01,
    #'dropout': 0.1,
    'stop_patience': 1000, # early stop

    'node_dim': 192, # d_ne
    'edge_dim': 192, # d_ed
    'edge_ref_dim': 768, # d_xpd

    'layer_num':12,
    'rbf_num':32,

    'predict_label': "e&f",
    'mol_type': "aspirin"
}