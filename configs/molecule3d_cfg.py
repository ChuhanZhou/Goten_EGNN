from tool import molecule3d_loader

import torch.nn as nn

#download from https://drive.google.com/uc?id=1C_KRf8mX-gxny7kL9ACNCEV4ceu_fUGy
config = {
    'title':"molecule3d_B",
    'dataset_path':"./dataset/molecule3d",
    'data_loader':molecule3d_loader.Loader(),

    'atom_types': ['K', 'Kr', 'Mg', 'B', 'V', 'Cu', 'Ni', 'Ge', 'Mn', 'C', 'He', 'Na', 'Cl', 'Br', 'Si', 'Se', 'S', 'Co', 'Ca', 'H', 'Be', 'As', 'Ti', 'O', 'N', 'Al', 'P', 'Ga', 'Cr', 'F', 'Ar', 'Sc', 'Zn', 'Li', 'Ne', 'Fe'],

    'warmup': 5000,
    'lr_max': 1e-4,
    'lr_decay': 0.8,
    'lr_patience': 5,
    'loss_func':nn.L1Loss(),
    'grad_clip': None,
    'batch_size': 256,
    'epochs': 300,
    #'weight_decay': 0.01,
    #'dropout': 0.1,
    'stop_patience': 25, # early stop

    'node_dim': 384, # d_ne
    'edge_dim': 384, # d_ed
    'edge_ref_dim': 384, # d_xpd

    'layer_num':12,
    'rbf_num':32,

    'predict_label': "homo"
}