from tool import qm9_loader

import torch.nn as nn

#download from https://www.kaggle.com/datasets/zaharch/quantum-machine-9-aka-qm9
config = {
    'title':"QM9_B",
    'dataset_path':"./dataset/dsgdb9nsd.xyz",
    'data_loader':qm9_loader.Loader(),
    #'train_size':110000,
    #'val_size':10000,
    #'test_size':10000,

    'warmup': 10000,
    'lr_max': 1e-4,
    'lr_decay': 0.8,
    'lr_patience': 15,
    'loss_func':nn.MSELoss(),
    'grad_clip': 10,
    'batch_size': 32,
    'epochs': 1000,
    #'weight_decay': 0.01,
    #'dropout': 0.1,
    'node_dim': 256, #d_ne
    'edge_dim': 256, #d_ed
    'edge_ref_dim': 256, #d_xpd

    'layer_num':6,
    'rbf_num':64,

    'predict_label': "homo"
}