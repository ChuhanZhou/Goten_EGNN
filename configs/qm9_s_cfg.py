from configs.qm9_cfg import config as qm9_cfg
from tool import qm9_torch_loader

config = {
    'title':"QM9_S",
    'layer_num':4,

    #'dataset_path':"./dataset/qm9",
    #'data_loader':qm9_torch_loader.Loader(),
}

qm9_cfg.update(config)
config = qm9_cfg