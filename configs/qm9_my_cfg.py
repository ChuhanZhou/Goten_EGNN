from configs.qm9_cfg import config as qm9_cfg

sub_config = {
    'title':"qm9_M",
    'layer_num':2,
}

config = qm9_cfg.copy()
config.update(sub_config)