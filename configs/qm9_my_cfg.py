from configs.qm9_cfg import config as qm9_cfg

sub_config = {
    'title':"qm9_M",
    'layer_num':6,

    'body_max': 4,
    'mult_body_dim': 64,
    'node_dim': 256, # d_ne
    'edge_dim': 256, # d_ed
    'edge_ref_dim': 256, # d_xpd
}

config = qm9_cfg.copy()
config.update(sub_config)