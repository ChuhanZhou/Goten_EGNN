from configs.config import config as cfg
from models.goten_net import GotenNet
from tool.utils import collate_fn,load_atom_mass

from tqdm import tqdm
import datetime
import torch
import time
from torch.utils.data import random_split, Subset
import numpy as np
import random
import re

def test(model,dataset,title="testing",use_tqdm=True):
    device = cfg['device']
    batch_size = cfg['batch_size']

    model.to(device)
    model.eval()
    if device != 'cpu':
        torch.cuda.empty_cache()

    test_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if use_tqdm:
        time.sleep(0.01)
        progress_bar = tqdm(desc="[{}] [{}]".format(datetime.datetime.now(),title), total=len(test_dataloader))

    total_loss = 0
    out_list = []
    target_list = []
    with torch.inference_mode():
        for i, data in enumerate(test_dataloader):
            [mass_center_vecs, atoms_type, ij_vecs, edge_index, prop_value, atoms_batch_index] = data

            out = model(mass_center_vecs.to(device), atoms_type.to(device), ij_vecs.to(device), edge_index.to(device), atoms_batch_index.to(device))

            std_prop_value = model.standardize(prop_value.to(device))
            loss = cfg["loss_func"](out, std_prop_value)

            total_loss += loss.item() * out.shape[0]

            out = model.destandardize(out)
            out_list.append(out.cpu().detach().numpy())
            target_list.append(prop_value.cpu().numpy())

            if use_tqdm:
                progress_bar.update()

    if use_tqdm:
        progress_bar.close()
    avg_loss = total_loss / len(dataset)

    outs = np.concatenate(out_list,axis=0)
    targets = np.concatenate(target_list,axis=0)
    avg_mae = np.mean(np.abs(targets - outs))

    out_labels = np.concatenate([outs,targets],axis=1)
    return avg_loss,avg_mae,out_labels

