from configs.config import config as cfg
from tool.data_loader import download,has_file,ensure_dir

import torch
import torch.nn as nn
import numpy as np
import random
import json
import math
import os

def collate_fn(batch):
    batch_atoms_pos = torch.zeros((0, 3))
    batch_atoms_types = torch.zeros((0,1),dtype=torch.int)

    batch_prop = []
    batch_edge_index = torch.zeros((2,0),dtype=torch.int)
    atoms_batch_index = []
    has_sub_prop = False
    for i, (id, atoms_pos, atoms_types, edge_index, prop) in enumerate(batch):
        edge_index = edge_index + batch_atoms_types.shape[0]

        batch_atoms_pos = torch.cat([batch_atoms_pos,atoms_pos], dim=0)

        batch_atoms_types = torch.cat([batch_atoms_types,atoms_types], dim=0)

        if isinstance(prop[cfg["predict_label"]], list):
            has_sub_prop = True
            for s_i,sub_prop in enumerate(prop[cfg["predict_label"]]):
                if len(batch_prop) <= s_i:
                    batch_prop.append([])

                batch_prop[s_i].append(sub_prop if isinstance(sub_prop, np.ndarray) else [sub_prop])
        else:
            batch_prop.append([prop[cfg["predict_label"]]])
        batch_edge_index = torch.cat([batch_edge_index,edge_index], dim=1)
        atoms_batch_index.append(torch.ones(len(atoms_types),dtype=torch.int)*i)

    if has_sub_prop:
        for s_i,sub_prop in enumerate(batch_prop):
            batch_prop[s_i] = torch.tensor(np.array(batch_prop[s_i]),dtype=torch.float32)
            batch_prop[s_i] = batch_prop[s_i].view(-1,batch_prop[s_i].shape[-1])
    else:
        batch_prop = torch.tensor(batch_prop,dtype=torch.float32)
    atoms_batch_index = torch.cat(atoms_batch_index)
    return batch_atoms_pos, batch_atoms_types, batch_edge_index, batch_prop, atoms_batch_index

def get_mean_std(prop_dict_list,prop_labels = None):
    prop_values = []
    for prop_dict in prop_dict_list:
        if prop_labels is None:
            prop_labels = list(prop_dict.keys())
        prop_values.append([prop_dict[l] if l != "e&f" else prop_dict[l][0] for l in prop_labels])

    prop_values = np.array(prop_values)

    mean = prop_values.mean(axis=0)
    std = prop_values.std(axis=0)

    return {l: (mean[i],std[i]) for i,l in enumerate(prop_labels) }

def unit_Ha2meV(ha):
    return ha * 27211.386245981

def load_atom_mass(file_path=None):
    if file_path == None:
        file_path = cfg["atom_mass"]["path"]

    dir = os.path.dirname(file_path)
    ensure_dir(dir)
    if not has_file(file_path):
        url = cfg["atom_mass"]["url"]
        download(url,dir,rename=os.path.basename(file_path))

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    columns = data['Table']['Columns']['Column']
    rows = data['Table']['Row']

    atom_mass_dict = {}

    for row in rows:
        values = row['Cell']
        value_dict = dict(zip(columns, values))
        atom_mass_dict[value_dict['Symbol']] = float(value_dict['AtomicMass'])

    return atom_mass_dict



