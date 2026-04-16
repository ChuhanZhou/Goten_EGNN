from tool.data_loader import DatasetLoader,download,has_file

import os
from tqdm import tqdm
import datetime
import numpy as np
import torch
from torch_cluster import radius_graph
from torch.utils.data import random_split, Subset
import pandas as pd
import re

source_file_urls = {
    "tetrapeptide": "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_Ac-Ala3-NHMe.npz",
    "dha": "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_DHA.npz",
    "stachyose": "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_stachyose.npz",
    "at-at": "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_AT-AT.npz",
    "at-at-cg-cg": "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_AT-AT-CG-CG.npz",
    "buckyball-catcher": "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_buckyball-catcher.npz",
    "double-walled-nanotube": "https://sgdml.org/secure_proxy.php?file=repo/datasets/md22_double-walled_nanotube.npz",
}

atom_id_dict = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}

train_data_nums = {
    "tetrapeptide": 6000,
    "dha": 8000,
    "stachyose": 8000,
    "at-at": 3000,
    "at-at-cg-cg": 2000,
    "buckyball-catcher": 600,
    "double-walled-nanotube": 8000,
}

class Loader(DatasetLoader):
    def __init__(self):
        super().__init__()

    def load_unsorted_data(self, folder_path, type_list, cutoff=None, atom_mass_dict=None, use_tqdm=True, key=None):
        dataset = {}
        atom_set = set()

        raw_path = "{}/raw".format(folder_path)
        for mol_label in source_file_urls:
            if key is not None and mol_label != key:
                continue

            data_url = source_file_urls[mol_label]
            file_name = os.path.basename(data_url)
            file_path = "{}/{}".format(raw_path, file_name)

            if not has_file(file_path):
                download(data_url, raw_path)

            sub_dataset,sub_atom_set = self.load_from_npz(file_path,type_list,cutoff,atom_mass_dict,use_tqdm)
            dataset[mol_label] = sub_dataset
            atom_set.update(sub_atom_set)

        print("Atom types: {}".format(atom_set))
        return dataset

    def split_data(self, dataset, split_nums, seed, folder_path, key):
        split_g = torch.Generator().manual_seed(seed)
        split_nums = [
            train_data_nums[key],
            round(train_data_nums[key]*split_nums[1]),
            len(dataset)-train_data_nums[key]-round(train_data_nums[key]*split_nums[1])]
        return random_split(dataset, split_nums, generator=split_g)[0:3]

    def load_from_npz(self, npz_file_path, type_list, cutoff=None, atom_mass_dict=None,use_tqdm=True):
        dataset = []
        atom_set = set()

        if not has_file(npz_file_path):
            print("Cannot find {}".format(npz_file_path))
            return dataset, atom_set

        npz_data = np.load(npz_file_path, allow_pickle=True)
        src_data = npz_data.f

        ids = list(range(len(src_data.E)))

        if use_tqdm:
            progress_bar = tqdm(desc="[{}] Loading data from {}".format(datetime.datetime.now(),npz_file_path), total=len(ids))
        else:
            print("[{}] Loading data from {}".format(datetime.datetime.now(),npz_file_path))

        nuclear_charges = src_data.z
        coords = src_data.R
        energies = src_data.E
        forces = src_data.F

        atoms_mapping = {k: type_list.index(v) for k,v in atom_id_dict.items()}
        set_atoms_type = np.vectorize(atoms_mapping.get)(nuclear_charges)
        atom_set = {type_list[i] for i in set(set_atoms_type.tolist())}

        for i in range(len(ids)):
            if use_tqdm:
                progress_bar.update()

            id = ids[i]
            atoms_xyz = coords[i,:,:]

            atoms_xyz = torch.tensor(np.array(atoms_xyz), dtype=torch.float32)

            prop = {
                "e&f": [energies[i],forces[i,:,:]],
            }

            edge_index = radius_graph(atoms_xyz, r=cutoff, loop=False, max_num_neighbors=32)  # [j,i]

            atoms_pos = atoms_xyz
            atoms_type = set_atoms_type
            if atom_mass_dict is not None:
                masses = torch.tensor(np.array([atom_mass_dict[type_list[i]] for i in atoms_type]).reshape(-1, 1),dtype=torch.float32)
                mass_center = (masses * atoms_xyz).sum(dim=0) / masses.sum()
                atoms_pos = atoms_xyz - mass_center

            atoms_type = torch.tensor(atoms_type, dtype=torch.int).unsqueeze(1)
            dataset.append([id, atoms_pos, atoms_type, edge_index, prop])

        if use_tqdm:
            progress_bar.close()
        return dataset, atom_set