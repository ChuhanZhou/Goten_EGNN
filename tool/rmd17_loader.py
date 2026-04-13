from tool.data_loader import DatasetLoader,unit_Ha2meV,unit_u2mu,download,has_file

import os
from tqdm import tqdm
import datetime
import numpy as np
import torch
from torch_cluster import radius_graph
from torch.utils.data import random_split, Subset
import pandas as pd
import re

data_url = "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1"
source_files = [
    "rmd17/npz_data/rmd17_aspirin.npz",
    "rmd17/npz_data/rmd17_azobenzene.npz",
    "rmd17/npz_data/rmd17_benzene.npz",
    "rmd17/npz_data/rmd17_ethanol.npz",
    "rmd17/npz_data/rmd17_malonaldehyde.npz",
    "rmd17/npz_data/rmd17_naphthalene.npz",
    "rmd17/npz_data/rmd17_paracetamol.npz",
    "rmd17/npz_data/rmd17_salicylic.npz",
    "rmd17/npz_data/rmd17_toluene.npz",
    "rmd17/npz_data/rmd17_uracil.npz",
    "rmd17/splits/index_test_01.csv",
    "rmd17/splits/index_test_02.csv",
    "rmd17/splits/index_test_03.csv",
    "rmd17/splits/index_test_04.csv",
    "rmd17/splits/index_test_05.csv",
    "rmd17/splits/index_train_01.csv",
    "rmd17/splits/index_train_02.csv",
    "rmd17/splits/index_train_03.csv",
    "rmd17/splits/index_train_04.csv",
    "rmd17/splits/index_train_05.csv",
]

atom_id_dict = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}

class Loader(DatasetLoader):
    def __init__(self):
        super().__init__()

    def load_unsorted_data(self, folder_path, type_list, cutoff=None, atom_mass_dict=None, use_tqdm=True):
        raw_path = "{}/raw".format(folder_path)
        for file in source_files:
            if not has_file("{}/{}".format(raw_path, file)):
                download(data_url, raw_path, extract="bz2")

        dataset = {}
        atom_set = set()

        for npz_path in source_files[0:10]:
            mol_label = re.findall(r'(?<=rmd17_)\w*?(?=.npz)', npz_path)[0]
            file_path = "{}/{}".format(raw_path, npz_path)
            sub_dataset,sub_atom_set = self.load_from_npz(file_path,type_list,cutoff,atom_mass_dict,use_tqdm=use_tqdm)
            dataset[mol_label] = sub_dataset
            atom_set.update(sub_atom_set)

        print("Atom types: {}".format(atom_set))
        return dataset

    def split_data(self, dataset, split_nums, seed, folder_path, key):
        # key in [1,2,3,4,5]
        raw_path = "{}/raw".format(folder_path)

        if key in [1,2,3,4,5]:
            train_csv = pd.read_csv("{}/rmd17/splits/index_train_0{}.csv".format(raw_path, key), header=None)
            test_csv = pd.read_csv("{}/rmd17/splits/index_test_0{}.csv".format(raw_path, key), header=None)
        else:
            raise NotImplementedError("Don't know how to split data based on [{}]".format(key))

        train_val_index = list(train_csv.values.squeeze(-1))
        test_index = list(test_csv.values.squeeze(-1))

        split_g = torch.Generator().manual_seed(seed)
        train_index_set, val_index_set = random_split(train_val_index, split_nums[0:2], generator=split_g)
        train_index = [train_val_index[i] for i in train_index_set.indices]
        val_index = [train_val_index[i] for i in val_index_set.indices]

        train_set = Subset(dataset, train_index)
        val_set = Subset(dataset, val_index)
        test_set = Subset(dataset, test_index)
        return train_set, val_set, test_set

    def load_from_npz(self, npz_file_path, type_list, cutoff=None, atom_mass_dict=None,use_tqdm=True):
        dataset = []
        atom_set = set()

        if not has_file(npz_file_path):
            print("Cannot find {}".format(npz_file_path))
            return dataset, atom_set

        npz_data = np.load(npz_file_path, allow_pickle=True)
        src_data = npz_data.f

        if use_tqdm:
            progress_bar = tqdm(desc="[{}] Loading data from {}".format(datetime.datetime.now(),npz_file_path), total=len(src_data.old_indices))
        else:
            print("[{}] Loading data from {}".format(datetime.datetime.now(),npz_file_path))


        ids = src_data.old_indices
        nuclear_charges = src_data.nuclear_charges
        coords = src_data.coords
        energies = src_data.energies
        forces = src_data.coords

        atoms_mapping = {k: type_list.index(v) for k,v in atom_id_dict.items()}
        set_atoms_type = np.vectorize(atoms_mapping.get)(nuclear_charges)
        atom_set = {type_list[i] for i in set(set_atoms_type.tolist())}

        for i in range(len(ids)):
            if use_tqdm:
                progress_bar.update()

            id = ids[i]
            atoms_xyz = coords[i,:,:]

            atoms_xyz = torch.tensor(np.array(atoms_xyz), dtype=torch.float32)

            #prop = {
            #    "e&f": [energies[i],forces[i,:,:].sum(axis=0,dtype=np.float32)],
            #}
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