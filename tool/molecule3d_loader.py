from tool.data_loader import DatasetLoader,has_file,ensure_dir

import os
from tqdm import tqdm
import datetime
import numpy as np
import torch
from torch.utils.data import Subset
import pandas as pd
import json

data_url = "https://drive.google.com/drive/u/2/folders/1y-EyoDYMvWZwClc2uvXrM4_hQBtM85BI?usp=sharing"
source_files = [
    "combined_mols_0_to_1000000.sdf",
    "combined_mols_1000000_to_2000000.sdf",
    "combined_mols_2000000_to_3000000.sdf",
    "combined_mols_3000000_to_3899647.sdf",
    "properties.csv",
    "random_split_inds.json",
    "random_test_split_inds.json",
    "scaffold_split_inds.json",
    "scaffold_test_split_inds.json",
]

class Loader(DatasetLoader):
    def __init__(self):
        super().__init__()

    def load_unsorted_data(self, folder_path, type_list, cutoff=None, atom_mass_dict=None, use_tqdm=True, key=None):
        raw_path = "{}/raw".format(folder_path)
        ensure_dir(raw_path)
        for file in source_files:
            if not has_file("{}/{}".format(raw_path, file)):
                raise FileNotFoundError("Cannot find {} in {}. Please download source files from {}".format(file,raw_path,data_url))

        prop_list = self.load_from_csv("{}/{}".format(raw_path,source_files[4]),use_tqdm=use_tqdm)

        dataset = []
        atom_set = set()

        for sdf_name in source_files[0:4]:
            file_path = "{}/{}".format(raw_path, sdf_name)
            sub_dataset, sub_atom_set = self.load_from_sdf(file_path, prop_list, type_list, cutoff, atom_mass_dict, use_tqdm=use_tqdm, init_index=len(dataset))
            dataset.extend(sub_dataset)
            atom_set.update(sub_atom_set)

        print("Atom types: {}".format(atom_set))
        return dataset

    def load_from_csv(self,csv_file_path,use_tqdm=True):

        df = pd.read_csv(csv_file_path)

        start_info = "[{}] Loading prop data from {}".format(datetime.datetime.now(), csv_file_path)
        if use_tqdm:
            iterator = tqdm(range(df.shape[0]), desc=start_info) if use_tqdm else range(df.shape[0])
        else:
            print(start_info)
            iterator = range(df.shape[0])

        id_data = df["cid"].values
        dipole_data = list(df[["dipole x","dipole y","dipole z"]].values.astype(np.float32))
        homo_data = df["homo"].values.astype(np.float32)
        lumo_data = df["lumo"].values.astype(np.float32)

        prop_list = [{
            "id": id_data[i],
            "prop": {
                "mu": np.linalg.norm(dipole_data[i], ord=2),
                "mu_3d": dipole_data[i],
                "homo": homo_data[i],
                "lumo": lumo_data[i],
            }} for i in iterator]
        return prop_list

    def split_data(self, dataset, split_nums, seed, folder_path, key):
        # key in ["random","scaffold"]
        raw_path = "{}/raw".format(folder_path)
        match key:
            case "random":
                set_file = "{}/{}".format(raw_path, source_files[5])
            case "scaffold":
                set_file = "{}/{}".format(raw_path, source_files[7])
            case other:
                raise NotImplementedError("Don't know how to split data based on [{}]".format(key))

        with open(set_file, 'r', encoding='utf-8') as f:
            subsets = json.load(f)

        train_set = Subset(dataset, subsets["train"])
        val_set = Subset(dataset, subsets["valid"])
        test_set = Subset(dataset, subsets["test"])
        return train_set, val_set, test_set