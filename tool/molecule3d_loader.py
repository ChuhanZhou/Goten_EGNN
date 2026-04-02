from tool.data_loader import DatasetLoader

import os
from tqdm import tqdm
import datetime
import numpy as np
import torch
from rdkit import Chem,RDLogger
from torch_cluster import radius_graph
import pandas as pd

RDLogger.DisableLog('rdApp.warning')

sdf_files = [
    "combined_mols_0_to_1000000.sdf",
    "combined_mols_1000000_to_2000000.sdf",
    "combined_mols_2000000_to_3000000.sdf",
    "combined_mols_3000000_to_3899647.sdf",
]

class Loader(DatasetLoader):
    def __init__(self):
        super().__init__()

    def load_unsorted_data(self, folder_path, type_list, cutoff=None, atom_mass_dict=None, use_tqdm=True):
        pre_file = "{}/preprocessed/preprocessed_data.pt".format(folder_path)
        if self.has_file(pre_file):
            print("[{}] Loading preprocessed data from {}".format(datetime.datetime.now(),pre_file))
            dataset = torch.load(pre_file, weights_only=False)
            return dataset

        dataset = []
        atom_set = set()

        prop_list = self.load_from_csv("{}/raw/properties.csv".format(folder_path))

        for sdf_name in sdf_files:
            file_path = "{}/raw/{}".format(folder_path, sdf_name)
            sub_dataset, sub_atom_set = self.load_from_sdf(file_path, prop_list, type_list, cutoff, atom_mass_dict, use_tqdm, len(dataset))
            dataset.extend(sub_dataset)
            atom_set.update(sub_atom_set)

        print("Atom types: {}".format(atom_set))
        torch.save(dataset,"{}/preprocessed/preprocessed_data.pt".format(folder_path))
        return dataset

    def load_from_csv(self,csv_file_path):
        print("[{}] Loading prop data from {}".format(datetime.datetime.now(), csv_file_path))
        df = pd.read_csv(csv_file_path)
        id_data = df["cid"].values
        dipole_data = list(df[["dipole x","dipole y","dipole z"]].values)
        homo_data = df["homo"].values
        lumo_data = df["homo"].values

        prop_list = [{
            "id": id_data[i],
            "prop": {
                "mu": np.linalg.norm(dipole_data[i], ord=2),
                "mu_3d": dipole_data[i],
                "homo": homo_data[i],
                "lumo": lumo_data[i],
            }} for i in range(df.shape[0])]
        return prop_list

    def load_from_sdf(self,sdf_file_path, prop_list, type_list, cutoff=None, atom_mass_dict=None, use_tqdm=True, init_index=0):
        dataset = []
        atom_set = set()

        if not self.has_file(sdf_file_path):
            print("Cannot find {}".format(sdf_file_path))
            return dataset,atom_set

        supplier = Chem.SDMolSupplier(sdf_file_path, sanitize=False)

        if use_tqdm:
            progress_bar = tqdm(desc="[{}] Loading data from {}".format(datetime.datetime.now(),sdf_file_path), total=len(supplier))
        else:
            print("[{}] Loading data from {}".format(datetime.datetime.now(),sdf_file_path))

        for i,mol in enumerate(supplier):
            if mol is None:
                continue

            if use_tqdm:
                progress_bar.update()

            index = init_index+i
            atoms_type = []
            atoms_xyz = []

            conf = mol.GetConformer()

            for atom in mol.GetAtoms():
                type = atom.GetSymbol()
                pos = conf.GetAtomPosition(atom.GetIdx())

                atom_set.add(type)
                atoms_type.append(type_list.index(type))
                atoms_xyz.append([pos.x, pos.y, pos.z])

            atoms_xyz = torch.tensor(np.array(atoms_xyz), dtype=torch.float32)

            prop = prop_list[index]["prop"]

            edge_index = radius_graph(atoms_xyz, r=cutoff, loop=False, max_num_neighbors=32) #[j,i]
            ij_pos_vecs = atoms_xyz[edge_index[1]] - atoms_xyz[edge_index[0]]

            mass_center = None
            if atom_mass_dict is not None:
                masses = torch.tensor(np.array([atom_mass_dict[type_list[i]] for i in atoms_type]).reshape(-1, 1), dtype=torch.float32)
                mass_center = (masses * atoms_xyz).sum(dim=0) / masses.sum()

            atoms_type = torch.tensor(atoms_type, dtype=torch.int).unsqueeze(1)
            dataset.append([prop_list[index]["id"], atoms_xyz, mass_center, atoms_type, ij_pos_vecs, edge_index, prop])

        if use_tqdm:
            progress_bar.close()
        return dataset, atom_set
