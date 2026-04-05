import os
from abc import ABC, abstractmethod
from torch_geometric.data import download_url,extract_zip
import numpy as np
import rdkit
import os.path as osp
import torch
from rdkit import Chem,RDLogger
from tqdm import tqdm
import datetime
from torch_cluster import radius_graph

#RDLogger.DisableLog('rdApp.warning')

class DatasetLoader(ABC):
    def load(self,folder_path,type_list,cutoff=None,atom_mass_dict=None,use_tqdm=True):
        dataset = self.load_unsorted_data(folder_path,type_list,cutoff,atom_mass_dict,use_tqdm)

        # sort dataset to ensure the same sequence on different devices
        dataset.sort(key=lambda x: x[0])

        return dataset

    def load_from_sdf(self,sdf_file_path, prop_list, type_list, cutoff=None, atom_mass_dict=None, atomrefs=None, use_tqdm=True, init_index=0, skip_list=[]):
        dataset = []
        atom_set = set()

        if not has_file(sdf_file_path):
            print("Cannot find {}".format(sdf_file_path))
            return dataset,atom_set

        supplier = Chem.SDMolSupplier(sdf_file_path, removeHs=False, sanitize=False)

        if use_tqdm:
            progress_bar = tqdm(desc="[{}] Loading data from {}".format(datetime.datetime.now(),sdf_file_path), total=len(supplier))
        else:
            print("[{}] Loading data from {}".format(datetime.datetime.now(),sdf_file_path))

        for i,mol in enumerate(supplier):
            if use_tqdm:
                progress_bar.update()

            if i in skip_list:
                continue
            if mol is None:
                continue

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

            for target in atomrefs:
                ref_energy = np.array(atomrefs[target],dtype=np.float32)[atoms_type].sum()
                prop[target] = prop[target] - unit_u2mu(ref_energy)

            atoms_pos = atoms_xyz
            if atom_mass_dict is not None:
                masses = torch.tensor(np.array([atom_mass_dict[type_list[i]] for i in atoms_type]).reshape(-1, 1), dtype=torch.float32)
                mass_center = (masses * atoms_xyz).sum(dim=0) / masses.sum()
                atoms_pos = atoms_xyz - mass_center

            atoms_type = torch.tensor(atoms_type, dtype=torch.int).unsqueeze(1)
            dataset.append([prop_list[index]["id"], atoms_pos, atoms_type, ij_pos_vecs, edge_index, prop])

        if use_tqdm:
            progress_bar.close()
        return dataset, atom_set

    @abstractmethod
    def load_unsorted_data(self, folder_path, type_list ,cutoff=None, atom_mass_dict=None, use_tqdm=True):
        pass

def has_file(file_path):
    if file_path == None:
        return False
    return os.path.isfile(file_path)

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def download(url,dir,rename=None,extract=False):
    ensure_dir(dir)
    file_path = download_url(url, dir)

    if extract:
        extract_zip(file_path, dir)
        os.unlink(file_path)

    if rename is not None:
        os.rename(osp.join(dir, os.path.basename(url)),osp.join(dir, rename))

def unit_Ha2meV(ha):
    return ha * 27211.386245981

def unit_u2mu(a):
    return a*1e3