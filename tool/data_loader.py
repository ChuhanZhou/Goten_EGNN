import os
from abc import ABC, abstractmethod
from torch_geometric.data import download_url,extract_zip
import numpy as np
from torch.utils.data import Subset
import os.path as osp
import torch
from rdkit import Chem,RDLogger
from tqdm import tqdm
import datetime
from torch_cluster import radius_graph
import gzip
import tarfile

RDLogger.DisableLog('rdApp.warning')

class DatasetLoader(ABC):
    def load(self,folder_path,type_list,cutoff=None,atom_mass_dict=None,use_tqdm=True,preprocess=False, key=None):
        if preprocess:
            dataset = load_processed_data(folder_path)
            if dataset is not None:
                return dataset

        dataset = self.load_unsorted_data(folder_path,type_list,cutoff,atom_mass_dict,use_tqdm,key)

        # sort dataset to ensure the same sequence on different devices
        # dataset.sort(key=lambda x: x[0])

        if preprocess:
            save_processed_data(dataset, folder_path)
        return dataset

    @abstractmethod
    def load_unsorted_data(self, folder_path, type_list, cutoff=None, atom_mass_dict=None, use_tqdm=True, key=None):
        pass

    @abstractmethod
    def split_data(self,dataset,split_nums,seed,folder_path,key):
        pass

    def load_from_sdf(self,sdf_file_path, prop_list, type_list, cutoff=None, atom_mass_dict=None, atomrefs={}, use_tqdm=True, init_index=0, skip_list=[]):
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
                print("None at {}".format(i))
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

            for target in atomrefs:
                ref_energy = np.array(atomrefs[target],dtype=np.float32)[atoms_type].sum()
                prop[target] = prop[target] - unit_u2mu(ref_energy)

            atoms_pos = atoms_xyz
            if atom_mass_dict is not None:
                masses = torch.tensor(np.array([atom_mass_dict[type_list[i]] for i in atoms_type]).reshape(-1, 1), dtype=torch.float32)
                mass_center = (masses * atoms_xyz).sum(dim=0) / masses.sum()
                atoms_pos = atoms_xyz - mass_center

            atoms_type = torch.tensor(atoms_type, dtype=torch.int).unsqueeze(1)
            dataset.append([prop_list[index]["id"], atoms_pos, atoms_type, edge_index, prop])

        if use_tqdm:
            progress_bar.close()
        return dataset, atom_set

def split_data_by_ids(dataset,split_ids):
    subsets = []
    for i,data in enumerate(dataset):
        id = data[0]
        for set_i,sub_ids in enumerate(split_ids):
            if len(subsets)<=set_i:
                subsets.append([])
            if id in sub_ids:
                subsets[set_i].append(i)
    return [Subset(dataset, sub_index) for sub_index in subsets]

def has_file(file_path):
    if file_path == None:
        return False
    return os.path.isfile(file_path)

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def download(url,dir,rename=None,extract=None):
    ensure_dir(dir)
    file_path = download_url(url, dir)

    match extract:
        case "zip":
            extract_zip(file_path, dir)
            os.unlink(file_path)
        case "bz2":
            with tarfile.open(file_path, "r:bz2") as tar:
                tar.extractall(path=dir)
            os.remove(file_path)

    if rename is not None:
        os.rename(osp.join(dir, os.path.basename(url)),osp.join(dir, rename))

def save_processed_data(data,dir_path,use_zip=False):
    pre_path = "{}/preprocessed".format(dir_path)
    ensure_dir(pre_path)
    if use_zip:
        with gzip.open("{}/preprocessed.pt.gz".format(pre_path), "wb") as f:
            torch.save(data, f)
    else:
        torch.save(data,"{}/preprocessed.pt".format(pre_path))

def load_processed_data(dir_path,use_zip=False):
    pre_path = "{}/preprocessed".format(dir_path)
    pre_file = "{}/preprocessed.pt.gz".format(pre_path) if use_zip else "{}/preprocessed.pt".format(pre_path)
    if has_file(pre_file):
        print("[{}] Loading preprocessed data from {}".format(datetime.datetime.now(), pre_file))
        if use_zip:
            with gzip.open(pre_file, "rb") as f:
                data = torch.load(f, weights_only=False)
        else:
            data = torch.load(pre_file, weights_only=False)
        return data
    return None

def unit_Ha2meV(ha):
    return unit_u2mu(ha * 27.211386246)

def unit_u2mu(a):
    return a * 1e3