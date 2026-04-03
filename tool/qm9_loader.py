from tool.data_loader import DatasetLoader,unit_Ha2meV,unit_u2mu

import os
from tqdm import tqdm
import datetime
import numpy as np
import torch
from torch_cluster import radius_graph

# Atom reference energy
# ['H', 'C', 'N', 'O', 'F'] unit: eV
atomrefs = {
    "u0": [
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ],
    "u": [
        -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
        -2713.44632457
    ],
    "h": [
        -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
        -2713.42063702
    ],
    "g": [
        -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
        -2713.88796536
    ],
}

class Loader(DatasetLoader):
    def __init__(self):
        super().__init__()

    def prop_str2dict(self,prop_str):
        prop_list = prop_str.split()
        prop_dict = {
            #"tag":prop_list[0],
            #"index":prop_list[1],
            "mu": unit_u2mu(float(prop_list[5])), # dipole moment
            "alpha": unit_u2mu(float(prop_list[6])), # isotropic polarizability
            "homo": unit_Ha2meV(float(prop_list[7])),  # energy of homo
            "lumo": unit_Ha2meV(float(prop_list[8])),  # energy of lumo
            "gap": unit_Ha2meV(float(prop_list[9])),  # gap(lumo-homo)
            "r2": unit_u2mu(float(prop_list[10])), # electronic spatial extent
            "zpve": unit_Ha2meV(float(prop_list[11])), # zero point vibrational energy
            "u0": unit_Ha2meV(float(prop_list[12])), # internal energy at 0K
            "u": unit_Ha2meV(float(prop_list[13])),  # internal energy at 298.15K
            "h": unit_Ha2meV(float(prop_list[14])),  # enthalpy at 298.15K
            "g": unit_Ha2meV(float(prop_list[15])),  # free energy at 298.15K
            "cv": unit_u2mu(float(prop_list[16])),  # heat capacity at 298.15K
        }
        return prop_dict

    def xyz_str2float(self,xyz_str_list):
        f_list = []
        for s in xyz_str_list:
            f_list.append(float(s.replace("*^","e")))
        return f_list

    def load_unsorted_data(self,folder_path,type_list,cutoff=None,atom_mass_dict=None,use_tqdm=True):
        dataset = []
        if use_tqdm:
            progress_bar = tqdm(desc="[{}] Loading data from {}".format(datetime.datetime.now(),folder_path), total=len(os.listdir(folder_path)))
        atom_set = set()
        for name in os.listdir(folder_path):
            if use_tqdm:
                progress_bar.update()
            data_path = "{}/{}".format(folder_path, name)
            with open(data_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            atoms_type = []
            atoms_xyz = []
            atom_num = int(lines[0])
            for line in lines[2:2+atom_num]:
                atom_set.add(line.split()[0])
                atoms_type.append(type_list.index(line.split()[0]))
                atoms_xyz.append(self.xyz_str2float(line.split()[1:-1]))

            atoms_xyz = torch.tensor(np.array(atoms_xyz), dtype=torch.float32)

            prop = self.prop_str2dict(lines[1])
            smiles = lines[-2].split()[0]

            edge_index = radius_graph(atoms_xyz, r=cutoff, loop=False, max_num_neighbors=32)  # [j,i]
            ij_pos_vecs = atoms_xyz[edge_index[1]] - atoms_xyz[edge_index[0]]

            for target in atomrefs:
                ref_energy = np.array(atomrefs[target],dtype=np.float32)[atoms_type].sum()
                prop[target] = prop[target] - unit_u2mu(ref_energy)

            mass_center = None
            atoms_pos = atoms_xyz
            if atom_mass_dict is not None:
                masses = torch.tensor(np.array([atom_mass_dict[type_list[i]] for i in atoms_type]).reshape(-1, 1), dtype=torch.float32)
                mass_center = (masses * atoms_xyz).sum(dim=0) / masses.sum()
                atoms_pos = atoms_xyz - mass_center

            atoms_type = torch.tensor(atoms_type,dtype=torch.int).unsqueeze(1)

            dataset.append([name, atoms_pos, atoms_type, ij_pos_vecs, edge_index, prop])
        if use_tqdm:
            progress_bar.close()
        print("Atom types: {}".format(atom_set))
        return dataset