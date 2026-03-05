from tool.data_loader import DatasetLoader,unit_Ha2meV

import os
from tqdm import tqdm
import datetime
import numpy as np
import torch

class Loader(DatasetLoader):
    def __init__(self):
        super().__init__()

    def prop_str2dict(self,prop_str):
        prop_list = prop_str.split()
        prop_dict = {
            #"tag":prop_list[0],
            #"index":prop_list[1],
            "mu": float(prop_list[5]), # dipole moment,
            "alpha": float(prop_list[6]), # isotropic polarizability
            "homo": unit_Ha2meV(float(prop_list[7])),  # energy of homo
            "lumo": unit_Ha2meV(float(prop_list[8])),  # energy of lumo
            "gap": unit_Ha2meV(float(prop_list[9])),  # gap(lumo-homo)
            "r2": float(prop_list[10]), # electronic spatial extent
            "zpve": unit_Ha2meV(float(prop_list[11])), # zero point vibrational energy
            "u0": unit_Ha2meV(float(prop_list[12])), # internal energy at 0K
            "u": unit_Ha2meV(float(prop_list[13])),  # internal energy at 298.15K
            "h": unit_Ha2meV(float(prop_list[14])),  # enthalpy at 298.15K
            "g": unit_Ha2meV(float(prop_list[15])),  # free energy at 298.15K
            "cv": float(prop_list[16]),  # heat capacity at 298.15K
        }
        return prop_dict

    def xyz_str2float(self,xyz_str_list):
        f_list = []
        for s in xyz_str_list:
            f_list.append(float(s.replace("*^","e")))
        return f_list

    def load(self,folder_path,type_list):
        dataset = []
        progress_bar = tqdm(desc="[{}] Loading data from {}".format(datetime.datetime.now(),folder_path), total=len(os.listdir(folder_path)))
        atom_set = set()
        for name in os.listdir(folder_path):
            progress_bar.update()
            data_path = "{}/{}".format(folder_path, name)
            with open(data_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            atoms_type = []
            atoms_xyz = []
            atom_num = int(lines[0])
            for line in lines[2:2+atom_num]:
                atom_set.add(line.split()[0])
                atoms_type.append([type_list.index(line.split()[0])])
                atoms_xyz.append(self.xyz_str2float(line.split()[1:-1]))

            atoms_type = torch.tensor(atoms_type,dtype=torch.int64)
            atoms_xyz = np.array(atoms_xyz)

            prop = self.prop_str2dict(lines[1])
            smiles = lines[-2].split()[0]

            edge_index = [
                [], # source (j)
                []] # target (i)
            ij_pos_vecs = []
            for i in range(atom_num):
                for j in range(i+1,atom_num):
                    edge_index[0].append(j)
                    edge_index[1].append(i)
                    ij_pos_vecs.append(atoms_xyz[i,:]-atoms_xyz[j,:])

                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    ij_pos_vecs.append(atoms_xyz[j,:]-atoms_xyz[i,:])

            edge_index = torch.tensor(edge_index, dtype=torch.int64)
            ij_pos_vecs = torch.tensor(np.array(ij_pos_vecs),dtype=torch.float32)

            mass_center_dists = None

            dataset.append([name,mass_center_dists,atoms_type,ij_pos_vecs,edge_index,prop])
        progress_bar.close()
        print("Atom types: {}".format(atom_set))
        return dataset