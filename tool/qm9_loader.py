from tool.data_loader import DatasetLoader,unit_Ha2meV,unit_u2mu

import os
from tqdm import tqdm
import datetime
import numpy as np
import torch
from torch_cluster import radius_graph
from torch_geometric.datasets import QM9
import pandas as pd

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

source_files = ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

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

    def load_from_csv(self,csv_file_path,use_tqdm=True):

        df = pd.read_csv(csv_file_path)

        start_info = "[{}] Loading prop data from {}".format(datetime.datetime.now(), csv_file_path)
        if use_tqdm:
            iterator = tqdm(range(df.shape[0]),desc=start_info) if use_tqdm else range(df.shape[0])
        else:
            print(start_info)
            iterator = range(df.shape[0])

        id_data = df["mol_id"].values
        mu_data = df["mu"].values.astype(np.float32)
        alpha_data = df["alpha"].values.astype(np.float32)
        homo_data = df["homo"].values.astype(np.float32)
        lumo_data = df["lumo"].values.astype(np.float32)
        gap_data = df["gap"].values.astype(np.float32)
        r2_data = df["r2"].values.astype(np.float32)
        zpve_data = df["zpve"].values.astype(np.float32)
        u0_data = df["u0"].values.astype(np.float32)
        u_data = df["u298"].values.astype(np.float32)
        h_data = df["h298"].values.astype(np.float32)
        g_data = df["g298"].values.astype(np.float32)
        cv_data = df["cv"].values.astype(np.float32)

        prop_list = [{
            "id": id_data[i],
            "prop": {
                "mu": unit_u2mu(mu_data[i]), # dipole moment
                "alpha": unit_u2mu(alpha_data[i]), # isotropic polarizability
                "homo": unit_Ha2meV(homo_data[i]),  # energy of homo
                "lumo": unit_Ha2meV(lumo_data[i]),  # energy of lumo
                "gap": unit_Ha2meV(gap_data[i]),  # gap(lumo-homo)
                "r2": unit_u2mu(r2_data[i]), # electronic spatial extent
                "zpve": unit_Ha2meV(zpve_data[i]), # zero point vibrational energy
                "u0": unit_Ha2meV(u0_data[i]), # internal energy at 0K
                "u": unit_Ha2meV(u_data[i]), # internal energy at 298.15K
                "h": unit_Ha2meV(h_data[i]), # enthalpy at 298.15K
                "g": unit_Ha2meV(g_data[i]), # free energy at 298.15K
                "cv": unit_u2mu(cv_data[i]), # heat capacity at 298.15K
            }} for i in iterator]
        return prop_list

    def load_unsorted_data(self,folder_path,type_list,cutoff=None,atom_mass_dict=None,use_tqdm=True):

        raw_path = "{}/raw".format(folder_path)
        for file in source_files:
            if not self.has_file("{}/{}".format(raw_path, file)):
                self.download(QM9.raw_url,raw_path,extract=True)
                self.download(QM9.raw_url2,raw_path,rename="uncharacterized.txt")

        prop_list = self.load_from_csv("{}/{}".format(raw_path,source_files[1]),use_tqdm=use_tqdm)

        with open("{}/{}".format(raw_path,source_files[2])) as f:
            skip_list = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        sdf_path = "{}/{}".format(raw_path, source_files[0])
        dataset, atom_set = self.load_from_sdf(sdf_path,prop_list, type_list, cutoff, atom_mass_dict, atomrefs=atomrefs, use_tqdm=use_tqdm,skip_list=skip_list)

        print("Atom types: {}".format(atom_set))
        return dataset