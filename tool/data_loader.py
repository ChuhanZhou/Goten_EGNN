import os
from abc import ABC, abstractmethod
import numpy as np

class DatasetLoader(ABC):
    @staticmethod
    def has_file(file_path):
        if file_path == None:
            return False
        return os.path.isfile(file_path)

    def load(self,folder_path,type_list,cutoff=None,atom_mass_dict=None,use_tqdm=True):
        dataset = self.load_unsorted_data(folder_path,type_list,cutoff,atom_mass_dict,use_tqdm)

        # sort dataset to ensure the same sequence on different devices
        dataset.sort(key=lambda x: x[0])

        return dataset

    @abstractmethod
    def load_unsorted_data(self, folder_path, type_list ,cutoff=None, atom_mass_dict=None, use_tqdm=True):
        pass

def unit_Ha2meV(ha):
    return ha * 27211.386245981

def unit_u2mu(a):
    return a*1e3