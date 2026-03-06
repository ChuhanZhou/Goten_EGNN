import os
from abc import ABC, abstractmethod
import numpy as np

class DatasetLoader(ABC):
    def has_file(self,file_path):
        if file_path == None:
            return False
        return os.path.isfile(file_path)

    @abstractmethod
    def load(self,folder_path,type_list,use_tqdm=True):
        pass

def unit_Ha2meV(ha):
    return ha * 27211.386245981

def unit_u2mu(a):
    return a*1e3