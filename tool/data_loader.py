import os
from abc import ABC, abstractmethod
import torch

class DatasetLoader(ABC):
    def has_file(self,file_path):
        if file_path == None:
            return False
        return os.path.isfile(file_path)

    @abstractmethod
    def load(self,folder_path,type_list):
        pass

def collate_fn(batch):
    batch_atoms_types = torch.zeros((0,1),dtype=torch.int64)
    batch_ij_vecs = torch.zeros((0,3))

    batch_prop = None
    prop_ids = None
    batch_edge_index = torch.zeros((2,0),dtype=torch.int64)
    atoms_batch_index = []
    for i, (id, atoms_types, ij_pos_vecs, edge_index, prop) in enumerate(batch):
        edge_index = edge_index + batch_atoms_types.shape[0]

        batch_atoms_types = torch.cat((batch_atoms_types,atoms_types), dim=0)
        batch_ij_vecs = torch.cat((batch_ij_vecs,ij_pos_vecs), dim=0)
        if batch_prop is None and prop_ids is None:
            prop_ids = list(prop.keys())
            batch_prop = []
        batch_prop.append(list(prop.values()))
        batch_edge_index = torch.cat((batch_edge_index,edge_index), dim=1)
        atoms_batch_index.append(torch.ones(len(atoms_types),dtype=torch.int64)*i)
    batch_prop = torch.tensor(batch_prop)
    atoms_batch_index = torch.cat(atoms_batch_index)
    return batch_atoms_types, batch_ij_vecs, batch_edge_index, prop_ids, batch_prop, atoms_batch_index

