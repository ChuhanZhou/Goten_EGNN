import sys

from configs.config import config as cfg,update_model_cfg
from models.goten_net import GotenNet
from tool.utils import collate_fn,load_atom_mass,get_mean_std

from tqdm import tqdm
import datetime
import torch
import time
from torch.utils.data import random_split, Subset
import numpy as np
import random
import re
import os

def test(model,dataset,title="testing",use_tqdm=True):
    device = cfg['device']
    batch_size = cfg['batch_size']

    model.to(device)
    model.eval()
    if device != 'cpu':
        torch.cuda.empty_cache()

    test_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if use_tqdm:
        time.sleep(0.01)
        progress_bar = tqdm(desc="[{}] [{}]".format(datetime.datetime.now(),title), total=len(test_dataloader))

    total_loss = 0
    out_list = []
    target_list = []
    has_sub_prop = False
    for i, data in enumerate(test_dataloader):
        [atoms_pos, atoms_type, edge_index, prop_value, atoms_batch_index] = data

        out = model(atoms_pos.to(device), atoms_type.to(device), edge_index.to(device), atoms_batch_index.to(device))

        if isinstance(prop_value, list): # for rMD17 and MD22
            has_sub_prop = True
            loss = 0
            for s_i,sub_prop_value in enumerate(prop_value):
                std_sub_prop_value = sub_prop_value.to(device)
                if s_i == 0:
                    std_sub_prop_value = model.standardize(sub_prop_value.to(device))
                sub_loss = cfg["loss_func"](out[s_i], std_sub_prop_value)
                loss+=cfg["loss_weights"][s_i]*sub_loss

                if s_i == 0:
                    sub_out = model.destandardize(out[s_i])
                    sub_target = sub_prop_value
                else:
                    sub_out = out[s_i]
                    sub_target = sub_prop_value
                    #sub_out = torch.norm(out[s_i],dim=-1,keepdim=True)
                    #sub_target = torch.norm(sub_prop_value,dim=-1,keepdim=True)

                if len(out_list) <= s_i:
                    out_list.append([])
                    target_list.append([])
                out_list[s_i].append(sub_out.cpu().detach().numpy())
                target_list[s_i].append(sub_target.cpu().numpy())
        else:
            std_prop_value = model.standardize(prop_value.to(device))
            loss = cfg["loss_func"](out, std_prop_value)

            out = model.destandardize(out)
            out_list.append(out.cpu().detach().numpy())
            target_list.append(prop_value.cpu().numpy())

        total_loss += loss.item() * torch.unique(atoms_batch_index).numel()

        if use_tqdm:
            progress_bar.update()

    if use_tqdm:
        progress_bar.close()
    avg_loss = total_loss / len(dataset)

    if has_sub_prop:
        out_labels = []
        mae = []
        for i in range(len(target_list)):
            sub_outs = np.concatenate(out_list[i], axis=0)
            sub_targets = np.concatenate(target_list[i], axis=0)
            if i==0:
                sub_mae = np.mean(np.abs(sub_targets - sub_outs))
            else:
                sub_mae = np.sum(np.linalg.norm(sub_targets - sub_outs, axis=1)) / len(dataset)
            sub_out_labels = np.concatenate([sub_outs,sub_targets],axis=1)
            out_labels.append(sub_out_labels)
            mae.append(sub_mae)
    else:
        outs = np.concatenate(out_list,axis=0)
        targets = np.concatenate(target_list,axis=0)
        mae = np.mean(np.abs(targets - outs))
        out_labels = np.concatenate([outs,targets],axis=1)

    return avg_loss,mae,out_labels

MODEL_TARGET_LIST = ["alpha","gap","homo","lumo","mu","mu_3d","cv","g","h","r2","u","u0","zpve","e&f"]

if __name__ == '__main__':
    title_label = "qm9_B_t110000_s0"
    test_results = {}
    test_stds = []
    test_logs = []

    train_set = val_set = test_set = dataset = None
    print_4f = False

    for target in MODEL_TARGET_LIST:
        ckpt_path = "./ckpt/{}_{}_best.pth".format(title_label,target)
        if not os.path.isfile(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)

        if dataset is None:
            update_model_cfg(ckpt["dataset"]["version"])
            dataset = cfg["data_loader"].load(cfg["dataset_path"], cfg['atom_types'], cutoff=cfg["cutoff_radius"],atom_mass_dict=load_atom_mass(),preprocess=cfg["preprocess"])
            if cfg["mol_type"] is not None:
                dataset = dataset[cfg["mol_type"]]

            train_set = Subset(dataset, ckpt["dataset"]["train"])
            val_set = Subset(dataset, ckpt["dataset"]["valid"])
            test_set = Subset(dataset, ckpt["dataset"]["test"])

        cfg['predict_label'] = ckpt["label"]

        model = GotenNet()
        model.load_state_dict(ckpt["model_ckpt"], strict=True)

        #val_loss, val_mae, _ = test(model, val_set, title="val_set")
        #time.sleep(0.01)
        #print("[{}] [MAE]: ({}):{:.2f}".format(datetime.datetime.now(), cfg['predict_label'], val_mae))

        test_loss, test_mae, test_out_labels = test(model, test_set, title=target)
        time.sleep(0.01)

        #test_result = "[{}] [test_loss]:{:.3e} [MAE]: ({}):{:.2f}".format(datetime.datetime.now(), test_loss, cfg['predict_label'], test_mae)
        #print(test_result)

        test_results[target] = {"mae":test_mae,"out_label":test_out_labels}
        test_stds.append(np.abs(test_out_labels[:,0]-test_out_labels[:,1]).std())

    if "homo" in test_results and "lumo" in test_results:
        gap_out_label = test_results["lumo"]["out_label"]-test_results["homo"]["out_label"]
        gap_mae = np.mean(np.abs(gap_out_label[:, 1] - gap_out_label[:, 0]), axis=0)
        test_results["gap"] = {"mae":gap_mae,"out_label":gap_out_label}

    dataset_results = []

    for target in MODEL_TARGET_LIST:
        if target not in test_results:
            continue

        if print_4f:
            dataset_results.append("[{}]: {:.4f}".format(target,test_results[target]["mae"]))
        else:
            dataset_results.append("[{}]: {:.2f}".format(target, test_results[target]["mae"]))

    print(" ".join(dataset_results))