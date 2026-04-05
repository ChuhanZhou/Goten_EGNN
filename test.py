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
    with torch.inference_mode():
        for i, data in enumerate(test_dataloader):
            [atoms_pos, atoms_type, ij_vecs, edge_index, prop_value, atoms_batch_index] = data

            out = model(atoms_pos.to(device), atoms_type.to(device), ij_vecs.to(device), edge_index.to(device), atoms_batch_index.to(device))

            #atom_norm = scatter_sum(torch.ones([len(atoms_batch_index), 1], device=out.device),atoms_batch_index.to(device), dim=0)
            std_prop_value = model.standardize(prop_value.to(device))
            loss = cfg["loss_func"](out, std_prop_value)

            total_loss += loss.item() * out.shape[0]

            out = model.destandardize(out)
            out_list.append(out.cpu().detach().numpy())
            target_list.append(prop_value.cpu().numpy())

            if use_tqdm:
                progress_bar.update()

    if use_tqdm:
        progress_bar.close()
    avg_loss = total_loss / len(dataset)

    outs = np.concatenate(out_list,axis=0)
    targets = np.concatenate(target_list,axis=0)
    avg_mae = np.mean(np.abs(targets - outs))

    out_labels = np.concatenate([outs,targets],axis=1)
    return avg_loss,avg_mae,out_labels

MODEL_TARGET_LIST = ["alpha","gap","homo","lumo","mu","mu_3d","cv","g","h","r2","u","u0","zpve","f","e"]

if __name__ == '__main__':
    title_label = "qm9_B_t110000_s1"
    test_results = {}

    train_set = val_set = test_set = dataset = None
    print_4f = False

    for target in MODEL_TARGET_LIST:
        ckpt_path = "./ckpt/{}_{}_best.pth".format(title_label,target)
        if not os.path.isfile(ckpt_path):
            continue

        ckpt = torch.load(ckpt_path, weights_only=False)
        cfg['predict_label'] = ckpt["label"]

        if dataset is None:
            update_model_cfg(ckpt["dataset"]["version"])
            dataset = cfg["data_loader"].load(cfg["dataset_path"], cfg['atom_types'], cutoff=cfg["cutoff_radius"],atom_mass_dict=load_atom_mass())
            train_set = Subset(dataset, ckpt["dataset"]["train"])
            val_set = Subset(dataset, ckpt["dataset"]["valid"])
            test_set = Subset(dataset, ckpt["dataset"]["test"])

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