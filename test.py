from configs.config import config as cfg
from models.goten_net import GotenNet
from tool.utils import collate_fn,unit_Ha2meV,get_std_mat

from tqdm import tqdm
import datetime
import torch
import time
from torch.utils.data import random_split
import numpy as np
import random
import re

def test(model,dataset,title="testing"):
    device = cfg['device']
    batch_size = cfg['batch_size']

    model.to(device)
    model.eval()
    if device != 'cpu':
        torch.cuda.empty_cache()

    test_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    time.sleep(0.01)
    progress_bar = tqdm(desc="[{}] [{}]".format(datetime.datetime.now(),title), total=len(test_dataloader))

    total_loss = 0
    out_list = []
    target_list = []
    with torch.inference_mode():
        for i, data in enumerate(test_dataloader):
            [mass_center_dists, atoms_type, ij_vecs, edge_index, prop_value, atoms_batch_index] = data

            out = model(atoms_type.to(device), ij_vecs.to(device), edge_index.to(device), atoms_batch_index.to(device))

            std_prop_value = model.standardize(prop_value.to(device))
            loss = cfg["loss_func"](out, std_prop_value)

            total_loss += loss.item() * out.shape[0]

            out = model.destandardize(out)
            out_list.append(out.cpu().detach().numpy())
            target_list.append(prop_value.cpu().numpy())

            progress_bar.update()

    progress_bar.close()
    avg_loss = total_loss / len(dataset)

    outs = np.concatenate(out_list,axis=0)
    targets = np.concatenate(target_list,axis=0)
    avg_mae = np.mean(np.abs(targets - outs))

    out_labels = np.concatenate([outs,targets],axis=1)
    return avg_loss,avg_mae,out_labels

if __name__ == '__main__':
    ckpt_path = "./ckpt/QM9_S_d107109_e1000_s1.pth"
    ckpt = torch.load(ckpt_path, weights_only=False)
    seed = ckpt["seed"]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = cfg["data_loader"].load(cfg["dataset_path"],cfg['atom_types'])

    if cfg["train_size"] >= 1 and cfg["val_size"] >= 1 and cfg["test_size"] >= 1:
        data_drop_len = len(dataset) - (cfg["train_size"] + cfg["val_size"] + cfg["test_size"])
    else:
        data_drop_len = 1.0 - (cfg["train_size"] + cfg["val_size"] + cfg["test_size"])

    if data_drop_len != 0:
        _, val_set, test_set, _ = random_split(dataset, [cfg["train_size"], cfg["val_size"], cfg["test_size"],data_drop_len])
    else:
        _, val_set, test_set = random_split(dataset, [cfg["train_size"], cfg["val_size"], cfg["test_size"]])

    prop_mean_std = ckpt["mean_std"]
    model = GotenNet()
    model.load_state_dict(ckpt["model_ckpt"], strict=False)

    val_loss, val_mae, _ = test(model, val_set, title="validating")
    print("[{}] [MAE]: ({}):{:.2f}".format(datetime.datetime.now(),cfg['predict_label'], val_mae))

    test_loss, test_mae, _ = test(model, test_set)

    test_result = "[{}] [test_loss]:{:.3e} [MAE]: ({}):{:.2f}".format(datetime.datetime.now(), test_loss, cfg['predict_label'], test_mae)
    print(test_result)