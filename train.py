import math
import sys

from configs.config import config as cfg,update_model_cfg
from models.goten_net import GotenNet
from models.decoder import get_decoder
from tool.utils import collate_fn,get_mean_std,load_atom_mass
from test import test
from tool.log_utils import print_log,LogFileName,load_log

from tqdm import tqdm
import datetime
import torch
import time
from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
from torch.utils.data import random_split, Subset
from torch.amp import autocast, GradScaler
import numpy as np
import random
import argparse
import os

torch.backends.cuda.matmul.allow_tf32 = False

def warmup_lambda(current_step):
    if current_step < cfg["warmup"]:
        return float(current_step) / float(max(1, cfg["warmup"]))
    return 1.0

def get_epoch_dataloader(base_seed , epoch, dataset, batch_size, collate_fn):
    g = torch.Generator()
    g.manual_seed(base_seed + epoch)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, generator=g, shuffle=True, collate_fn=collate_fn,drop_last=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument('--ckpt', help="ckpt path for continue training", type=str, default=None)
    parser.add_argument('--title', help="model title", type=str, default=None)
    parser.add_argument('--seed', help="random seed", type=int, default=None)
    parser.add_argument('--label', help="target label", type=str, default=None)
    parser.add_argument('--mol', help="target molecule", type=str, default=None)
    parser.add_argument('--epoch', help="maximum epoch number", type=int, default=None)
    parser.add_argument('--batch', help="batch size", type=int, default=None)
    parser.add_argument('--tqdm', help="print progress bar", type=str, default="True")
    parser.add_argument('--ckpt_def', help="continue on default ckpt", type=str, default="False")
    parser.add_argument('--ver', help="model type", type=str, default="qm9_s")
    parser.add_argument('--decoder', help="decoder type", type=str, default=None)
    args = parser.parse_args()
    args.tqdm = args.tqdm.lower() == "true"
    args.ckpt_def = args.ckpt_def.lower() == "true"

    update_model_cfg(args.ver)
    if args.title is not None:
        cfg['title'] = args.title
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.epoch is not None:
        cfg['epochs'] = args.epoch
    if args.batch is not None:
        cfg['batch_size'] = args.batch
    if args.label is not None:
        cfg['predict_label'] = args.label
    if args.mol is not None:
        cfg['mol_type'] = args.mol

    if args.ckpt is None and args.ckpt_def:
        if cfg['mol_type'] is None:
            args.ckpt = "./ckpt/{}_t{}_s{}_{}.pth".format(cfg['title'], cfg["train_size"], cfg['seed'], cfg["predict_label"])
        else:
            args.ckpt = "./ckpt/{}_t{}_s{}_{}.pth".format(cfg['title'], cfg["train_size"], cfg['seed'], cfg["mol_type"])

    if args.ckpt and not os.path.isfile(args.ckpt):
        print_log("Can't find ckpt at [{}]".format(args.ckpt))
        args.ckpt = None

    val_mae_history = []
    if args.ckpt:
        ckpt = torch.load(args.ckpt, weights_only=False)
        update_model_cfg(ckpt["dataset"]["version"])
        cfg['seed'] = ckpt['seed']
        cfg["predict_label"] = ckpt["label"]
        val_mae_history += ckpt["val_history"]
        load_log(ckpt['log'])
    else:
        ckpt = None

    dataset = cfg["data_loader"].load(cfg["dataset_path"],cfg['atom_types'],cutoff=cfg["cutoff_radius"],atom_mass_dict=load_atom_mass(),use_tqdm=args.tqdm,preprocess=cfg["preprocess"],key=cfg["mol_type"])
    if cfg["mol_type"] is not None:
        dataset = dataset[cfg["mol_type"]]

    device = cfg['device']
    epoch_num = cfg['epochs']
    batch_size = cfg['batch_size']
    lr = cfg['lr_max']
    seed = cfg['seed']

    if seed is None:
        seed = random.randint(1, 10000)
        cfg['seed'] = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.ckpt:
        train_set = Subset(dataset, ckpt["dataset"]["train"])
        val_set = Subset(dataset, ckpt["dataset"]["valid"])
        test_set = Subset(dataset, ckpt["dataset"]["test"])
    else:
        if cfg["train_size"]>=1 and cfg["val_size"]>=1 and cfg["test_size"]>=1:
            data_drop_len = len(dataset) - (cfg["train_size"] + cfg["val_size"] + cfg["test_size"])
        elif -1 in [cfg["train_size"],cfg["val_size"],cfg["test_size"]]:
            data_drop_len=0
            split_keys = ["train_size","val_size","test_size"]
            other_num = len(dataset)
            other_k = None
            for k in split_keys:
                if cfg[k]>=0:
                    other_num -= cfg[k]
                else:
                    other_k = k
            cfg[other_k] = other_num
        else:
            data_drop_len = 1.0 - (cfg["train_size"] + cfg["val_size"] + cfg["test_size"])

        train_set, val_set, test_set = cfg["data_loader"].split_data(dataset,[cfg["train_size"],cfg["val_size"],cfg["test_size"],data_drop_len],seed,cfg["dataset_path"],cfg["split_key"])

    prop_mean_std = get_mean_std([data[-1] for data in train_set],[cfg["predict_label"]])
    mean, std = prop_mean_std[cfg["predict_label"]]

    print_log("[{}({})] device: {} | random_seed: {} | total: {} | train: {} | val: {} | test: {}".format(cfg["title"],cfg["model_type"], device, seed, len(dataset), len(train_set), len(val_set), len(test_set)))

    model = GotenNet(mean=mean, std=std)
    if args.decoder is not None:
        model.decoder = get_decoder(args.decoder,mean,std)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'], eps=1e-7)
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg["lr_decay"],
        patience=cfg["lr_patience"],
        threshold=1e-6,
        threshold_mode='rel',
        cooldown=2,
        min_lr=1e-7)

    if args.ckpt:
        print_log("Continue training from [{}]".format(args.ckpt))
        model.load_state_dict(ckpt["model_ckpt"], strict=True)
        optimizer.param_groups[0]['lr'] = ckpt["lr"]
        scheduler_warmup.load_state_dict(ckpt["scheduler_warmup"])
        scheduler_plateau.load_state_dict(ckpt["scheduler_plateau"])

    min_val_mae = math.inf
    not_best_epoch = 0

    #Preparing to continue training
    for epoch in range(len(val_mae_history)):
        if scheduler_warmup.last_epoch < cfg["warmup"]:
            break
        else:
            val_mae = val_mae_history[epoch]
            if min_val_mae > val_mae:
                min_val_mae = val_mae
                not_best_epoch = 0
            else:
                not_best_epoch += 1

    sub_label = cfg["predict_label"] if  cfg["mol_type"] is None else cfg["mol_type"]
    ckpt_path = "./ckpt/{}_t{}_s{}_{}.pth".format(cfg['title'], len(train_set), seed, sub_label)
    best_ckpt_path = "./ckpt/{}_t{}_s{}_{}_best.pth".format(cfg['title'], len(train_set), seed, sub_label)

    start_epoch = len(val_mae_history)
    has_sub_prop = False
    for epoch in range(start_epoch, epoch_num):
        # early stop
        if cfg["stop_patience"] is not None and not_best_epoch >= cfg["stop_patience"]:
            break

        model.train()
        train_dataloader = get_epoch_dataloader(base_seed=seed,epoch=epoch,dataset=train_set,batch_size=batch_size,collate_fn=collate_fn)

        if args.tqdm:
            time.sleep(0.01)
            progress_bar = tqdm(desc="[{}] [epoch]:{}".format(datetime.datetime.now(), epoch),total=len(train_dataloader))

        total_loss = []
        avg_loss = None
        for i, data in enumerate(train_dataloader):
            [atoms_pos, atoms_type, edge_index, prop_value, atoms_batch_index] = data

            out = model(atoms_pos.to(device), atoms_type.to(device), edge_index.to(device), atoms_batch_index.to(device))

            if isinstance(prop_value, list): # for rMD17 and MD22
                has_sub_prop = True
                loss = 0
                for s_i,sub_prop_value in enumerate(prop_value):
                    sub_prop_value = sub_prop_value.to(device)
                    if s_i == 0:
                        sub_prop_value = model.standardize(sub_prop_value)
                    else:
                        sub_prop_value = sub_prop_value/model.std
                    sub_loss = cfg["loss_func"](out[s_i], sub_prop_value)
                    loss+=cfg["loss_weights"][s_i]*sub_loss
            else: # for QM9 and molecule3d
                std_prop_value = model.standardize(prop_value.to(device))
                loss = cfg["loss_func"](out,std_prop_value)

            lr = optimizer.param_groups[0]['lr']
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                total_loss.append(loss.item())

                if scheduler_warmup.last_epoch < cfg["warmup"]:
                    scheduler_warmup.step()
            optimizer.zero_grad()

            avg_loss = np.mean(total_loss) if len(total_loss) > 0 else 0
            if args.tqdm:
                progress_bar.set_postfix(
                    lr="{:.3e}".format(lr),
                    loss="{:.3e}".format(loss.item()),
                    avg_loss="{:.3e}".format(avg_loss))
                progress_bar.update()

        if args.tqdm:
            progress_bar.close()

        val_loss, val_mae, _ = test(model, val_set,"val_set",args.tqdm)

        step_mae=0
        if has_sub_prop:
            for i, sub_mae in enumerate(val_mae):
                step_mae += cfg["loss_weights"][i]*sub_mae
        else:
            step_mae += val_mae
        val_mae_history.append(step_mae)
        if scheduler_warmup.last_epoch >= cfg["warmup"]:
            scheduler_plateau.step(step_mae)

        epoch_result = "[epoch]:{} [lr]:{:.3e} [avg_loss]:{:.3e} [val_loss]:{:.3e} [MAE]: ({}):".format(epoch, lr, avg_loss, val_loss, cfg["predict_label"])
        val_results = []
        if has_sub_prop:
            for sub_mae in val_mae:
                val_results.append("{:.4f}".format(sub_mae))
        else:
            val_results.append("{:.2f}".format(val_mae))
        epoch_result += " ".join(val_results)

        if cfg["test_in_train"]:
            _, test_mae, _ = test(model, test_set, "test_set", args.tqdm)
            test_results = []
            if has_sub_prop:
                for sub_mae in test_mae:
                    test_results.append("{:.4f}".format(sub_mae))
            else:
                test_results.append("{:.2f}".format(test_mae))
            epoch_result += " [{}]".format(" ".join(test_results))

        print_log(epoch_result)

        ckpt = {
            "model_ckpt":model.state_dict(),
            "seed": seed,
            "label": cfg["predict_label"],
            "dataset": {
                "version": cfg["model_type"],
                "train": train_set.indices,
                "valid": val_set.indices,
                "test": test_set.indices,
            },
            "log": LogFileName,
            "val_history":val_mae_history,
            "lr":optimizer.param_groups[0]['lr'],
            "scheduler_warmup":scheduler_warmup.state_dict(),
            "scheduler_plateau":scheduler_plateau.state_dict(),
        }

        torch.save(ckpt, ckpt_path)

        if min_val_mae > step_mae:
            min_val_mae = step_mae
            torch.save(ckpt, best_ckpt_path)
            not_best_epoch = 0
        else:
            not_best_epoch += 1

    best_ckpt = torch.load(best_ckpt_path, weights_only=False)["model_ckpt"]
    model.load_state_dict(best_ckpt, strict=True)
    _, test_mae,_ = test(model, test_set,use_tqdm=args.tqdm)
    test_results = []
    if has_sub_prop:
        for sub_mae in test_mae:
            test_results.append("{:.4f}".format(sub_mae))
    else:
        test_results.append("{:.2f}".format(test_mae))
    train_result = "[test_best_validation] [MAE]: {}".format(" ".join(test_results))
    print_log(train_result)
