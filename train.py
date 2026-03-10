import math
import sys

from configs.config import config as cfg
from models.goten_net import GotenNet
from tool.utils import collate_fn,get_mean_std
from test import test
from tool.log_utils import print_log,LogFileName,load_log

from tqdm import tqdm
import datetime
import torch
import time
from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
from torch.utils.data import random_split, Subset
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
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, generator=g, shuffle=True, collate_fn=collate_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument('--ckpt', help="ckpt path for continue training", type=str, default=None)
    parser.add_argument('--title', help="model title", type=str, default=cfg['title'])
    parser.add_argument('--seed', help="random seed", type=int, default=cfg['seed'])
    parser.add_argument('--label', help="target label", type=str, default=cfg['predict_label'])
    parser.add_argument('--epoch', help="maximum epoch number", type=int, default=cfg['epochs'])
    parser.add_argument('--batch', help="batch size", type=int, default=cfg['batch_size'])
    parser.add_argument('--tqdm', help="print progress bar", type=str, default="True")
    args = parser.parse_args()
    args.tqdm = args.tqdm.lower() == "true"

    if args.ckpt and not os.path.isfile(args.ckpt):
        print_log("[{}] Can't find ckpt at [{}]".format(datetime.datetime.now(),args.ckpt))
        args.ckpt = None

    cfg['title'] = args.title
    cfg['seed'] = args.seed
    cfg['predict_label'] = args.label
    cfg['epochs'] = args.epoch
    cfg['batch_size'] = args.batch

    val_mae_history = []
    if args.ckpt:
        ckpt = torch.load(args.ckpt, weights_only=False)
        cfg['seed'] = ckpt['seed']
        cfg["predict_label"] = ckpt["label"]
        val_mae_history += ckpt["val_history"]
        load_log(ckpt['log'])
    else:
        ckpt = None

    dataset = cfg["data_loader"].load(cfg["dataset_path"],cfg['atom_types'],use_tqdm=args.tqdm)

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
        train_set = Subset(dataset, ckpt["datasets"]["train"])
        val_set = Subset(dataset, ckpt["datasets"]["valid"])
        test_set = Subset(dataset, ckpt["datasets"]["test"])
    else:
        if cfg["train_size"]>=1 and cfg["val_size"]>=1 and cfg["test_size"]>=1:
            data_drop_len = len(dataset) - (cfg["train_size"] + cfg["val_size"] + cfg["test_size"])
        else:
            data_drop_len = 1.0 - (cfg["train_size"] + cfg["val_size"] + cfg["test_size"])

        if data_drop_len!=0:
            train_set, val_set, test_set, _ = random_split(dataset,[cfg["train_size"],cfg["val_size"],cfg["test_size"],data_drop_len])
        else:
            train_set, val_set, test_set = random_split(dataset, [cfg["train_size"], cfg["val_size"], cfg["test_size"]])

    prop_mean_std = get_mean_std([data[-1] for data in train_set])
    mean, std = prop_mean_std[cfg["predict_label"]]

    print_log("[{}] [{}] device: {} | random_seed: {} | train: {} | val: {} | test: {}".format(datetime.datetime.now(), cfg["title"], device, seed, len(train_set), len(val_set), len(test_set)))

    model = GotenNet(mean=mean,std=std)
    if args.ckpt:
        print_log("[{}] Continue training from [{}]".format(datetime.datetime.now(), args.ckpt))
        model.load_state_dict(ckpt["model_ckpt"], strict=False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg['weight_decay'],eps=1e-7)
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg["lr_decay"],
        patience=cfg["lr_patience"],
        min_lr=1e-7)

    current_step = 0
    min_val_mae = math.inf

    #Preparing to continue training
    for epoch in range(len(val_mae_history)):
        if current_step < cfg["warmup"]:
            train_dataloader = get_epoch_dataloader(base_seed=seed, epoch=epoch, dataset=train_set, batch_size=batch_size,collate_fn=collate_fn)
            for i, data in enumerate(train_dataloader):
                current_step += 1
                optimizer.step()
                optimizer.zero_grad()
                if current_step <= cfg["warmup"]:
                    scheduler_warmup.step()
        else:
            val_mae = val_mae_history[epoch]
            scheduler_plateau.step(val_mae)
            if min_val_mae > val_mae:
                min_val_mae = val_mae

    start_epoch = len(val_mae_history)
    for epoch in range(start_epoch,epoch_num):
        model.train()
        train_dataloader = get_epoch_dataloader(base_seed=seed,epoch=epoch,dataset=train_set,batch_size=batch_size,collate_fn=collate_fn)

        if args.tqdm:
            time.sleep(0.01)
            progress_bar = tqdm(desc="[{}] [epoch]:{}".format(datetime.datetime.now(), epoch),total=len(train_dataloader))

        total_loss = 0
        avg_loss = None
        for i, data in enumerate(train_dataloader):
            [mass_center_dists, atoms_type, ij_vecs, edge_index, prop_value, atoms_batch_index] = data

            out = model(atoms_type.to(device), ij_vecs.to(device), edge_index.to(device), atoms_batch_index.to(device))

            std_prop_value = model.standardize(prop_value.to(device))
            loss = cfg["loss_func"](out,std_prop_value)

            if torch.isnan(loss) or torch.isinf(loss):
                ckpt = {
                    "model_ckpt": model.state_dict(),
                    "seed": seed,
                    "label": cfg["predict_label"],
                    "datasets": {
                        "train": train_set.indices,
                        "valid": val_set.indices,
                        "test": test_set.indices,
                    },
                    "log": LogFileName,
                    "val_history":val_mae_history,
                    "out": out.cpu().detach().numpy(),
                    "target": std_prop_value.cpu().numpy(),
                    "nan_data": data,
                }
                torch.save(ckpt, "./ckpt/{}_t{}_s{}_{}_nan.pth".format(cfg['title'], len(train_set), seed, cfg["predict_label"]))
                print_log("Training ended because of nan.")
                sys.exit()

            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (i+1)
            if args.tqdm:
                progress_bar.set_postfix(
                    lr="{:.3e}".format(lr),
                    loss="{:.3e}".format(loss.item()),
                    avg_loss="{:.3e}".format(avg_loss))
                progress_bar.update()

            current_step+=1
            if current_step <= cfg["warmup"]:
                scheduler_warmup.step()

        if args.tqdm:
            progress_bar.close()

        val_loss, val_mae, _ = test(model, val_set,"validating",args.tqdm)
        val_mae_history.append(val_mae)
        if current_step > cfg["warmup"]:
            scheduler_plateau.step(val_mae)

        epoch_result = "[{}] [epoch]:{} [lr]:{:.3e} [avg_loss]:{:.3e} [val_loss]:{:.3e} [MAE]: ({}):{:.2f}".format(datetime.datetime.now(),epoch,lr,avg_loss,val_loss,cfg["predict_label"],val_mae)
        print_log(epoch_result)

        ckpt = {
            "model_ckpt":model.state_dict(),
            "seed": seed,
            "label": cfg["predict_label"],
            "datasets": {
                "train": train_set.indices,
                "valid": val_set.indices,
                "test": test_set.indices,
            },
            "log": LogFileName,
            "val_history":val_mae_history,
        }
        torch.save(ckpt, "./ckpt/{}_t{}_s{}_{}.pth".format(cfg['title'], len(train_set), seed, cfg["predict_label"]))
        if min_val_mae > val_mae:
            min_val_mae = val_mae
            torch.save(ckpt, "./ckpt/{}_t{}_s{}_{}_best.pth".format(cfg['title'], len(train_set), seed, cfg["predict_label"]))

    _, test_mae,_ = test(model, test_set,use_tqdm=args.tqdm)
    test_result = "[{}] [test] [MAE]: ({}):{:.2f}".format(datetime.datetime.now(), cfg["predict_label"], test_mae)
    print_log(test_result)
