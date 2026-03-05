from configs.config import config as cfg

import datetime
import os
import argparse
import matplotlib.pyplot as plt
import re

LogHistory = []
StrHistory = ""
InitTime = datetime.datetime.now()
LogFileName = "{}_{}_{}_{}_{}_{}".format(
        InitTime.year,
        InitTime.month,
        InitTime.day,
        InitTime.hour,
        InitTime.minute,
        InitTime.second
    )

def print_log(str_info):
    global StrHistory
    StrHistory += "{}\n".format(str_info)
    LogHistory.append(str_info)

    print(str_info)

    export_log_history()

def export_log_history():
    with open("{}/{}.txt".format(cfg['log_path'],LogFileName), "w", encoding="utf-8") as file:
        file.write(StrHistory)

def load_log(file_name):
    with open("{}/{}.txt".format(cfg['log_path'],file_name), 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        global StrHistory
        StrHistory += line
        LogHistory.append(line.strip())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Log analyzer")
    parser.add_argument('--path', help="log file path", type=str, default=None)
    parser.add_argument('--mae', help="show MAE of which label", type=str, default=None)
    parser.add_argument('--target', help="label target value", type=float, default=None)
    args = parser.parse_args()

    #args.path = "../log/2026_3_4_10_24_28.txt"
    #args.mae = "gap"
    #args.target = 21.2
    #args.mae = "homo"
    #args.target = 16.9
    #args.mae = "lumo"
    #args.target = 13.9

    with open(args.path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    train_loss_list = []
    val_loss_list = []
    mae_list = []
    for line in lines:
        train_loss = re.findall(r'(?<=\[avg_loss]:).*?(?=\s)', line)
        if train_loss:
            train_loss_list.append(float(train_loss[0]))

        val_loss = re.findall(r'(?<=\[val_loss]:).*?(?=\s)', line)
        if val_loss:
            val_loss_list.append(float(val_loss[0]))

        label_mae = re.findall(r'(?<=\({}\):).*?(?=\s)'.format(args.mae), line)
        if label_mae:
            mae_list.append(float(label_mae[0]))

    plt.figure(figsize=(12, 8))

    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='train loss')
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='validation loss')

    plt.yscale('log')
    plt.ylim(top=1)

    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Loss Trend')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    plt.figure(figsize=(12, 8))

    if len(mae_list) > 0:
        plt.plot(range(1, len(mae_list) + 1), mae_list, label='MAE of {}'.format(args.mae))

        if args.target is not None:
            plt.plot([len(mae_list), cfg["epochs"]], [mae_list[-1], args.target],linestyle='-.')
            plt.axhline(y=args.target, color='r', linestyle='--', linewidth=1, label='target value ({:.2f})'.format(args.target))

        #plt.xscale('log')
        #plt.xlim(right=1000)

        plt.yscale('log')
        plt.ylim(top=1e3)

        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE Trend')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()

