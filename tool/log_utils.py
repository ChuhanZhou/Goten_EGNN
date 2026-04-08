from configs.config import config as cfg
from tool.data_loader import ensure_dir

import datetime

LogHistory = []
StrHistory = ""
InitTime = datetime.datetime.now()
LogFileName = "{}_{}_{}_{}_{}_{}".format(
    InitTime.year,
    InitTime.month,
    InitTime.day,
    InitTime.hour,
    InitTime.minute,
    InitTime.second)

def print_log(str_info):
    global StrHistory
    str_info = "[{}] {}".format(datetime.datetime.now(),str_info)
    StrHistory += "{}\n".format(str_info)
    LogHistory.append(str_info)

    print(str_info)

    export_log_history()

def export_log_history():
    ensure_dir(cfg['log_path'])
    with open("{}/{}_{}.txt".format(cfg['log_path'],LogFileName,cfg["predict_label"]), "w", encoding="utf-8") as file:
        file.write(StrHistory)

def load_log(file_name):
    with open("{}/{}_{}.txt".format(cfg['log_path'],file_name,cfg["predict_label"]), 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        global StrHistory
        StrHistory += line
        LogHistory.append(line.strip())