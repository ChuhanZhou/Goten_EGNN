from configs.config import config as cfg
from tool.data_loader import ensure_dir,has_file

import datetime

LogHistory = []
NewLogStr = ""
InitTime = datetime.datetime.now()
LogFileName = "{}_{}_{}_{}_{}_{}".format(
    InitTime.year,
    InitTime.month,
    InitTime.day,
    InitTime.hour,
    InitTime.minute,
    InitTime.second)

def print_log(str_info):
    global NewLogStr
    str_info = "[{}] {}".format(datetime.datetime.now(),str_info)
    NewLogStr += "{}\n".format(str_info)
    LogHistory.append(str_info)

    print(str_info)

    export_new_log()

def export_new_log():
    ensure_dir(cfg['log_path'])
    sub_label = cfg["predict_label"] if cfg["mol_type"] is None else cfg["mol_type"]
    with open("{}/{}_{}.txt".format(cfg['log_path'],LogFileName,sub_label), "a", encoding="utf-8") as file:
        global NewLogStr
        file.write(NewLogStr)
        NewLogStr = ""

#def export_log_history():
#    ensure_dir(cfg['log_path'])
#    sub_label = cfg["predict_label"] if cfg["mol_type"] is None else cfg["mol_type"]
#    with open("{}/{}_{}.txt".format(cfg['log_path'],LogFileName,sub_label), "w", encoding="utf-8") as file:
#        file.write(StrHistory)

def load_log(file_name):
    sub_label = cfg["predict_label"] if cfg["mol_type"] is None else cfg["mol_type"]

    log_file = "{}/{}_{}.txt".format(cfg['log_path'],file_name,sub_label)
    if has_file(log_file):
        with open(log_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            global NewLogStr
            NewLogStr += line
            LogHistory.append(line.strip())