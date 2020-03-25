import os
import random
import torch
import numpy as np
import json
import pickle
import torch.nn as nn
import  logging
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("Maverick_Ner_01")


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    """
    设置输出日志和文本日志
    :param log_file:
    :param log_file_level:
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s: %(filename)s: %(lineno)d: %(message)s'))
    logger.handlers.clear()
    logger.addHandler(stream_handler)

    if isinstance(log_file, Path):
        log_file = str(log_file)
    if log_file and log_file != "":
        file_handler = logging.FileHandler()
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(logging.Formatter('%(levelname)s: %(asctime)s: %(filename)s: %(lineno)d: %(message)s'))
        logger.addHandler(file_handler)


def save_pickle(data, file_path):
    """
    保存成pickle文件
    :param data:
    :param file_path:
    :return:
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path) as file:
        pickle.dump(data, file)


def load_pickle(input_file):
    """
    加载读取pickle文件
    :param input_file:
    :return:
    """
    with open(str(input_file), 'rb') as file:
        data = pickle.load(file)
        return data


def save_json(data, file＿path):

    """
    将数据保存到json文件中　
    :param data:
    :param file＿path:
    :return:
    """
    if isinstance(file＿path, Path):
        file＿path = str(file＿path)
    with open(file＿path, 'w') as file:
        json.dump(data, file)


def load_json(file_path):
    """
    加载json文本中数据
    :param file_path:
    :return:
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def seed_everything(seed=9527):
    """
    设置整个开发护环境的随机数种子
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ["seed"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


class AverageMeter(object):

    def __len__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




