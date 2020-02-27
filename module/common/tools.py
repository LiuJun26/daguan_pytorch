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
