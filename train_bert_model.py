import torch
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
from tempfile import TemporaryDirectory
from module.common.tools import logger, init_logger
from module.config.base import config
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from module.common.tools import AverageMeter
from module.train.metrics import LMAccuracy
from module.model.pytorch_transformers.modeling_bert import BertForMaskedLM, BertConfig
from module.model.pytorch_transformers.tokenization_bert import BertTokenizer
from module.model.pytorch_transformers.optimization import AdanW, WarmuipLinearSchedule
from module.common.tools import seed_everything


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids")
init_logger(log_file=config["log_dir"] / "train_bert_model.log")


def main():
    parser = ArgumentParser()
    parser.add_argument("--file_num", type=int, default=10, help="Number of pre_generator")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on=disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train for")
    parser.add_argument("--num_eval_steps", default=200)
    parser.add_argument("--num_save_steps", default=5000)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no-cuda", action="store_true", help="Whether not to use cuda when available")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=-1,
                        help="Number of update steps to accumulate before performing a backward/update pass")

    parser.add_argument("--train_batch_size", default=18, type=int, help="Total batch size for training")
    parser.add_argument("--loss_scale", type=float, default=0,
                        help="Loss scaling")


if __name__ == '__main__':
    main()
