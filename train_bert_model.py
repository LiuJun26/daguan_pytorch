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
from module.model.pytorch_transformers.file_utils import CONFIG_NAME
from module.model.pytorch_transformers.tokenization_bert import BertTokenizer
from module.model.pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from module.common.tools import seed_everything


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids")
init_logger(log_file=config["log_dir"] / "train_bert_model.log")


def convert_example_to_features(example, tokenizer, max_seq_len):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    # 此处需要复习
    masked_lm_position = example["masked_lm_position"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_len
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    input_array = np.zeros(max_seq_len, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    mask_array = np.zeros(max_seq_len, dtype=np.bool)
    mask_array[:len(input_ids)] = 1
    segment_array = np.zeros(max_seq_len, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids
    lm_label_array = np.full(max_seq_len, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_position] = masked_label_ids
    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, file_id, tokenizer, reduce_meomry=False):
        self.tokenizer = tokenizer
        self.file_id = file_id
        data_file = training_path / f"file_{self.file_id}.json"
        metric_file = training_path / f"file_{self.file_id}_metrics.json"
        assert data_file.is_file() and metric_file.is_file()
        metrics = json.loads(metric_file.read_text())
        num_samples = metrics["num_training_examples"]
        seq_len = metrics["max_seq_len"]
        if reduce_meomry:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir / "input_ids.memmap", mode='w',
                                  dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / "input_masks.mmemap", shape=(num_samples, seq_len),
                                    mode='w+',dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / "segment_ids.memmap", mode='w+',
                                    shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / "lm_label_ids.memmap",
                                     mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            lm_label_ids[:] = -1
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
        logger.info(f"Loading training examples for {str(data_file)}")
        with data_file.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
        assert i == num_samples-1
        logger.info("Loading complete")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)))


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
                        help="Loss scaling to improve fp16 numeric stability, Only used when fp16 set to True."
                             "0( default value) dynamic loss scaling.\n Positive power of 2: static loss scaling value")
    parser.add_argument("--warm_up_proportion", default=0.1, type=float, help="Linear warmup over warmup_steps")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--learning-rate", default=2e-4, type=float, help=" the initial learning rate for Adam")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16_opt_level", type=str, default='O2',
                        help="fp16_opt: Apex AMP optimizer level selected in ['O0','O1','O2','O3']")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()
    pre_generated_data = config["data_dir"] / "corpus/train"
    assert pre_generated_data.is_dir(), "--pre_generated_data should point to folder" \
                                        " of files made by prepare_lm_data_mask.py"
    samples_per_epoch = 0
    # 此处主要是获取训练数据量大小,判断之前prepare_mask_lm_data.py 生成的文件数是否都含有数据
    for i in range(args.file_num):
        data_file = pre_generated_data / f"file_{i}.json"
        metrics_file = pre_generated_data / f"file_{i}_metrics.json"
        if data_file.is_file() and metrics_file.is_file():
            metric = json.loads(metrics_file.read_text())
            samples_per_epoch += metric["num_training_examples"]
        else:
            if i == 0:
                exit("No training data was found")
            print(f"Warning!  There are fewer epochs of  pre_generated data ({i}) than training epochs ({args.epochs})")
            print("This script will loop over the available data, but training diversity may be negative")
            break
    logger.info(f"sample_per_epoch: {samples_per_epoch}")

    # 获取gpu数量
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(f"cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # 分布式训练
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    logger.info(f"device: {device}, distributed training : {bool(args.local_rank != -1)},"
                f" 16 bit training: {args.fp16}")
    if args.gradient_accumulation_steps < 1:
        raise ValueError(f"Invalid gradient_accumulation_steps parameters: { args.gradient_accumulation_steps}")

    # 此处没有理解
    args.train_batch_size = args.train_batch_size // args.gradient_accumulationm_steps
    seed_everything(args.seed)

    tokenizer = BertTokenizer(vocab_file=config["checkpoint_dir"] / "vocab.txt")
    logger.info("--------------------tokenizer[:2]-----------------")
    logger.info(tokenizer[:2])
    total_train_examples = samples_per_epoch * args.epochs
    # 总共需要更新多少次
    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps
    )
    if args.local_rank != -1:
        num_trian_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    # 前面多少步作为预热阶段
    args.warmup_steps = int(num_train_optimization_steps * args.warmup_steps)
    # Prepare model
    with open(str(config["checkpoint_dir"]/ "config.json"), 'r', encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    logger.info("------------------json_config-----------------")
    logger.info(json_config)
    bert_config = BertConfig.from_json_file(str(config["checkpoint_dir"] / "config.json"))

    # 每次训练之前现判断是否有检查点
    checkpoint_models = sorted([f for f in config["checkpoint_dir"].iterdir() if "lm-checkpoint" in f
                                and len(f.iterdir()) > 0])

    if len(checkpoint_models) <= 1:
        model = BertForMaskedLM(bert_config)
    else:

        model = BertForMaskedLM.from_pretrained(config["checkpoint_dir"] / checkpoint_models[-1])
    # 将模型加载指定设备比如GPU
    model.to(device)
    param_optimizer = list(model.named_parameters())
    logger.info("-----------------param_optimizers----------------------")
    logger.info(param_optimizer[:2])
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(i in n for i in no_decay)], 'weight_decay': 0.01},
        {"params": [p for n, p in param_optimizer if any(n in i for i in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_trian_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    global_steps = 0
    metric = LMAccuracy()
    tr_acc = AverageMeter()
    tr_loss = AverageMeter()

    train_logs = {}
    logger.info("----------Running training-----------")
    logger.info(f" Num examples = {total_train_examples}")
    logger.info(f" Batch size = {args.train_batch_size}")
    logger.info(f" Num steps = {num_train_optimization_steps}")
    logger.info(f" warmup steps = {args.warmup_steps}")

    seed_everything(args.seed)
    for epoch in range(args.epochs):
        for idx in range(args.file_num):
            epoch_dataset = PregeneratedDataset(file_idx=idx, training_path=pre_generated_data, tokenizer=tokenizer,
                                                reduce_memory=args.reduce_memory)
            if args.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()
        nb_tr_example, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # 将数据保存到对于设备
            batch = (t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids,
                            attention_mask=input_mask, masked_lm_labels=lm_label_ids)
            predoutput = outputs[1]
            loss = outputs[0]
            metric(logits=predoutput.view(-1, bert_config.vocab_size), target=lm_label_ids)
            if args.n_gpu > 1:
                loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            nb_tr_steps += 1
            tr_acc.update(metric.value(), n=input_ids.size(0))
            tr_loss.update(loss.item(), n=1)
            if (step+1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    # 剃度裁减
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_steps += 1

            if global_steps % args.num_eval_steps == 0:
                train_logs["loss"] = tr_loss.avg
                train_logs["acc"] = tr_acc.avg
                show_info = f"\n[Training]:[{epoch}/{args.epochs}]{global_steps}/{num_train_optimization_steps}" + '-'.join(
                    [f'{key}: {value:.4f}' for key, value in train_logs.items()]
                )
                logger.info(show_info)
                tr_acc.reset()
                tr_loss.reset()
            if global_steps % args.num_save_steps == 0:
                if args.local_rank in [-1, 0] and args.num_save_steps > 0:
                    # 保存模型路径
                    output_dir = config['checkpoint_dir'] / f'lm-checkpoint-{global_steps}'
                    if not output_dir.exists():
                        output_dir.mkdir()
                    #  保存模型
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(str(output_dir))

                    torch.save(args, str(output_dir / 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    output_config_file = output_dir / CONFIG_NAME
                    with open(str(output_config_file), 'w') as fr:
                        fr.write(model_to_save.config.to_json_string())

                    tokenizer.save_vocabulary(output_dir)


if __name__ == '__main__':
    main()


# 针对你说的,我们