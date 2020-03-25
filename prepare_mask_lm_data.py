import os
import random
import json
import collections
import numpy as np
from module.common.tools import save_json
from module.config.base import config
from module.config.bert_config import bert_base_config
from module.common.tools import logger, init_logger
from argparse import ArgumentParser
from module.io.vocabulary import Vocabulary
from module.common.tools import seed_everything

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ['index', 'label'])
init_logger(log_file=config["log_dir"] / "pre_generate_training_data.log")


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """
    按照论文来获取bert训练的数据
    :param tokens: [ [CLS], 343, 132, 5544, 1, 882, 1224.... [SEP]]
    :param masked_lm_prob: 0.15 , Probability of masking each token for the LM
    :param max_predictions_per_seq: 20,Maximum number of tokens to mask in each sequence
    :param vocab_list:词汇表
    :return:
    """
    filter_indices = []
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEPS]"]:
            continue
        filter_indices.append(i)
        if i == 0:
            logger.info("---------------tokens--------------")
            logger.info(" ".join(tokens))

    # 需要掩码的地方有两个限制，　有控制总掩码长度的max_prediction_per_seq,
    # 根据tokens 和每个语句掩码概率的乘积得到可以掩码的长度
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    random.shuffle(filter_indices)
    mask_indexs = sorted(random.sample(filter_indices, num_to_mask))
    masked_token_labels = []
    for idx in mask_indexs:
        if random.ramdom < 0.8:
            masked_token = "[MASK]"
        else:
            if random.random < 0.5:
                masked_token = tokens[idx]
            else:
                masked_token = random.choice(vocab_list)
        # 保存被掩码之前的数据
        masked_token_labels.append(tokens[idx])
        # 进行掩码替换
        tokens[idx] = masked_token
    return tokens, mask_indexs, masked_token_labels


def build_examples(file_path, max_seq_len, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """
    构建bert训练数据,按照论文格式构建
    :param file_path:
    :param max_seq_len:
    :param masked_lm_prob:
    :param max_predictions_per_seq:
    :param vocab_list:
    :return:
    """
    f = open(file_path, 'r')
    lines = f.readlines()
    examples = []
    max_num_tokens = max_seq_len - 2
    for line_cnt, line in enumerate(lines):
        if line_cnt % 5000 == 0:
            logger.info(f"Loading  line {line_cnt}")
        example = {}
        guid = f"corpus-{line_cnt}"
        tokens_a = line.strip("\n").split(" ")[:max_num_tokens]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0 for _ in range(len(tokens))]
        if len(tokens_a) < 5:
            continue
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)

        if line_cnt <= 3:
            print("------------------example--------------------")
            print(f"corpus lines: {guid}")
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("masked_lm_labels: %s" " ".join([str(x) for x in masked_lm_labels]))
            print("segment_ids: %s" % " ".join([str(idx) for idx in segment_ids]))
            print("masked_lm_positions:　%s" % " ".join([str(pos) for pos in masked_lm_positions]))
        example["guid"] = guid
        example["tokens"] = tokens
        example["segment_ids"] = segment_ids
        example["masked_lm_positions"] = masked_lm_positions
        example["masked_lm_labels"] = masked_lm_labels
        examples.append(example)
    return examples


def main():
    parser = ArgumentParser()
    parser.add_argument("--do_data", action="store_action")
    parser.add_argument("--do_corpus", action="store_action")
    parser.add_argument("--do_vocab", action="store_true")
    parser.add_argument("--do_split", action="store_true")
    parser.add_argument("--seed", default=1023, type=int)
    parser.add_argument("--min_freq", default=0, type=int)
    parser.add_argument("--line_per_file", default=100000000, type=int)
    parser.add_argument("--file_num", type=int, default=10, help="Number of dynamic masking to pre_generate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_len", type=float, default=0.1)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of masking a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM")

    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--add_unused", action="store_true")

    args = parser.parser_args()
    seed_everything(args.seed)
    vocab = Vocabulary(min_freq=args.min_freq, add_unused=args.add_unused)
    if args.do_corpus:
        # 将train.txt, test.txt. corpus.txt 中的文本
        corpus = []
        train_path = str(config["data_dir"] / "train.txt")
        with open(train_path, 'r') as fr:
            for ex_id, line in enumerate(fr):
                line = line.strip("\n")
                lines = [" ".join(x.split("/")[0].split("_ ")) for x in line.split("  ")]
                if ex_id == 0:
                    logger.info(f"Train example:{''.join(lines)}")
                corpus.append(' '.join(lines))

        test_path = str(config["data_dir"] / 'test.txt')
        with open(test_path, 'r') as fr:
            for ex_id, line in enumerate(fr):
                line = line.strip('\n')
                lines = line.split("_")

                if ex_id == 0:
                    logger.info(f" Test example:{' '.join(lines)}")
                corpus.append(" ".join(lines))

        corpus_path = str(config["data_dir"] / "corpus.txt")
        with open(corpus_path, 'r') as fr:
            for idx, line in enumerate(fr):
                line = line.strip("\n")
                lines = line.split("_")
                if idx == 0:
                    logger.info(f"Corpus example: {' '.join(lines)}")
                corpus.append(" ".join(lines))
        corpus = list(set(corpus))

        logger.info(f"corpus size: { len(corpus)}")
        # 创建随机id
        random_order = list(range(len(corpus)))
        np.random.shuffle(random_order)
        corpus = [corpus[i] for i in random_order]
        new_corpus_path = config["data_dir"] / "corpus/corpus.txt"
        if not new_corpus_path.exists():
            new_corpus_path.parent.mkdir(exist_ok=True)

        # 随机后保存
        with open(new_corpus_path, 'w') as fr:
            for line in corpus:
                fr.write(line + "\n")
        # 对上面保存的所有语料　按照设定的每个文件包含多少行数的形式进行切分,得到切分之后的数据重新保存
        if args.do_split:

            new_corpus_path = config["data_dir"] / "corpus/corpus.txt"
            split_save_path = config["data_dir"] / "corpus/train"
            if not split_save_path.exists():
                split_save_path.mkdir(exist_ok=True)

            line_per_file = args.line_per_file
            # 每个文件含有 line_per_file 行数据
            command = f"split -a 4 -l {line_per_file} -d {new_corpus_path} {split_save_path} /shard_"
            os.system(command)

        if args.do_vocab:
            # 统计词频
            vocab.read_data(data_path=config["data_dir"] / "corpus/train")
            vocab.build_vocab()
            # 将词汇按照两种形式保存到以下文件中
            vocab.save(file_path=config["data_dir"] / "corpus/vocab_mapping.pkl")
            vocab.save_bert_vocab(file_path=config["checkpoint_dir"] / "vocab.txt")
            logger.info(f"vocab size:{len(vocab)}")
            bert_base_config["vocab_size"] = len(vocab)
            # 重新保存bert的配置文件
            save_json(data=bert_base_config, file＿path=config["checkpoint_dir"] / "config.json")

        if args.do_data:
            vocab_list = vocab.load_bert_vocab(config["checkpoint_dir"] / "vocab.txt")
            data_path = config["data"] / "corpus/train"
            # 获取prepare_fold_data.py 切分的ｋ折数据　　如shard_0000
            files = sorted([f for f in data_path.iterdir() if f.exists() and '.' not in f])

            logger.info("--------pre_generate training data parameters--------")
            logger.info(f"max_seq_len:  {args.max_seq_len}")
            logger.info(f"max_prediction_per_seq: {args.max_prediction_per_seq}")
            logger.info(f"mask_lm_prob: {args.mask_lm_prob}")
            logger.info(f"seed: {args.seed}")
            logger.info(f"file num: {args.file_num}")
            # Number of dynamic masking to pre_generate , 要生成的文件数
            for idx in range(args.file_num):
                logger.info(f"pre_generate file_{idx}.json")
                save_filename = data_path / f"file_{idx}.json"
                # 这里这样重复保存的作用是什么没搞定，每一个保存的file_idx.json文件内容都是一样.
                num_instances = 0
                # 要保存的文件名
                with save_filename.open("w") as fw:
                    # 这里的shard_0000 等文件数量是受之前　 if args.do_split这一段里的代码影响也许只有一个文件
                    for _, file in enumerate(files):
                        file_examples = build_examples(file, max_seq_len=args.max_seq_len,
                                                       masked_lm_prob=args.masked_lm_prob,
                                                       max_predictions_per_seq=args.max_prediction_per_seq,
                                                       vocab_list=vocab_list)
                        file_examples = [json.dumps(instance) for instance in file_examples]
                        for instance in file_examples:
                            fw.write(instance + "\n")
                            num_instances += 1
                metrics_file = data_path / f"file_{idx}_metrics.json"
                print(f"num_instance: {num_instances}")
                with metrics_file.open("w") as metrics_file:
                    metrics = {
                        "num_training_examples": num_instances,
                        "max_seq_len": args.max_seq_len
                    }
                    metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()

