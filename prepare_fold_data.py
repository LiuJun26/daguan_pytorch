import argparse
from collections import Counter
from module.config.base import config
from module.common.tools import logger
from module.common.tools import init_logger
from module.common.tools import save_pickle
from module.common.tools import load_pickle
from sklearn.model_selection import StratifiedKFold


def data_aug1(data):
    new_data = []
    i = 0
    for line in data:
        tags = [x.split("-")[1] for x in line['tag'].split(" ") if "-" in x]
        tags = list(set(tags))
        if ('b' in tags or 'a' in tags) and 'c' in tags:
            c_ = []
            t_ = []
            context = line['context'].split(" ")
            raw_tags = line['tag'].split(" ")
            for c, t in zip(context, raw_tags):
                if 'c' in t:
                    continue
                c_.append(c)
                t_.append(t)
            if i <= 5:
                logger.info("--------- data aug1 -----------")
                logger.info(f"raw: {line['context']}")
                logger.info(f'new: {" ".join(c_)}')
                logger.info(f"raw_tag: {line['tag']}")
                logger.info(f'tag: {" ".join(t_)}')
                i += 1
            new_data.append({"context": " ".join(c_),
                             "tag": " ".join(t_),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
        else:
            continue
    logger.info(f"data aug size: {len(new_data)}")
    return new_data


def data_aug2(data):
    new_data = []
    i = 0
    for line in data:
        tags = [x.split("-")[1] for x in line['tag'].split(" ") if "-" in x]
        tags = list(set(tags))
        if 'b' in tags and 'a' in tags:
            c_ = []
            t_ = []
            context = line['context'].split(" ")
            raw_tags = line['tag'].split(" ")
            for c, t in zip(context, raw_tags):
                if 'c' in t or 'b' in t:
                    continue
                c_.append(c)
                t_.append(t)
            if i <= 2:
                logger.info("--------- data aug2 -----------")
                logger.info(f"raw: {line['context']}")
                logger.info(f'new: {" ".join(c_)}')
                logger.info(f"raw_tag: {line['tag']}")
                logger.info(f'tag: {" ".join(t_)}')
                i += 1
            new_data.append({"context": " ".join(c_),
                             "tag": " ".join(t_),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
        else:
            continue
    logger.info(f"data2 aug size: {len(new_data)}")
    return new_data


def data_aug3(data):
    new_data = []
    i = 0
    for line in data:
        tags = [x.split("-")[1] for x in line['tag'].split(" ") if "-" in x]
        tags = list(set(tags))
        if 'b' in tags and 'a' in tags and 'c' in tags:
            c_1 = []
            t_1 = []
            c_2 = []
            t_2 = []
            context = line['context'].split(" ")
            raw_tags = line['tag'].split(" ")
            for c, t in zip(context, raw_tags):
                if 'a' in t :
                    continue
                c_1.append(c)
                t_1.append(t)

            for c, t in zip(context, raw_tags):
                if 'b' in t:
                    continue
                c_2.append(c)
                t_2.append(t)
            if i <= 2:
                logger.info("--------- data aug3 -----------")
                logger.info(f"raw: {line['context']}")
                logger.info(f'new: {" ".join(c_1)}')
                logger.info(f"raw_tag: {line['tag']}")
                logger.info(f'tag: {" ".join(t_1)}')
                i += 1
            new_data.append({"context": " ".join(c_1),
                             "tag": " ".join(t_1),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
            new_data.append({"context": " ".join(c_2),
                             "tag": " ".join(t_2),
                             'id': line['id'],
                             'raw_context': line['raw_context']})
        else:
            continue
    logger.info(f"data3 aug size: {len(new_data)}")
    return new_data


def make_folds(args):
    # 3028_7118_19302 / a   17506_20312_13333_15274 / o
    seed = args.seed
    folds = args.folds
    data_aug = args.data_aug
    data_name = args.data_name
    train_data = []
    train_data_path = config["data_dir"] / 'train.txt'
    with open(train_data_path, 'r') as file:
        idx = 0
        for line in file:
            json_obj = {}
            context = []
            tags = []
            line = line.strip("\n")
            lines = line.split("  ")
            for seg in lines:
                segs = seg.split("/")
                seg_text = segs[0].split("_")
                seg_label = segs[1]
                context.extend(seg_text)
                if seg_label == 'o':
                    tags.extend(['O'] * len(seg_text))
                elif len(seg_text) == 1:
                    tags.extend([f'S-{seg_label}'])
                else:
                    tags.extend(f"B-{seg_label}")
                    tags.extend(f"I-{seg_label}" * (len(seg_text) - 1))
            json_obj["id"] = idx
            json_obj["context"] = " ".join(context)
            json_obj["tag"] = " ".join(tags)
            json_obj["raw_context"] = line
            label = [tag.split('-')[1] for tag in tags if '-' in tag]
            label_set = list(set(label))
            if len(label_set) == 1:
                y = 0
            elif len(label_set) == 3:
                y = 4
            elif len(label_set) == 2:
                if 'a' in label_set and 'b' in label_set:
                    y = 1
                elif 'a' in label_set and 'c' in label_set:
                    y = 2
                else:
                    y = 3
            else:
                raise ValueError("tag number is error.....please check your code ")
            json_obj['y'] = y
            idx += 1
            train_data.append(json_obj)

        # 以上是对原始数据的标注进行剥离,重新标注,使其符合标准标注形式
        y_counter = Counter()
        y_counter.update([d["y"] for d in train_data])
        a_t = train_data
        b_t = [d["y"] for d in train_data]
        SKF = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        # SKF.split(A_T,B_T)
        for idx, (train_index, test_index) in enumerate(SKF.split(a_t, b_t)):
            logger.info(f"fold {idx} info:")
            logger.info(f"raw train data size {len(train_data)}")
            logger.info(f"raw test data size {len(test_index)}")
            X_train = [a_t[i] for i in train_data]
            if data_aug:
                # 对训练数据进行增强(数据增强)
                aug1 = data_aug1(X_train)
                aug2 = data_aug2(X_train)
                aug3 = data_aug3(X_train)
                pass
            X_test = [a_t[i] for i in test_index]
            train_file_name = f"{data_name}_train_{idx}.pkl"
            valid_file_name = f"{data_name}_valid_{idx}.pkl"
            save_pickle(X_train, config["data_dir"] / train_file_name)
            save_pickle(X_test, config["data_dir"] / valid_file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=44, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--do_aug", action='store_true')
    parser.add_argument("--data_name", default='datagrand', type=str)
    args = parser.parse_args()
    init_logger(log_file=config["log_dir"] / "prepare_fold_data.log")
    make_folds(args)


if __name__ == '__main__':
    main()

