from collections import Counter
from collections import OrderedDict
from ..common.tools import save_pickle
from ..common.tools import load_pickle


class Vocabulary(object):

    def __init__(self, max_size=None,
                 min_freq=None,
                 pad_token="[PAD]",
                 unk_token="[UNK]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 add_unused=False):

        self.max_size = max_size
        self.min_freq = min_freq
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = None
        self.rebuild = True
        self.add_unused = add_unused
        self.word_counter = Counter()

    def reset(self):
        """
        将集合word2idx重置,
        :return:
        """
        ctrl_symbols = [self.pad_token, self.unk_token, self.cls_token, self.mask_token, self.sep_token]
        for index, syb in enumerate(ctrl_symbols):
            self.word2idx[syb] = index
        if self.add_unused:
            for i in range(20):
                self.word2idx[f"UNUSED{i}"] = len(self.word2idx)

    def update(self, word_list):
        """
        每次调用都会更新词频.
        :param word_list:
        :return:
        """
        self.word_counter.update(word_list)

    def read_data(self, data_path):
        """
        读取数据,用来统计目录下所有文件中的数据词频
        :param data_path:
        :return:
        """
        if data_path.is_dir():
            files = sorted([f for f in data_path.iterdir() if f.exists()])
        else:
            files = [data_path]
        for file in files:
            f = open(file, 'r')
            # 读取数据
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n")
                words = line.split(" ")
                self.update(words)

    def build_reverse_vocab(self):
        """
        通过self.word2idx生成　self.idx2word
        :return:
        """
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}

    def build_vocab(self):
        """
        获取词汇量,主要从所有训练文本中获取满足词频限制条件的词，将他们构建成一个词汇文件用来保存
        self.max_size: 词频最大的前max_size个数据
        self.min_freq: 词频下限
        :return:
        """
        max_size = min(self.max_size, len(self.word_counter)) if self.max_size else None
        most_common_words = self.word_counter.most_common(max_size)
        if self.min_freq is not None:
            most_common_words = filter(lambda x: x[1] >= self.min_freq, most_common_words)
        if self.word2idx:
            most_common_words = filter(lambda x: x[0] not in self.word2idx, most_common_words)
        start_idx = len(self.word2idx)
        self.word2idx.update({w: i+start_idx for i, (w, _) in enumerate(most_common_words)})
        self.build_reverse_vocab()
        self.rebuild = False

    def clear(self):
        """
        重置恢复初始状态
        :return:
        """
        self.word_counter.clear()
        self.word2idx = None
        self.idx2word = None
        self.rebuild = True
        self.reset()

    def save(self, file_path):
        """
        将获取到的词汇保存 save vocab
        :param file_path:
        :return:
        """
        mapping = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word
        }
        save_pickle(mapping, file_path)

    def load_file(self, file_path):
        """
        加载文件
        :param file_path:
        :return:
        """
        mapping = load_pickle(file_path)

        self.word2idx = mapping["word2idx"]
        self.idx2word = mapping["idx2word"]

    def save_bert_vocab(self, file_path):
        """
        保存vocab成bert模式
        :param file_path:
        :return:
        """
        bert_vocab = [x for x, y in self.word2idx.items()]
        with open(file_path, 'w') as fr:
            for token in bert_vocab:
                fr.write(token + '\n')

    def load_bert_vocab(self, vocab_file):
        """
        读取bert文件中的数据用list返回
        :param vocab_file:
        :return:
        """
        vocab = OrderedDict()
        idx = 0
        with open(vocab_file, 'r') as file:
            while True:
                token = file.readline()
                if not token:
                    break
                vocab[token.strip()] = idx
                idx += 1
        return list(vocab.keys())

    def __len__(self):
        return len(self.idx2word)
