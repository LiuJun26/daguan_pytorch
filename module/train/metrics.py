import torch
from tqdm import tqdm
import numpy as nmp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):

        raise NotImplementedError

    def reset(self):

        raise NotImplementedError

    def value(self):

        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class LMAccuracy(Metric):

    def __init__(self, top_K =1):

        super(LMAccuracy).__init__()
        self.top_K = top_K
        self.reset()

    def __call__(self, logits, target):
        # 返回指定维度值最大的索引　此处ｙ，竖直方向
        pred = torch.argmax(logits, 1)
        active_acc = target.view(-1) != -1
        active_pred = pred[active_acc]
        active_labels = target[active_acc]

        correct = active_pred.eq(active_labels)
        self.correct_k = correct.float().sum(0)
        self.total = active_labels.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        # 准确度
        return "accuracy"

