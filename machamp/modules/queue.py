import logging
import random
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data import Batch
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataloader import DataLoader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DataLoader.register("queue")
class Queue(DataLoader):
    def __init__(self, n=3, batch_size=None, rebatch_size=4):
        self.n = n
        self.data = []
        self.batch_size = batch_size
        self.pop_cnt = 0
        self.sum_loss = 0.0
        self.smooth_loss = DataLoss()
        self.sum_loss = 0.0
        self.is_changed = False
        if batch_size and rebatch_size:
            self.rebatch_size = rebatch_size
            assert batch_size % rebatch_size == 0, "rebatch_size must devide batch_size"
        else:
            self.rebatch_size = rebatch_size

    def append(self, item):
        if len(self.data) >= self.n:
            self.data.pop(0)
            self.pop_cnt += 1
        self.data.append(item)
        self.is_changed = True

    def __len__(self):
        return len(self.data)

    def empty(self):
        self.data = []
        self.is_changed = True

    def __getitem__(self, idx):
        return self.data[idx]

    def calculate_loss(self):
        if self.is_changed:
            self.sum_loss = 0.0
            cnt = len(self.data)
            # if cnt > 0:
            #     return max([loss for data, loss in self.data])
            # return 0.0  
            for i, (data, loss) in enumerate(self.data):
                self.sum_loss += loss
            if cnt > 0:
                self.sum_loss /= cnt
            self.is_changed = False
        return self.sum_loss

    def sample_cnt(self):
        #TODO: this is hacky
        return self.data[0][0]['tokens']['tokens']['token_ids'].size()[1]

    def __iter__(self): 
        # need to put a cap here
        if self.batch_size and self.rebatch_size and (self.batch_size != self.rebatch_size):
            for i, (data, loss) in enumerate(self.data):
                total_meta, total_data = data
                total_data = [torch.split(a_tensor, self.rebatch_size, dim=0) for a_tensor in total_data]
                uids = total_meta['uids']
                meta_list = []
                for j in range(0, self.batch_size, self.rebatch_size):
                    meta = copy.deepcopy(total_meta)
                    meta["uids"] = copy.deepcopy(uids[j:j+self.rebatch_size])
                    # print("batch_uids", meta["uids"])
                    meta_list.append(meta)
                for j, batch_data in enumerate(zip(*total_data)):
                    # print(meta_list[j])
                    # for tmp in batch_data:
                    #     print(tmp.size())
                    print('yield but I do not expect you to come here')
                    yield (meta_list[j], list(batch_data))
        else:
            for i, (data, loss) in enumerate(self.data):
                yield data


class DataLoss():
    def __init__(self, n=30, beta=0.8, beta_2=0.9):
        self.n = n
        self.beta = beta
        self.beta_2 = beta_2
        self.data = []
        self.loss = 0.0
        self.out_loss = None
        self.min_loss = 1000000.0
        self.min_out_loss = 1000000.0

    def add(self, value):
        if len(self.data) >= self.n:
            removed_data = self.data[0]
            self.data.pop(0)
            if self.out_loss is None:
                self.out_loss = removed_data
            else:
                self.out_loss = self.update(self.out_loss, removed_data, self.beta_2)

        self.data.append(value)
        self.loss = self.update(self.loss, value, self.beta)
        if self.loss < self.min_loss:
            self.min_loss = self.loss
        if self.out_loss is not None and self.out_loss < self.min_out_loss:
            self.min_out_loss = self.out_loss

    def update(self, avg, value, beta):
        return avg * beta + value * (1.0 - beta)

    def get_loss(self):
        return self.loss

    def get_out_loss(self):
        if self.out_loss is None:
            return 0
        else:
            return self.out_loss

    def get_loss_change(self):
        if self.out_loss is None:
            return 0.0
        else:
            return self.out_loss - self.loss
