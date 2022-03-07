import sys
import json
import torch
import random
import copy
import numpy as np
from shutil import copyfile
import math
from allennlp.common.registrable import Registrable
from machamp.modules.controller import Controller

class BasicPhiController(Controller):
    def __init__(self, phi=0.3, K=2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.phi = float(phi)
        self.task_index = -1
        self.length = 0
        self.cnt = 0
        self.sampled_cnt = [0] * self.n_task

        self.is_first_loop = True
        self.not_loop_total = K * self.n_task

    def initialization(self):
        super().initialization()
        self.task_index = -1
        self.is_first_loop = True
        self.cnt = 0
        self.sampled_cnt = [0] * self.n_task

    def __len__(self):
        return self.lengths

    def choose_task_index_to_update(self):
        if self.cnt > self.not_loop_total:
            self.cnt = 0
            self.is_first_loop = True
            # option 1: current loss
            losses = self.calculate_loss()
            # option 2: current smoothing loss
            if random.random() < self.phi:
                arg_task_index = np.argmax(losses)
                # print("Choose max index, ", arg_task_index)
            else:
                p = np.array(losses)
                p /= p.sum()
                arg_task_index = np.random.choice(list(range(self.n_task)), p=p, replace=False)
                # print("Choose random index, ", arg_task_index)
            self.cur_step += len(self.buffer[arg_task_index])
            return arg_task_index
        else:
            return None


@Controller.register("acl_controller")
class ACLController(BasicPhiController):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # exp3 algorithm
        self.w = [1.0] * self.n_task
        self.epsilon = 0.001
        self.threshold = self.max_cnt // 3
        self.pi = self.calculate_pi()

    def initialization(self):
        super().initialization()
        self.w = [1.0] * self.n_task
        self.pi = self.calculate_pi()

    def calculate_pi(self):
        pi = np.array(self.w)
        return (1-self.epsilon)*pi/pi.sum() + self.epsilon/self.n_task

    def update_weight(self, task_id, reward):
        r = reward/self.pi[task_id]
        self.w[task_id] *= math.exp(self.epsilon * r / self.n_task)
        self.w[task_id] = min(self.w[task_id], 100)

    def get_task_id(self):
        if self.cur_step >= self.max_step:
            return None
        if self.is_first_loop:
            self.task_index += 1
            # if buffer surpass threshold, then pass the task
            while self.task_index < self.n_task and len(self.buffer[self.task_index]) > self.threshold:
                self.task_index += 1
            if self.task_index >= self.n_task:
                self.task_index = 0
                self.is_first_loop = False
        else:
            self.cnt += 1
            self.pi = self.calculate_pi()
            self.task_index = np.random.choice(list(range(self.n_task)), p=self.pi, replace=False)
            self.sampled_cnt[self.task_index] += 1
        return self.task_index

    def choose_task_index_to_update(self):
        arg_task_index = super().choose_task_index_to_update()
        if arg_task_index is not None:
            # update weight
            max_sampled_cnt = max(self.sampled_cnt)
            # losses = [task_loss.get_loss() for task_loss in self.task_losses]
            # max_loss = max(losses)
            for i in range(self.n_task):
                # if max_loss > 0.0:
                #     importance = losses[i] / max_loss
                # else:
                #     importance = 0.0
                if arg_task_index == i:
                    self.update_weight(i, 1.0 * self.sampled_cnt[i] / max_sampled_cnt)
                else:
                    self.update_weight(i, -1.0 * self.sampled_cnt[i] / max_sampled_cnt)
            self.sampled_cnt = [0] * self.n_task
        return arg_task_index

    def summary(self):
        st = super().summary()
        w = ["%.4f" % v for v in self.w]
        st += 'List of self.w : {}\n'.format(", ".join(w))
        pi = ["%.4f" % v for v in self.pi]
        st += 'List of self.pi : {}\n'.format(", ".join(pi))
        return st

class ChangingController(ACLController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.phi = min(1.0, epoch*0.2)
        print("self phi", self.phi)


@Controller.register("random")
class RandomController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_indices = []

    def __len__(self):
        return sum(self.lengths)

    def initialization(self):
        super().initialization()
        self.task_indices = []
        for i, length in enumerate(self.lengths):
            self.task_indices += [i] * length
        random.shuffle(self.task_indices)

    def get_task_id(self):
        if self.cur_step >= self.max_step:
            return None
        self.cur_step += 1
        return self.task_indices[self.cur_step - 1]

    def choose_task_index_to_update(self):
        choice = [i for i, buf in enumerate(self.buffer) if len(buf) > 0]
        if len(choice) == 1:
            return choice[0]
        elif len(choice) > 1:
            print("TWO MORE CHOICES!")
            arg_task_index = np.random.choice(choice, replace=False)
            return arg_task_index
        else:
            return None

