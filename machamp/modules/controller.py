import json
import torch
import random
import copy
import numpy as np
from shutil import copyfile
import math
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from machamp.modules.queue import Queue
from allennlp.common.util import lazy_groups_of

class Controller(Registrable):
    #TODO: no default for n_task and batch size
    def __init__(self, trainer=None, n_task=2, dataset_names=None, dataset_sizes=None, max_cnt=20, batch_size=8, rebatch_size=None, max_step=1000, tensorboard=None, log_filename=None) -> None:
        self.trainer = trainer
        self.n_task = n_task
        self.buffer = [Queue(max_cnt, batch_size=batch_size, rebatch_size=rebatch_size) for _ in range(n_task)]
        self.task_losses = [buff.smooth_loss for buff in self.buffer]
        self.cur_step = 0
        self.cur_epoch = 0
        self.max_step = max_step
        self.max_cnt = max_cnt
        self.tensorboard = tensorboard
        self.sampled_cnt = [0] * n_task
        self.chosen_cnt = [0] * n_task
        self.trained_cnt = [0] * n_task
        self.dataset_names = dataset_names
        # the number of samples
        self.dataset_sizes = dataset_sizes
        # the number of batchs
        self.lengths = 0
        # print("dataset sizes, ", dataset_sizes)
        assert batch_size is not None, "batch size should not be None"
        self.batch_size = batch_size
        self.rebatch_size = rebatch_size
        self.name_dict = {}
        self.scaled_dict = {}
        self.updates = 0
        if dataset_names is None:
            self.dataset_names = ["task_%02d"%i for i in range(n_task)]

        for i, (x, y) in enumerate(zip(self.dataset_names, self.chosen_cnt)):
            self.name_dict[x] = y
            if self.dataset_sizes:
                self.scaled_dict[x] = self.trained_cnt[i] * 1.0 / self.dataset_sizes[i]

        self.global_step = 0
        if log_filename:
            self.log_file = open(log_filename, "w")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def insert(self, task_id, data, loss):
        self.buffer[task_id].append((data, loss))
        self.task_losses[task_id].add(loss)
        self.global_step += 1
        self.sampled_cnt[task_id] += 1

    def calculate_loss(self):
        losses = []
        for i, data in enumerate(self.buffer):
            loss = data.calculate_loss()
            losses.append(loss)
        return losses

    def set_lengths(self, lengths):
        self.lengths = copy.deepcopy(lengths)

    def initialization(self):
        self.max_step = self.lengths
        self.cur_step = 0
        for data in self.buffer:
            data.empty()

    def get_loss(self):
        avg_loss = ["%.5f"%task_loss.get_loss() for task_loss in self.task_losses]
        out_loss = ["%.5f"%task_loss.get_out_loss() for task_loss in self.task_losses]
        loss_change = ["%.5f"%task_loss.get_loss_change() for task_loss in self.task_losses]
        min_loss = ["%.5f"%task_loss.min_loss for task_loss in self.task_losses]
        min_out_loss = ["%.5f"%task_loss.min_out_loss for task_loss in self.task_losses]
        return avg_loss, out_loss, loss_change, min_loss, min_out_loss

    def __len__(self):
        raise NotImplementedError("__length__ function not implemented!")

    def get_task_id(self):
        raise NotImplementedError("Getting task index function must be overwritten!")

    def choose_task_index_to_update(self):
        raise NotImplementedError("Choosing update task index function must be overwritten!")

    def update_chosen_cnt(self, task_id):
        # tensorboard record chosen count
        self.chosen_cnt[task_id] += 1
        self.trained_cnt[task_id] += self.buffer[task_id].sample_cnt()

        task_name = self.dataset_names[task_id]


        self.name_dict[task_name] = self.chosen_cnt[task_id]
        if self.dataset_sizes:
            self.scaled_dict[task_name] = self.trained_cnt[task_id]*1.0 / self.dataset_sizes[task_id]        
        loss = self.buffer[task_id].calculate_loss()
        ema_loss = self.task_losses[task_id].get_loss()

        # tensorboard
        if self.tensorboard and random.random() < 0.3:
            self.tensorboard.add_scalars('train/chosen', self.name_dict, global_step=self.global_step)
            self.tensorboard.add_scalar('train/loss_%s'%task_name, loss, global_step=self.global_step)
            self.tensorboard.add_scalars('train/task_loss', {task_name: loss}, global_step=self.global_step)
            if self.dataset_sizes:
                self.tensorboard.add_scalars('train/scaled_chosen', self.scaled_dict, global_step=self.global_step)

        if self.log_file:
            line = {'task': task_name, 'chosen': self.name_dict[task_name]}
            if self.dataset_sizes:
                line['scaled_chosen'] = self.scaled_dict[task_name]
            self.write_file(**line)
            line = {'task': task_name, 'loss': loss, "ema_loss": ema_loss}
            self.write_file(**line)
            line = {'task': task_name, 'sampled_cnt': self.sampled_cnt[task_id], "pop_cnt": self.buffer[task_id].pop_cnt}
            line['valid_cnt'] = line['sampled_cnt'] - line['pop_cnt']
            self.write_file(**line)

    def step(self):
        loss = 0.0
        arg_task_index = self.choose_task_index_to_update()
        #print(f'Task chosen: {arg_task_index}')
        if arg_task_index is not None:
            self.update_chosen_cnt(arg_task_index)
            # update task
            for batch in self.buffer[arg_task_index]:
                batch_outputs = self.trainer.batch_outputs(batch, for_training=True)
                loss = batch_outputs.get("loss")
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")
                if self.trainer._scaler is not None:
                    self.trainer._scaler.scale(loss).backward()
                else:
                    loss.backward()

            self.updates +=1
            self.buffer[arg_task_index].empty()
        return loss


    def write_file(self, **kwargs):
        line = {"global_step": self.global_step, "epoch": self.epoch, "cur_step": self.cur_step}
        line.update(kwargs)
        json.dump(line, self.log_file, indent=2)

    def summary(self):
        st = "Current Step {0} / {1}  ({2:.2%})\n".format(self.cur_step, self.max_step, self.cur_step*1.0/self.max_step)
        avg_loss, out_loss, loss_change, min_loss, min_out_loss = self.get_loss()
        st += 'List of Task Names: {}\n'.format(", ".join(self.dataset_names))
        if self.cur_step == 0:
            if self.dataset_sizes is not None:
                dataset_sizes = ["%d"%v for v in self.dataset_sizes]
                st += 'List of dataset size {}\n'.format(", ".join(dataset_sizes))
                total_size = sum(self.dataset_sizes)
                dataset_sizes = ["%.4f"%(v*1.0/total_size) for v in self.dataset_sizes]
                st += 'List of dataset percentage {}\n'.format(", ".join(dataset_sizes))
        loss_values = ["%.6f" % v for v in self.calculate_loss()]
        st += 'List of current loss {}\n'.format(", ".join(loss_values))
        st += 'List of average smoothing loss {}\n'.format(", ".join(avg_loss))
        st += 'List of out_loss {}\n'.format(", ".join(out_loss))
        st += 'List of loss_change {}\n'.format(", ".join(loss_change))
        st += 'List of min_loss {}\n'.format(", ".join(min_loss))
        st += 'List of min_out_loss {}\n'.format(", ".join(min_out_loss))
        chosen = ["%s:%d"%(k,v) for k, v in self.name_dict.items()]
        st += 'List of chosen times {}\n'.format(", ".join(chosen))
        chosen = ["%s:%.3f"%(k,v) for k, v in self.scaled_dict.items()]
        st += 'List of scaled chosen times {}\n'.format(", ".join(chosen))
        buffer_cnts = ["%d"% len(v) for v in self.buffer]
        st += 'List of buffer size {}\n'.format(", ".join(buffer_cnts))
        sampled_cnt = ["%d" % v  for v in self.sampled_cnt]
        st += 'List of sampled count {}\n'.format(", ".join(sampled_cnt))
        pop_cnt = ["%d" % queue.pop_cnt for queue in self.buffer]
        st += 'List of pop count {}\n'.format(", ".join(pop_cnt))

        return st

