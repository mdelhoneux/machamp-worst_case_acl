import logging
import random
import os
import math
from overrides import overrides
from torch.utils import data

from allennlp.data.samplers import BatchSampler, BucketBatchSampler
#from allennlp.common.util import lazy_groups_of
from collections import deque, defaultdict
from typing import Tuple, Iterable, Optional, List, Iterator, TypeVar
from allennlp.data.batch import Batch
from allennlp.data.instance import Instance
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params

from machamp.modules.controller import Controller

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

A = TypeVar("A")

def lazy_groups_of(indices: Iterable[A], group_size: int, sizes: Iterable[A], batch_size, max_tokens) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    cur_batch = []
    cur_size = 0
    for indice, size in zip(indices, sizes):
        if (len(cur_batch) == batch_size and batch_size != -1) or (max_tokens != -1 and cur_size + size[0] > max_tokens):
            yield cur_batch
            cur_batch = []
            cur_size = 0
        else:
            cur_batch.append(indice)
            cur_size += size[0]
    if len(cur_batch) > 0:
        yield cur_batch

def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise

def group_by_dataset(instances: List[Instance]) -> Tuple[List[int], List[Instance]]:
    #from machamp's bucket_batch_sampler
    # returns list of tuples; one tuple for each dataset
    # tuple contains list of instances with their index in the full instances list
    by_datasets = {}
    for instanceIdx, instance in enumerate(instances):
        dataset = instance.fields['dataset'].label
        if dataset not in by_datasets:
            by_datasets[dataset] = ([], [])
        by_datasets[dataset][0].append(instanceIdx)
        by_datasets[dataset][1].append(instance)
    for dataset in by_datasets:
        yield by_datasets[dataset]

@BatchSampler.register("acl_sampler")
class ACLSampler(BucketBatchSampler):
    def __init__(self,
            data_source: data.Dataset,
            batch_size:int = 8,
            controller: Controller = None,
            shuffle: bool = True,
            instances_per_epoch: int = None,
            max_tokens: Optional[int] = None,
            sorting_keys: List[str] = None,
            padding_noise: float = 0.1,
            cache_instances: bool = False,
            track_epoch: bool = False,
            drop_last: bool = False,
            skip_smaller_batches: bool = False) -> None:
        super().__init__(data_source, -1, sorting_keys, padding_noise, False)
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.vocab = data_source.vocab
        self.data_source = data_source
        self.shuffle = shuffle
        self.first = True
        self.drop_last = drop_last
        self.controller = controller
        self.n_task = 2 #gets overwritten
        self.controller.set_lengths(len(self.data_source))
        self.controller.initialization()


    def __iter__(self, initialize=True) -> BucketBatchSampler:
        if initialize:
            self.controller.initialization()
        if self.first and os.path.isfile('docs/machamp.txt'):
            champTxt = "\nMaChAmp succesfully initialized\n"
            for line in open('docs/machamp.txt'):
                champTxt += line
            logger.info(champTxt)
            self.first = False

        is_train = True
        for instance in self.data_source.instances:
            is_train = instance['metadata'].metadata['is_train']
            break

        self.all_iters = []
        for dataset_indices, dataset_instances in group_by_dataset(self.data_source.instances):
            sorted_indices, lengths = self._argsort_by_padding(dataset_instances, dataset_indices)
            group_iter = []
            dataset_batches = []
            for group in lazy_groups_of(sorted_indices, self.batch_size, lengths, self.batch_size, self.max_tokens):
                batch_indices = list(group)
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue
                dataset_batches.append(batch_indices)
                group_iter.extend(dataset_batches)
            if self.shuffle:
                random.shuffle(group_iter)
            self.all_iters.append(iter(group_iter))

        if self.shuffle:
            random.shuffle(self.all_iters)
        return self

    def __len__(self):
        return len(self.controller)

    def __next__(self):
        local_task_idx = self.controller.get_task_id()
        if local_task_idx is not None:
            try:
                batch = next(self.all_iters[local_task_idx])
            except StopIteration:
                #this will recreate the all_iters
                # Sheng's code did this just for the task
                # but I don't know how easy it is to read just the task data.
                #TODO: might be an improvement
                self.__iter__(initialize=False)
                batch = next(self.all_iters[local_task_idx])
            return batch
        else:
            raise StopIteration

    def _argsort_by_padding(
        self, instances: Iterable[Instance], indices: Iterable[int]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        taken from machamp/modules/bucket_batch_sampler
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        if not self.sorting_keys:
            logger.info("No sorting keys given; trying to guess a good one")
            self._guess_sorting_keys(instances)
            logger.info(f"Using {self.sorting_keys} as the sorting keys")

        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:
                if field_name not in instance.fields:
                    raise ConfigurationError(
                        f'Sorting key "{field_name}" is not a field in instance. '
                        f"Available fields/keys are {list(instance.fields.keys())}."
                    )
                lengths.append(len(instance.fields[field_name]))
                noisy_lengths.append(add_noise_to_value(lengths[-1], self.padding_noise))
            instances_with_lengths.append((noisy_lengths, lengths, instance))

        with_indices = [(x, i) for i, x in zip(indices, instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return (
            [instance_with_index[-1] for instance_with_index in with_indices],
            [instance_with_index[0][1] for instance_with_index in with_indices],
        )

    def __len__(self):
        batch_count_float = len(self.data_source) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)

