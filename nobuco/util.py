import random
import time
from copy import deepcopy
from typing import Callable, Tuple

import torch
from torch import nn


def find_index(collection, el):
    for i, c in enumerate(collection):
        if c is el:
            return i
    return None


# Tensor identifier contains time_ns to be truly unique,
# as tensors pop in and out of existence and `Two objects with non-overlapping lifetimes may have the same id() value`.
def get_torch_tensor_identifier(tensor):
    if hasattr(tensor, 'original_id'):
        return tensor.original_id
    else:
        tid = int(f'{time.time_ns()}{random.randint(0, 10^9)}')
        tensor.original_id = tid
        return tid


def set_torch_tensor_id(tensor, id):
    tensor.original_id = id


def collect_recursively_func(obj, predicate: Callable[[object], bool]):
    collected = []
    memo_ids = []

    def collect(obj):
        if predicate(obj):
            collected.append(obj)
        elif id(obj) not in memo_ids:
            memo_ids.append(id(obj))
            if isinstance(obj, (list, tuple)):
                for el in obj:
                    collect(el)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    collect(k)
                    collect(v)
            elif isinstance(obj, slice):
                collect(obj.start)
                collect(obj.stop)
                collect(obj.step)
            elif hasattr(obj, '__dict__') and not isinstance(obj, nn.Module):
                v = vars(obj)
                collect(v)

    collect(obj)
    return collected


def collect_recursively(obj, classes):
    def predicate(obj):
        return isinstance(obj, classes)
    return collect_recursively_func(obj, predicate)


def replace_recursively_func(obj, collect_func: Callable[[object], bool], replace_func: Callable[[object], object]):
    collected = collect_recursively_func(obj, collect_func)
    replace_dict = {id(c): replace_func(c) for c in collected}
    replaced = deepcopy(obj, memo=replace_dict)
    return replaced


def clone_torch_tensors_recursively(obj, annotate=True):
    collected = collect_recursively(obj, torch.Tensor)

    def replace_func(obj):
        if obj.is_leaf:
            cloned = obj.clone()
            if annotate:
                set_torch_tensor_id(cloned, get_torch_tensor_identifier(obj))
            return cloned
        else:
            return obj

    replace_dict = {id(c): replace_func(c) for c in collected}
    return deepcopy(obj, memo=replace_dict)


def str_parents(node):
    def get_type(op):
        if isinstance(op, torch.nn.Module):
            type = op.__class__
        else:
            type = op
        return type
    return '->'.join([get_type(w_op.op).__name__ for w_op in node.parent_list])
