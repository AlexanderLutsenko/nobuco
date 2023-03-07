from copy import deepcopy
from typing import Callable, Tuple

import torch
from torch import nn


def find_index(collection, el):
    for i, c in enumerate(collection):
        if c is el:
            return i
    return None


# def replace_recursively_func(obj, replace_func: Callable[[object], Tuple[bool, object]]):
#     if isinstance(obj, (list, tuple)):
#         result = []
#         for el in obj:
#             result.append(replace_recursively_func(el, replace_func))
#         result = obj.__class__(result)
#         return result
#     elif isinstance(obj, dict):
#         result = {}
#         for k, v in obj.items():
#             k = replace_recursively_func(k, replace_func)
#             v = replace_recursively_func(v, replace_func)
#             result[k] = v
#         return result
#
#     ret, replacement = replace_func(obj)
#     if ret:
#         return replacement
#     else:
#         return obj
#
#
# def replace_recursively(obj, replace_map):
#     def replace_func(obj):
#         if obj in replace_map.keys():
#             replacement = replace_map[obj]
#             return True, replacement
#         else:
#             return False, obj
#
#     return replace_recursively_func(obj, replace_func)


def get_torch_tensor_id(tensor):
    if hasattr(tensor, 'original_id'):
        return tensor.original_id
    else:
        return id(tensor)


def set_torch_tensor_id(tensor, id):
    tensor.original_id = id


# def clone_torch_tensors_recursively(obj, annotate=True):
#     def replace_func(obj):
#         if isinstance(obj, torch.Tensor):
#             cloned = obj.clone()
#             if annotate:
#                 cloned.original_id = get_torch_tensor_id(obj)
#             return True, cloned
#         else:
#             return False, obj
#
#     return replace_recursively_func(obj, replace_func)


# def collect_recursively(container, classes):
#     result = []
#
#     def replace_func(obj):
#         if isinstance(obj, classes):
#             result.append(obj)
#         return False, obj
#
#     replace_recursively_func(container, replace_func)
#     return result


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
                set_torch_tensor_id(cloned, get_torch_tensor_id(obj))
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
