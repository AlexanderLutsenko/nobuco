from typing import Callable, Tuple

import torch


def find_index(collection, el):
    for i, c in enumerate(collection):
        if c is el:
            return i
    return None


def replace_recursively_func(obj, replace_func: Callable[[object], Tuple[bool, object]]):
    if isinstance(obj, (list, tuple)):
        result = []
        for el in obj:
            result.append(replace_recursively_func(el, replace_func))
        result = obj.__class__(result)
        return result
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            k = replace_recursively_func(k, replace_func)
            v = replace_recursively_func(v, replace_func)
            result[k] = v
        return result

    ret, replacement = replace_func(obj)
    if ret:
        return replacement
    else:
        return obj


def replace_recursively(obj, replace_map):
    def replace_func(obj):
        if obj in replace_map.keys():
            replacement = replace_map[obj]
            return True, replacement
        else:
            return False, obj

    return replace_recursively_func(obj, replace_func)


def get_torch_tensor_id(tensor):
    if hasattr(tensor, 'original_id'):
        return tensor.original_id
    else:
        return id(tensor)


def clone_torch_tensors_recursively(obj, annotate=True):
    def replace_func(obj):
        if isinstance(obj, torch.Tensor):
            cloned = obj.clone()
            if annotate:
                cloned.original_id = get_torch_tensor_id(obj)
            return True, cloned
        else:
            return False, obj

    return replace_recursively_func(obj, replace_func)


def collect_recursively(container, classes):
    result = []

    def replace_func(obj):
        if isinstance(obj, classes):
            result.append(obj)
        return False, obj

    replace_recursively_func(container, replace_func)
    return result


def str_parents(node):
    def get_type(op):
        if isinstance(op, torch.nn.Module):
            type = op.__class__
        else:
            type = op
        return type
    return '->'.join([get_type(w_op.op).__name__ for w_op in node.parent_list])
