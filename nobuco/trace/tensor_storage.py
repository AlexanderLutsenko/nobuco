from copy import deepcopy
import torch
from nobuco.util import collect_recursively, set_torch_tensor_id, get_torch_tensor_identifier


class TensorStorage:
    def __init__(self):
        self.storage: dict[any, list[torch.Tensor]] = {}

    def make_key(self, tensor: torch.Tensor):
        return get_torch_tensor_identifier(tensor), tensor.dtype, tensor.shape

    def get(self, tensor):
        key = self.make_key(tensor)
        bucket = self.storage.get(key, [])
        for b_tensor in bucket:
            if torch.equal(tensor, b_tensor):
                return b_tensor
        return None

    def add(self, tensor: torch.Tensor):
        key = self.make_key(tensor)
        bucket = self.storage.get(key, [])
        bucket.append(tensor)
        self.storage[key] = bucket


def clone_torch_tensors_recursively_with_cache(obj, storage: TensorStorage, annotate=True):
    collected = collect_recursively(obj, torch.Tensor)

    def clone(tensor):
        cloned = tensor.clone()
        set_torch_tensor_id(cloned, get_torch_tensor_identifier(tensor))
        return cloned

    def replace_func(tensor):
        cached = storage.get(tensor)
        if cached is None:
            cached = clone(tensor)
            storage.add(cached)
        return cached

    replace_dict = {id(c): replace_func(c) for c in collected}
    return deepcopy(obj, memo=replace_dict)
