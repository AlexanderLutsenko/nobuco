from copy import deepcopy
import torch
from nobuco.util import collect_recursively, set_torch_tensor_id, get_torch_tensor_identifier


class TensorStorage:
    def __init__(self):
        self.storage: dict[any, list[torch.Tensor]] = {}

    def get(self, tensor):
        key = get_torch_tensor_identifier(tensor)
        bucket = self.storage.get(key, [])
        for b_tensor in bucket:
            if torch.equal(tensor, b_tensor):
                return b_tensor
        return None

    def add(self, tensor: torch.Tensor):
        key = get_torch_tensor_identifier(tensor)
        bucket = self.storage.get(key, [])
        bucket.append(tensor)
        self.storage[key] = bucket


def clone_torch_tensors_recursively_with_cache(obj, storage: TensorStorage):
    collected = collect_recursively(obj, torch.Tensor)

    def clone(tensor):
        tensor_id = get_torch_tensor_identifier(tensor)
        cloned = tensor.cpu().clone()
        set_torch_tensor_id(cloned, tensor_id)
        return cloned

    def replace_func(tensor):
        tensor = clone(tensor)
        cached = storage.get(tensor)
        if cached is None:
            storage.add(tensor)
            return tensor
        else:
            return cached

    replace_dict = {id(c): replace_func(c) for c in collected}
    return deepcopy(obj, memo=replace_dict)
