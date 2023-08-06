import traceback
import warnings
from enum import Enum
import math

import numpy as np
import torch

from nobuco.commons import ChannelOrder, TF_TENSOR_CLASSES
from nobuco.converters.channel_ordering import t_keras2pytorch, pytorch2keras_recursively
from nobuco.locate.link import get_link_to_obj
from nobuco.util import str_parents, collect_recursively


class ValidationStatus(Enum):
    SUCCESS = 1
    FAIL = 2
    INACCURATE = 3


class ValidationResult:
    def __init__(self, diff, status):
        self.diff = diff
        self.status = status


class ConversionResult:
    def __init__(self, converted_manually, is_implemented=True, is_duplicate=False, connectivity_status=None, converter=None):
        self.converted_manually = converted_manually
        self.is_implemented = is_implemented
        self.is_duplicate = is_duplicate
        self.connectivity_status = connectivity_status
        self.converter = converter

    def get_converter_link(self):
        if self.converter is None:
            return None
        else:
            convert_func = self.converter.convert_func
            location_link = get_link_to_obj(convert_func)
            return location_link


def validate(node, pytorch_op, keras_op, input_args, input_kwargs, output_tensors, op_type, tolerance=1e-4):
    try:
        diffs = validate_diff_default(keras_op, pytorch_op, input_args, input_kwargs, output_tensors)

        if len(diffs):
            for i, diff in enumerate(diffs):
                if diff > tolerance or math.isnan(diff):
                    warnings.warn(
                        f'[{op_type}|{str_parents(node)}] conversion procedure might be incorrect: max. discrepancy for output #{i} is {diff:5f}',
                        category=RuntimeWarning
                    )
                    pass
            diff = max(diffs)
        else:
            diff = 0

        if diff > tolerance or math.isnan(diff):
            return diff, ValidationStatus.INACCURATE
        else:
            return diff, ValidationStatus.SUCCESS
    except Exception as e:
        # raise Exception(f"Validation exception on node '{op_type.__name__}': {e}")
        warnings.warn(f"Validation exception on node '{op_type.__name__}': {e}")
        traceback.print_exc()
        return None, ValidationStatus.FAIL


def validate_diff_default(keras_op, pytorch_op, args_pt, kwargs_pt, outputs_pt, is_training=False):
    args_tf = pytorch2keras_recursively(args_pt, channel_order=ChannelOrder.TENSORFLOW)
    kwargs_tf = pytorch2keras_recursively(kwargs_pt, channel_order=ChannelOrder.TENSORFLOW)

    outputs_tf = keras_op(*args_tf, **kwargs_tf)

    outputs_tf = collect_recursively(outputs_tf, TF_TENSOR_CLASSES)
    outputs_tf_converted = [t_keras2pytorch(t, restore_channel_order=True) for t in outputs_tf]

    # with torch.no_grad():
    #     outputs_pt = pytorch_op(*args_pt, **kwargs_pt)
    #     outputs_pt = collect_recursively(outputs_pt, torch.Tensor)

    if len(outputs_tf_converted) != len(outputs_pt):
        raise Exception(f"Number of outputs do not match: (Pytorch) {len(outputs_pt)} vs {len(outputs_tf_converted)} (Tensorflow)")

    for i, (t_tf, t_pt) in enumerate(zip(outputs_tf_converted, outputs_pt)):
        if t_tf.shape != t_pt.shape:
            raise Exception(f"Tensor shapes of output #{i} don't match: (Pytorch) {list(t_pt.shape)} vs {list(t_tf.shape)} (Tensorflow)")

        # if t_tf.dtype != t_pt.dtype:
        #     raise Exception(f"Tensor dtypes don't match: (Pytorch) {t_pt.dtype} vs {t_tf.dtype} (Tensorflow)")

    def calc_diff(t1, t2):
        def calc_diff_numerical(t1, t2):
            nan_mask = torch.isnan(t1) & torch.isnan(t2)
            diff = t1[~nan_mask] - t2[~nan_mask]
            if diff.numel() == 0:
                return 0
            else:
                return diff.abs().max().numpy()

        def calc_diff_boolean(t1, t2):
            diff = t1 ^ t2
            return diff.to(torch.float32).max().numpy()

        if t1.numel() == t2.numel() == 0:
            return 0

        if t1.dtype == t2.dtype == torch.bool:
            return calc_diff_boolean(t1, t2)
        else:
            return calc_diff_numerical(t1, t2)

    diffs = [calc_diff(t1, t2) for t1, t2 in zip(outputs_tf_converted, outputs_pt)]
    return diffs
