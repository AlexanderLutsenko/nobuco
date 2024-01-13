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
    def __init__(self, diff_abs, diff_rel, status):
        self.diff_abs = diff_abs
        self.diff_rel = diff_rel
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


def validate(node, pytorch_op, keras_op, input_args, input_kwargs, output_tensors, op_type, tolerance=1e-4) -> ValidationResult:
    try:
        diffs = validate_diff_default(keras_op, pytorch_op, input_args, input_kwargs, output_tensors)

        if len(diffs):
            for i, (diff_abs, diff_rel) in enumerate(diffs):
                if diff_abs > tolerance or math.isnan(diff_abs):
                    warn_string = f'[{op_type}|{str_parents(node)}] conversion procedure might be incorrect: max. discrepancy for output #{i} is {diff_abs:.5f}'
                    if diff_rel is not None:
                        warn_string += f' ({(diff_rel*100):.3f}%)'
                    warnings.warn(warn_string, category=RuntimeWarning)
            diff_abs, diff_rel = max(diffs, key=lambda x: x[0])
        else:
            diff_abs, diff_rel = 0, 0

        if diff_abs > tolerance or math.isnan(diff_abs):
            status = ValidationStatus.INACCURATE
        else:
            status = ValidationStatus.SUCCESS
        return ValidationResult(diff_abs, diff_rel, status)
    except Exception as e:
        warnings.warn(f"Validation exception on node '{op_type.__name__}': {e}")
        traceback.print_exc()
        return ValidationResult(None, None, ValidationStatus.FAIL)


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

        if t_tf.dtype != t_pt.dtype:
            raise Exception(f"Tensor dtypes don't match: (Pytorch) {t_pt.dtype} vs {t_tf.dtype} (Tensorflow)")

    def calc_diff(t1, t2):
        def calc_diff_numerical(t1, t2):
            t1_nan_mask = torch.isnan(t1)
            t2_nan_mask = torch.isnan(t2)
            if not torch.equal(t1_nan_mask, t2_nan_mask):
                return np.nan, None

            if torch.is_complex(t1) or torch.is_complex(t2):
                t1_inf_mask = t1_ninf_mask = torch.isinf(t1)
                t2_inf_mask = t2_ninf_mask = torch.isinf(t2)
                if not torch.equal(t1_inf_mask, t2_inf_mask):
                    return np.inf, None
            else:
                t1_inf_mask = torch.isposinf(t1)
                t2_inf_mask = torch.isposinf(t2)
                if not torch.equal(t1_inf_mask, t2_inf_mask):
                    return np.inf, None

                t1_ninf_mask = torch.isneginf(t1)
                t2_ninf_mask = torch.isneginf(t2)
                if not torch.equal(t1_ninf_mask, t2_ninf_mask):
                    return np.inf, None

            t1_mask = t1_nan_mask | t1_inf_mask | t1_ninf_mask
            t2_mask = t2_nan_mask | t2_inf_mask | t2_ninf_mask
            diff = t1[~t1_mask] - t2[~t2_mask]
            if diff.numel() == 0:
                return 0, 0
            else:
                # return diff.abs().max().numpy()
                diff_abs = diff.abs().numpy()
                diff_max = diff_abs.max()
                diff_argmax = diff_abs.argmax()
                t1_el = t1[~t1_mask][diff_argmax]
                t2_el = t2[~t2_mask][diff_argmax]
                diff_rel = max(np.abs(t1_el / t2_el), np.abs(t2_el / t1_el)) - 1
                return diff_max, diff_rel

        def calc_diff_boolean(t1, t2):
            diff = t1 ^ t2
            return diff.to(torch.float32).max().numpy(), None

        if t1.numel() == t2.numel() == 0:
            return 0, None

        if t1.dtype == t2.dtype == torch.bool:
            return calc_diff_boolean(t1, t2)
        else:
            return calc_diff_numerical(t1, t2)

    diffs = [calc_diff(t1, t2) for t1, t2 in zip(outputs_tf_converted, outputs_pt)]
    return diffs
