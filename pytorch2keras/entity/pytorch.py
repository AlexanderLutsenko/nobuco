import time
from typing import Collection, List

import torch
from torch import nn

from pytorch2keras.convert.layers.container import ConnectivityStatus
from pytorch2keras.convert.validation import ValidationStatus
from sty import bg, ef, fg, rs

from pytorch2keras.converters.channel_ordering import make_template_recursively
from pytorch2keras.util import collect_recursively, get_torch_tensor_id


class WrappedOp:
    def __init__(self, op):
        self.op = op
        self.token = time.time_ns()


class FunctionArgs:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class TensorNameAssigner():
    def __init__(self):
        self.name_dict = {}
        self.last_name = 0

    def fill(self, hierarchy):
        node = hierarchy.node
        for t in node.input_tensors:
            self.get_name(t)
        for child_node in hierarchy.children:
            self.fill(child_node)
        for t in node.output_tensors:
            self.get_name(t)

    def get_name(self, tensor):
        id = get_torch_tensor_id(tensor)
        name = self.name_dict.get(id, None)
        if name is None:
            name = self.last_name
            self.last_name += 1
        self.name_dict[id] = name
        return name


class TierStatus:
    def __init__(self, color: str, is_last: bool, is_disconnected: bool, has_parent_outputs: bool):
        self.color = color
        self.is_last = is_last
        self.is_disconnected = is_disconnected
        self.has_parent_outputs = has_parent_outputs


class PytorchNode:
    def __init__(self, wrapped_op: WrappedOp, module_name, parent_list, instance, input_args, input_kwargs, outputs):
        self.wrapped_op = wrapped_op
        self.module_name = module_name
        self.parent_list = parent_list
        self.instance = instance
        self.input_args = input_args
        self.input_kwargs = input_kwargs
        self.outputs = outputs

    def make_inputs_template(self):
        args_template, kwargs_template = make_template_recursively((self.input_args, self.input_kwargs))
        return args_template, kwargs_template

    def make_outputs_template(self):
        outputs_template = make_template_recursively(self.outputs)
        return outputs_template

    @property
    def input_tensors(self):
        return collect_recursively((self.input_args, self.input_kwargs), torch.Tensor)

    @property
    def output_tensors(self):
        return collect_recursively(self.outputs, torch.Tensor)

    @property
    def input_names(self):
        return [get_torch_tensor_id(t) for t in self.input_tensors]

    @property
    def output_names(self):
        return [get_torch_tensor_id(t) for t in self.output_tensors]

    def get_type(self):
        if isinstance(self.wrapped_op.op, torch.nn.Module):
            type = self.wrapped_op.op.__class__
        else:
            type = self.wrapped_op.op
        return type

    def get_op(self):
        return self.wrapped_op.op

    def is_module(self):
        return isinstance(self.wrapped_op.op, torch.nn.Module)


class PytorchNodeHierarchy:
    def __init__(self, node: PytorchNode, children: Collection):
        self.node = node
        self.children = children

    def __str__(self, tier=0,
                tier_statuses=None,
                validation_result_dict=None, conversion_result_dict=None,
                with_legend=False, parent_connectivity_status=None,
                tensor_name_assigner: TensorNameAssigner = None
                ) -> str:
        if tier_statuses is None:
            tier_statuses = []

        if validation_result_dict is None:
            validation_result_dict = {}

        if conversion_result_dict is None:
            conversion_result_dict = {}

        if tensor_name_assigner is None:
            tensor_name_assigner = TensorNameAssigner()
            tensor_name_assigner.fill(self)

        def to_str(obj, self_connectivity_status=None, parent_connectivity_status=None, is_input=False):
            if isinstance(obj, torch.Tensor):
                color = connectivity_status_to_color(obj, self_connectivity_status, parent_connectivity_status, is_input)
                name = tensor_name_assigner.get_name(obj)
                return color + f'tens{name}' + '{' + ','.join([str(s) for s in obj.shape]) + '}' + rs.all
            elif isinstance(obj, torch.Size):
                return f'Size{tuple(obj)}'
            elif isinstance(obj, str):
                return f'"{str(obj)}"'
            elif isinstance(obj, slice):
                return f"{str(obj.start) if obj.start is not None else ''}:{obj.stop if obj.stop is not None else ''}{f':{obj.step}' if obj.step is not None else ''}"
            elif isinstance(obj, type(Ellipsis)):
                return "..."
            elif isinstance(obj, list):
                return '[' + ', '.join([to_str(el, connectivity_status, parent_connectivity_status, is_input) for el in obj]) + ']'
            elif isinstance(obj, tuple):
                return '(' + ', '.join([to_str(el, connectivity_status, parent_connectivity_status, is_input) for el in obj]) + ')'
            elif isinstance(obj, dict):
                return '{' + ', '.join([f'{to_str(k, connectivity_status, parent_connectivity_status, is_input)}: {to_str(v, connectivity_status, parent_connectivity_status, is_input)}' for k, v in obj.items()]) + '}'
            elif isinstance(obj, FunctionArgs):
                return \
                    '(' + \
                    ', '.join([to_str(el, connectivity_status, parent_connectivity_status, is_input) for el in obj.args]) + \
                    (', ' if obj.args and obj.kwargs else '') + \
                    ', '.join([f'{str(k)}={to_str(v, connectivity_status, parent_connectivity_status, is_input)}' for k, v in obj.kwargs.items()]) + \
                    ')'
            elif isinstance(obj, nn.Module):
                return str(obj.__class__.__name__)
            else:
                return str(obj)

        def validation_status_to_color(status: ValidationStatus, converted_manually):
            if status == ValidationStatus.SUCCESS:
                color = fg.green
            elif status == ValidationStatus.FAIL:
                color = fg.red
            elif status == ValidationStatus.INACCURATE:
                color = fg.yellow
            else:
                color = rs.all

            if converted_manually:
                color += ef.bold
            return color

        def connectivity_status_to_color(tensor, self_status: ConnectivityStatus, parent_status: ConnectivityStatus, is_input) -> str:
            color = ''

            if self_status is not None:
                if is_input:
                    if get_torch_tensor_id(tensor) in self_status.unused_inputs:
                        color += fg.da_grey
                else:
                    if get_torch_tensor_id(tensor) in self_status.unreached_outputs:
                        color += ef.inverse

            if parent_status is not None:
                if is_input:
                    if get_torch_tensor_id(tensor) in parent_status.unprovided_inputs:
                        color += ef.underl
                else:
                    if get_torch_tensor_id(tensor) in parent_status.unused_nodes:
                        color += fg.da_grey
            return color

        color_not_implemented = fg.red + ef.inverse

        status = None
        converted_manually = None
        is_implemented = None
        is_duplicate = None
        is_disconnected = None
        connectivity_status = None

        validation_result = validation_result_dict.get(self.node, None)
        if validation_result is not None:
            status = validation_result.status

        conversion_result = conversion_result_dict.get(self.node, None)
        if conversion_result is not None:
            converted_manually = conversion_result.converted_manually
            is_implemented = conversion_result.is_implemented
            is_duplicate = conversion_result.is_duplicate
            if conversion_result.connectivity_status is not None:
                connectivity_status = conversion_result.connectivity_status
                is_disconnected = not conversion_result.connectivity_status.is_connected()

        result = ''

        if with_legend:
            result += 'Legend:\n'
            c = validation_status_to_color(ValidationStatus.SUCCESS, False)
            result += '    ' + c + 'Green' + rs.all + ' — conversion successful\n'
            c = validation_status_to_color(ValidationStatus.INACCURATE, False)
            result += '    ' + c + 'Yellow' + rs.all + ' — conversion imprecise\n'
            c = validation_status_to_color(ValidationStatus.FAIL, False)
            result += '    ' + c + 'Red' + rs.all + ' — conversion failed\n'
            c = color_not_implemented
            result += '    ' + c + 'Red' + rs.all + ' — no converter found\n'
            c = validation_status_to_color(None, True)
            result += '    ' + c + 'Bold' + rs.all + ' — conversion applied directly\n'
            result += '    ' + '*' + ' — subgraph reused\n'
            c = ef.inverse
            result += '    ' + c + 'Tensor' + rs.all + " — this output is not dependent on any of subgraph's input tensors\n"
            c = ef.underl
            result += '    ' + c + 'Tensor' + rs.all + " — this input is a parameter / constant\n"
            c = fg.da_grey
            result += '    ' + c + 'Tensor' + rs.all + " — this tensor is useless\n"
            result += '\n'

        color = validation_status_to_color(status, converted_manually)
        if is_implemented == False: # noqa
            color = color_not_implemented

        def get_tier_str(tier_statuses, is_additional=False):
            res = ''
            tiers_is_last = [False] * len(tier_statuses)
            if len(tiers_is_last) > 0:
                tiers_is_last[-1] = True

            block_has_ended_list: List[bool] = [s.is_last for s in tier_statuses]
            acc = len(self.children) == 0 or is_duplicate
            for i in range(len(block_has_ended_list) - 1, -1, -1):
                acc = block_has_ended_list[i] and acc
                block_has_ended_list[i] = acc

            for i, (s, tier_is_last, block_has_ended) in enumerate(zip(tier_statuses, tiers_is_last, block_has_ended_list)):
                if is_additional:
                    symbol = ' │ '
                else:
                    if s.is_last and not tier_is_last:
                        if block_has_ended:
                            symbol = ' └ '
                        else:
                            symbol = ' │ '
                    elif s.has_parent_outputs:
                        if s.is_last and block_has_ended:
                            symbol = ' └' + ef.bold + '·' + rs.all
                        elif tier_is_last:
                            symbol = ' ├' + ef.bold + '·' + rs.all
                        else:
                            symbol = ' │ '
                    else:
                        symbol = ' │ '
                res += f'{s.color}{symbol}{rs.all} '

            return res

        if status == ValidationStatus.INACCURATE or is_disconnected:
            result += get_tier_str(tier_statuses, is_additional=True)

            if status == ValidationStatus.INACCURATE:
                result += color + ef.inverse + f' (!) Max diff {validation_result.diff} ' + rs.all + ' '
            if is_disconnected:
                result += ef.inverse + f' (!) Subgraph disconnected ' + rs.all + ' '
            result += '\n'

        result += get_tier_str(tier_statuses)
        result += \
            color + \
            f'{"*" if is_duplicate else ""}' + \
            f'{self.node.get_type().__name__}' + f'[{self.node.module_name}]' + \
            rs.all + \
            f'{to_str(FunctionArgs(self.node.input_args, self.node.input_kwargs), connectivity_status, parent_connectivity_status, is_input=True)}' + \
            f' -> {to_str(self.node.outputs, connectivity_status, parent_connectivity_status)}' + \
            '\n'

        if not is_duplicate:

            parent_output_names = set(self.node.output_names)
            has_parent_outputs_list = []
            for child in reversed(self.children):
                outputs_intersection = parent_output_names.intersection(child.node.output_names)
                parent_output_names = parent_output_names.difference(outputs_intersection)
                has_parent_outputs = len(outputs_intersection) > 0
                has_parent_outputs_list.append(has_parent_outputs)
            has_parent_outputs_list = list(reversed(has_parent_outputs_list))

            for i, child in enumerate(self.children):
                is_last = i == len(self.children) - 1
                result += child.__str__(tier + 1,
                                        tier_statuses + [TierStatus(color=color, is_last=is_last, is_disconnected=is_disconnected, has_parent_outputs=has_parent_outputs_list[i])],
                                        validation_result_dict, conversion_result_dict,
                                        with_legend=False, parent_connectivity_status=connectivity_status,
                                        tensor_name_assigner=tensor_name_assigner
                                        )
        return result
