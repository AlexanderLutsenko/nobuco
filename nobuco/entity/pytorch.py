import time
from typing import Collection, List

import torch
from nobuco.locate.link import get_link_to_obj
from torch import nn

from nobuco.commons import TraceLevel
from nobuco.converters.validation import ValidationStatus

from nobuco.converters.channel_ordering import make_template_recursively
from nobuco.util import collect_recursively, get_torch_tensor_identifier
from nobuco.vis.console_stylizer import ConsoleStylizer


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
        id = get_torch_tensor_identifier(tensor)
        name = self.name_dict.get(id, None)
        if name is None:
            name = self.last_name
            self.last_name += 1
        self.name_dict[id] = name
        return name


class TierStatus:
    def __init__(self, style: str, is_last: bool, is_disconnected: bool, has_parent_outputs: bool):
        self.style = style
        self.is_last = is_last
        self.is_disconnected = is_disconnected
        self.has_parent_outputs = has_parent_outputs


class PytorchNode:
    def __init__(self, wrapped_op: WrappedOp, module_name, parent_list, instance, input_args, input_kwargs, outputs, is_inplace, traceback_summary):
        self.wrapped_op = wrapped_op
        self.module_name = module_name
        self.parent_list = parent_list
        self.instance = instance
        self.input_args = input_args
        self.input_kwargs = input_kwargs
        self.outputs = outputs
        self.is_inplace = is_inplace
        self.traceback_summary = traceback_summary

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
        return [get_torch_tensor_identifier(t) for t in self.input_tensors]

    @property
    def output_names(self):
        return [get_torch_tensor_identifier(t) for t in self.output_tensors]

    @property
    def output_types(self):
        return [t.dtype for t in self.output_tensors]

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
                tensor_name_assigner: TensorNameAssigner = None,
                stylizer=None,
                debug_traces: TraceLevel = TraceLevel.NEVER,
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

        if stylizer is None:
            stylizer = ConsoleStylizer()

        def to_str(obj, self_connectivity_status=None, parent_connectivity_status=None, is_input=False):
            if isinstance(obj, torch.Tensor):
                style = stylizer.connectivity_status_to_style(obj, self_connectivity_status, parent_connectivity_status, is_input)
                name = tensor_name_assigner.get_name(obj)
                dtype = str(obj.dtype).split('.')[-1]
                return stylizer.stylize(f'{dtype}_{name}' + '<' + ','.join([str(s) for s in obj.shape]) + '>', style)
            elif isinstance(obj, torch.Size):
                return f'Size{tuple(obj)}'
            elif isinstance(obj, str):
                return f'"{str(obj)}"'
            elif isinstance(obj, slice):
                return f"{to_str(obj.start, connectivity_status, parent_connectivity_status, is_input) if obj.start is not None else ''}:" \
                       f"{to_str(obj.stop, connectivity_status, parent_connectivity_status, is_input) if obj.stop is not None else ''}" \
                       f"{f':{to_str(obj.step, connectivity_status, parent_connectivity_status, is_input)}' if obj.step is not None else ''}"
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
            elif hasattr(obj, '__dict__'):
                fields = '(' + ', '.join([f'{k} = {to_str(v, connectivity_status, parent_connectivity_status, is_input)}' for k, v in vars(obj).items()]) + ')'
                return f"{obj.__class__.__name__}{fields}"
            else:
                return str(obj)

        status = None
        converted_manually = None
        is_implemented = None
        is_duplicate = None
        is_disconnected = None
        connectivity_status = None
        is_inplace = None

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
            is_inplace = self.node.is_inplace

        result = ''

        if with_legend:
            result += 'Legend:\n'
            st = stylizer.validation_status_to_style(ValidationStatus.SUCCESS, False)
            result += '    ' + stylizer.stylize('Green', st) + ' — conversion successful\n'
            st = stylizer.validation_status_to_style(ValidationStatus.INACCURATE, False)
            result += '    ' + stylizer.stylize('Yellow', st) + ' — conversion imprecise\n'
            st = stylizer.validation_status_to_style(ValidationStatus.FAIL, False)
            result += '    ' + stylizer.stylize('Red', st) + ' — conversion failed\n'
            st = stylizer.style_not_implemented
            result += '    ' + stylizer.stylize('Red', st) + ' — no converter found\n'
            st = stylizer.validation_status_to_style(None, True)
            result += '    ' + stylizer.stylize('Bold', st) + ' — conversion applied directly\n'
            result += '    ' + '*' + ' — subgraph reused\n'
            st = stylizer.style_inverse
            result += '    ' + stylizer.stylize('Tensor', st) + " — this output is not dependent on any of subgraph's input tensors\n"
            st = stylizer.style_underl
            result += '    ' + stylizer.stylize('Tensor', st) + " — this input is a parameter / constant\n"
            st = stylizer.style_grey
            result += '    ' + stylizer.stylize('Tensor', st) + " — this tensor is useless\n"
            result += '\n'

        style = stylizer.validation_status_to_style(status, converted_manually)
        if is_implemented == False: # noqa
            style = stylizer.style_not_implemented

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
                            symbol = ' └·'
                        elif tier_is_last:
                            symbol = ' ├·'
                        else:
                            symbol = ' │ '
                    else:
                        symbol = ' │ '
                res += stylizer.stylize(symbol, s.style) + ' '

            return res

        if status == ValidationStatus.INACCURATE or is_disconnected or is_inplace:
            result += get_tier_str(tier_statuses, is_additional=True)

            if status == ValidationStatus.INACCURATE:
                status_str = f' (!) Max diff {validation_result.diff_abs:.5f} '
                if validation_result.diff_rel is not None:
                    status_str += f'({validation_result.diff_rel*100:.3f}%) '
                result += stylizer.stylize(status_str, stylizer.styles_join(style, stylizer.style_inverse)) + ' '
            if is_disconnected:
                result += stylizer.stylize(f' (!) Subgraph disconnected ', stylizer.style_inverse) + ' '
            if is_inplace:
                result += stylizer.stylize(f' (!) Inplace ', stylizer.style_inplace) + ' '
            result += '\n'

        are_problems = status == ValidationStatus.FAIL or status == ValidationStatus.INACCURATE or is_disconnected

        if debug_traces == TraceLevel.ALWAYS or (debug_traces == TraceLevel.DEFAULT and are_problems):
            result += get_tier_str(tier_statuses, is_additional=True)
            summary = self.node.traceback_summary
            result += stylizer.stylize(f' I ', stylizer.style_grey) + stylizer.stylize(f' File "{summary.filename}", line {summary.lineno}', stylizer.style_grey) + ' '
            result += '\n'

            link = get_link_to_obj(self.node.get_op().__class__)
            if link is not None:
                result += get_tier_str(tier_statuses, is_additional=True)
                result += stylizer.stylize(f' D ', stylizer.style_grey) + stylizer.stylize(f' {link} ', stylizer.style_grey) + ' '
                result += '\n'

            if conversion_result is not None:
                converter_link = conversion_result.get_converter_link()
                if converter_link is not None:
                    result += get_tier_str(tier_statuses, is_additional=True)
                    result += stylizer.stylize(f' C ', style) + stylizer.stylize(f' {converter_link} ', stylizer.style_grey) + ' '
                    result += '\n'

        result += get_tier_str(tier_statuses)
        result += \
            stylizer.stylize(
                f'{"*" if is_duplicate else ""}' + f'{self.node.get_type().__name__}' + f'[{self.node.module_name}]',
                style
            ) + \
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
                                        tier_statuses + [TierStatus(style=style, is_last=is_last, is_disconnected=is_disconnected, has_parent_outputs=has_parent_outputs_list[i])],
                                        validation_result_dict, conversion_result_dict,
                                        with_legend=False, parent_connectivity_status=connectivity_status,
                                        tensor_name_assigner=tensor_name_assigner,
                                        stylizer=stylizer,
                                        debug_traces=debug_traces,
                                        )
        if tier == 0:
            result = stylizer.postprocess(result)
        return result
