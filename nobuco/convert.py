from __future__ import annotations

import time
import traceback
import warnings
from typing import Callable, Dict, Collection, Optional, List, Union, Tuple

import torch
from nobuco.converters.tensor import permute_pytorch2keras
from torch import nn
import tensorflow as tf
from tensorflow import keras

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy, TF_TENSOR_CLASSES, TraceLevel
from nobuco.converters.channel_ordering import t_pytorch2keras, set_channel_order, t_keras2pytorch
from nobuco.converters.validation import validate, ValidationResult, ConversionResult
from nobuco.layers.channel_order import ChangeOrderingLayer
from nobuco.layers.container import TransientContainer
from nobuco.layers.stub import UnimplementedOpStub, FailedConversionStub
from nobuco.util import get_torch_tensor_identifier, collect_recursively, replace_recursively_func, \
    clone_torch_tensors_recursively
from nobuco.entity.keras import KerasConvertedNode
from nobuco.entity.pytorch import PytorchNode, PytorchNodeHierarchy
from nobuco.trace.trace import Tracer
from nobuco.converters.node_converter import CONVERTER_DICT, Pytorch2KerasNodeConverter
from nobuco.vis.html_stylizer import HtmlStylizer

# Load default converters
# noinspection PyUnresolvedReferences
from nobuco.node_converters import *

# Trace pytorch ops right away
Tracer.decorate_all()


def has_converter(node: PytorchNode, converter_dict: Dict[object, Pytorch2KerasNodeConverter]) -> bool:
    return node.get_type() in converter_dict.keys()


def find_converter(node: PytorchNode, converter_dict: Dict[object, Pytorch2KerasNodeConverter]) -> Optional[Pytorch2KerasNodeConverter]:
    return converter_dict.get(node.get_type(), None)


def find_unimplemented(hierarchy: PytorchNodeHierarchy, converter_dict: Dict[object, Pytorch2KerasNodeConverter]) -> Optional[PytorchNodeHierarchy]:
    # Test if the node itself has a converter
    if has_converter(hierarchy.node, converter_dict):
        return None
    elif len(hierarchy.children) == 0:
        return PytorchNodeHierarchy(hierarchy.node, hierarchy.children)
    else:
        children_unimplemented = []
        for child in hierarchy.children:
            child_unimplemented = find_unimplemented(child, converter_dict)
            if child_unimplemented is not None:
                children_unimplemented.append(child_unimplemented)

        if len(children_unimplemented) > 0:
            # The node is unimplemented
            return PytorchNodeHierarchy(hierarchy.node, children_unimplemented)


def convert_node(node: PytorchNode, node_converter: Pytorch2KerasNodeConverter) -> Callable:
    input_args = node.input_args
    if node.instance is not None:
        input_args = (node.instance, *input_args)

    # clone the inputs to be on the safe side
    input_args, input_kwargs = clone_torch_tensors_recursively((input_args, node.input_kwargs))
    layer = node_converter.convert(_pytorch_node=node, *input_args, **input_kwargs)
    return layer


def convert_container(
        node: PytorchNode,
        children: Collection[PytorchNodeHierarchy],
        children_converted_nodes: Collection[KerasConvertedNode],
        input_names: List[int], output_names: List[int],
        output_tensors: List[torch.Tensor],
        constants_to_variables: bool) -> TransientContainer:

    def collect_disconnected_node_names(input_names, output_names, keras_converted_nodes) -> Collection[int]:
        input_set = set(input_names)
        disconnected_set = set()
        for converted_node in keras_converted_nodes:
            for input_name in converted_node.input_names:
                if input_name not in input_set:
                    disconnected_set.add(input_name)
            for output_name in converted_node.output_names:
                input_set.add(output_name)
        for name in output_names:
            if name not in input_set:
                disconnected_set.add(name)
        return list(disconnected_set)

    def collect_tensors_by_ids(tensor_ids: Collection[int], node_hierarchies: Collection[PytorchNodeHierarchy], output_tensors) -> Dict[int, torch.Tensor]:
        result = {}
        for hierarchy in node_hierarchies:
            for input_tensor in hierarchy.node.input_tensors:
                input_id = get_torch_tensor_identifier(input_tensor)
                if input_id in tensor_ids:
                    result[input_id] = input_tensor
        for tensor in output_tensors:
            output_id = get_torch_tensor_identifier(tensor)
            if output_id in tensor_ids:
                result[output_id] = tensor
        return result

    disconnected_names = collect_disconnected_node_names(input_names, output_names, children_converted_nodes)
    disconnected_tensors_pytorch = collect_tensors_by_ids(disconnected_names, children, output_tensors)
    disconnected_tensors_keras = {k: t_pytorch2keras(t) for k, t in disconnected_tensors_pytorch.items()}

    outputs_template = node.make_outputs_template()

    node_keras = TransientContainer.create(input_names, output_names, outputs_template, disconnected_tensors_keras, children_converted_nodes, constants_to_variables=constants_to_variables)
    return node_keras


def convert_hierarchy(
        node_hierarchy: PytorchNodeHierarchy,
        converter_dict: Dict[object, Pytorch2KerasNodeConverter],
        reuse_layers: bool = True,
        full_validation: bool = True,
        tolerance=1e-4,
        constants_to_variables: bool = True,
) -> KerasConvertedNode:

    def convert(hierarchy: PytorchNodeHierarchy, converted_op_dict:Dict, reuse_layers: bool, full_validation: bool, depth):
        node = hierarchy.node
        children = hierarchy.children

        input_names = node.input_names
        output_names = node.output_names

        converter = find_converter(node, converter_dict)

        keras_op, children_converted_nodes = converted_op_dict.get(node.get_op(), (None, []))
        node_is_reusable = False
        if reuse_layers and keras_op is not None:
            conversion_result = ConversionResult(converted_manually=False, is_duplicate=True, converter=converter)
        elif has_converter(node, converter_dict):
            children_converted_nodes = []
            node_converter: Pytorch2KerasNodeConverter = converter_dict.get(node.get_type(), None)
            try:
                keras_op = convert_node(node, node_converter)
                node_is_reusable = node_converter.reusable
            except Exception as e:
                warnings.warn(f"Conversion exception on node '{node.get_type().__name__}': {e}")
                traceback.print_exc()
                keras_op = FailedConversionStub(node.get_op())
            conversion_result = ConversionResult(converted_manually=True, converter=converter)
        elif len(children) > 0:
            children_converted_nodes = [convert(child, converted_op_dict, reuse_layers, full_validation, depth + 1) for child in children]
            keras_op = convert_container(node, children, children_converted_nodes, input_names, output_names, node.output_tensors, constants_to_variables=constants_to_variables)

            connectivity_status = keras_op.get_connectivity_status()
            if not connectivity_status.is_connected():
                warnings.warn(f'[{node.get_type()} : {keras_op}] is disconnected!', category=RuntimeWarning)
            conversion_result = ConversionResult(converted_manually=False, connectivity_status=connectivity_status, converter=converter)
        else:
            children_converted_nodes = []
            keras_op = UnimplementedOpStub(node.get_op())
            conversion_result = ConversionResult(converted_manually=False, is_implemented=False, converter=converter)

        if full_validation or depth == 0:
            validation_result = validate(node, node.wrapped_op.op, keras_op, node.input_args, node.input_kwargs, node.output_tensors, node.get_type(), tolerance=tolerance)
        else:
            validation_result = None

        if reuse_layers and node_is_reusable and node.is_module() and not isinstance(keras_op, TransientContainer):
            converted_op_dict[node.get_op()] = (keras_op, children_converted_nodes)

        keras_converted_node = KerasConvertedNode(keras_op, node, validation_result, conversion_result, input_names, output_names, children_converted_nodes)
        return keras_converted_node

    # converted_op_dict = CONVERTED_OP_DICT
    converted_op_dict = {}
    return convert(node_hierarchy, converted_op_dict, reuse_layers=reuse_layers, full_validation=full_validation, depth=0)


def collect_validation_results(keras_node: KerasConvertedNode) -> Dict[PytorchNode, ValidationResult]:
    validation_result_dict = {keras_node.pytorch_node: keras_node.validation_result}
    for child in keras_node.children:
        validation_result_dict.update(collect_validation_results(child))
    return validation_result_dict


def collect_conversion_results(keras_node: KerasConvertedNode) -> Dict[PytorchNode, ConversionResult]:
    conversion_result_dict = {keras_node.pytorch_node: keras_node.conversion_result}
    for child in keras_node.children:
        conversion_result_dict.update(collect_conversion_results(child))
    return conversion_result_dict


def prepare_inputs_tf(inputs_pt, inputs_channel_order, input_shapes):

    def collect_func(obj):
        return isinstance(obj, torch.Tensor)

    def replace_func(obj: torch.Tensor) -> torch.Tensor:
        if isinstance(inputs_channel_order, Dict):
            channel_order = inputs_channel_order.get(obj, ChannelOrder.TENSORFLOW)
        else:
            channel_order = inputs_channel_order

        tens = t_pytorch2keras(obj, channel_order=channel_order)

        if input_shapes is not None and obj in input_shapes:
            shape = input_shapes.get(obj)
            if channel_order == ChannelOrder.TENSORFLOW:
                shape = permute_pytorch2keras(shape)
        else:
            shape = tens.shape
        return set_channel_order(keras.backend.placeholder(shape=shape, dtype=tens.dtype), channel_order)

    return replace_recursively_func(inputs_pt, collect_func, replace_func)


def postprocess_outputs_tf(outputs, outputs_channel_order):
    processed = []
    outputs = collect_recursively(outputs, TF_TENSOR_CLASSES)
    for i, output in enumerate(outputs):
        if isinstance(outputs_channel_order, Dict):
            channel_order = outputs_channel_order.get(i, None)
        else:
            channel_order = outputs_channel_order

        if channel_order == ChannelOrder.TENSORFLOW:
            strategy = ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER
        elif channel_order == ChannelOrder.PYTORCH:
            strategy = ChannelOrderingStrategy.FORCE_PYTORCH_ORDER
        else:
            strategy = ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS

        ordering_layer = ChangeOrderingLayer(func=lambda x: x, channel_ordering_strategy=strategy, autocast=False)
        output = ordering_layer(output)
        processed.append(output)

    # Ensure outputs order
    processed = [tf.identity(t) for t in processed]
    return processed


def pytorch_to_keras(
        model: nn.Module | Callable,
        args: List[object] = None,
        kwargs: Dict[str, object] = None,
        input_shapes: Dict[torch.Tensor, Collection[Optional[int]]] = None,
        inputs_channel_order: ChannelOrder | Dict[torch.Tensor, ChannelOrder] = ChannelOrder.TENSORFLOW,
        outputs_channel_order: ChannelOrder | Dict[int, ChannelOrder] | None = None,
        trace_shape: bool = False,
        constants_to_variables: bool = True,
        full_validation: bool = True,
        validation_tolerance: float = 1e-4,
        return_outputs_pt: bool = False,
        save_trace_html: bool = False,
        debug_traces: TraceLevel = TraceLevel.DEFAULT,
) -> keras.Model | Tuple[keras.Model, object]:
    """Converts Pytorch program to Keras graph

    Args:
        model: Pytorch module or function
        args: Pytorch model arguments.
            Default: None
        kwargs: Pytorch model keyword arguments.
            Default: None
        input_shapes: Desired input shapes. Set dim to None for dynamic size.
            Default: None
        inputs_channel_order: Desired channel order of the converted graph's inputs.
            Default: `ChannelOrder.TENSORFLOW`
        outputs_channel_order: Desired channel order of the converted graph's outputs.
            Set to None if you don't care to minimize the amount of transpositions.
            Default: None
        trace_shape: If True, replaces all `torch.Tensor.shape` and `torch.Tensor.size` calls with `nobuco.shape`
            so they can be traced and added to the Keras graph.
            Default: False
        constants_to_variables: If True, replaces each Pytorch constant tensor with `WeightLayer` returning corresponding value.
            Due to how Keras works, this allows multiple layers to use the same set of constants without parameter duplication.
            Default: True
        full_validation: If True, all nested modules will be independently tested for conversion accuracy.
            Set to False to reduce conversion time.
            Default: True
        validation_tolerance: Absolute conversion discrepancy above which alert will be triggered.
            Default: 1e-4
        return_outputs_pt: If True, returns Pytorch outputs. Useful for implementing nested converters.
            Default: False
        save_trace_html: If True, saves model trace as `trace.html`.
            Default: False
        debug_traces:
            Default: `TraceLevel.DEFAULT`

    Returns:
        Converted Keras model
        Pytorch outputs if `return_outputs_pt` is True
    """

    if args is None:
        args = []

    if kwargs is None:
        kwargs = {}

    start = time.time()
    node_hierarchy = Tracer.trace(model, trace_shape, args, kwargs)

    keras_converted_node = convert_hierarchy(node_hierarchy, CONVERTER_DICT,
                                             reuse_layers=True, full_validation=full_validation, constants_to_variables=constants_to_variables,
                                             tolerance=validation_tolerance,
                                             )

    validation_result_dict = collect_validation_results(keras_converted_node)
    conversion_result_dict = collect_conversion_results(keras_converted_node)

    vis_params = {
        'validation_result_dict': validation_result_dict,
        'conversion_result_dict': conversion_result_dict,
        'debug_traces': debug_traces,
    }

    print(node_hierarchy.__str__(with_legend=True, **vis_params))

    if save_trace_html:
        html = node_hierarchy.__str__(with_legend=True, stylizer=HtmlStylizer(), **vis_params)
        with open('trace.html', 'w') as f:
            f.write(html)

    unimplemented_hierarchy = find_unimplemented(node_hierarchy, CONVERTER_DICT)
    if unimplemented_hierarchy is not None:
        print('Unimplemented nodes:')
        print(unimplemented_hierarchy.__str__(**vis_params))
        raise Exception('Unimplemented nodes')

    keras_op = keras_converted_node.keras_op

    args_tf, kwargs_tf = prepare_inputs_tf((args, kwargs), inputs_channel_order, input_shapes)
    outputs_tf = keras_op(*args_tf, **kwargs_tf)
    outputs_tf = postprocess_outputs_tf(outputs_tf, outputs_channel_order)

    inputs_tf_flat = collect_recursively((args_tf, kwargs_tf), TF_TENSOR_CLASSES)
    keras_model = keras.Model(inputs_tf_flat, outputs_tf)

    elapsed = time.time() - start
    print(f'Conversion complete. Elapsed time: {elapsed:.2f} sec.')

    if return_outputs_pt:
        outputs_pt = node_hierarchy.node.outputs
        return keras_model, outputs_pt
    else:
        return keras_model
