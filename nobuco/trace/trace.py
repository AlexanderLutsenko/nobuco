from __future__ import annotations

import inspect
import traceback
import types
from copy import deepcopy
from typing import List, Collection, Callable, Union

import torch
import torchvision
from torch import nn

import nobuco
from nobuco.entity.pytorch import PytorchNode, WrappedOp, PytorchNodeHierarchy
from nobuco.trace.tensor_storage import clone_torch_tensors_recursively_with_cache, TensorStorage
from nobuco.util import collect_recursively, set_torch_tensor_id, get_torch_tensor_identifier, collect_recursively_func


class TracingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func_raw, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        func = Tracer.op_undecorate(func_raw)
        is_decorated = Tracer.is_decorated(func_raw)

        # if func in Tracer.op_trace_blacklist:
        #     with Tracer.tracing_suspend():
        #         args, kwargs = unwrap_torch_tensors_recursively((args, kwargs))
        #         outputs = func(*args, **kwargs)
        #         outputs = wrap_torch_tensors_recursively(outputs)
        #         return outputs
        # elif is_decorated:
        #     with Tracer.tracing_suspend():
        #         args, kwargs = unwrap_torch_tensors_recursively((args, kwargs))
        #     outputs = func(*args, **kwargs)
        #     with Tracer.tracing_suspend():
        #         outputs = wrap_torch_tensors_recursively(outputs)
        #     return outputs

        if is_decorated or func in Tracer.op_trace_blacklist:
            with Tracer.tracing_suspend():
                args, kwargs = unwrap_torch_tensors_recursively((args, kwargs))
                outputs = func(*args, **kwargs)
                outputs = wrap_torch_tensors_recursively(outputs)
                return outputs
        else:
            if hasattr(func, '__objclass__'):
                op_cls_name = func.__objclass__.__name__
            elif hasattr(func, '__module__'):
                op_cls_name = func.__module__
            else:
                raise Exception()

            with torch._C.DisableTorchFunctionSubclass():
                func = Tracer.op_tracing_decorator(func, op_cls_name, need_trace_deeper=False)
                outputs = func(*args, **kwargs)
                return outputs


def wrap_torch_tensors_recursively(obj, annotate=True):
    # Here I try to evade Pytorch's weird behaviour
    collected = collect_recursively_func(obj, lambda obj: isinstance(obj, torch.Tensor) and not isinstance(obj, TracingTensor))
    if not collected:
        # Nothing to wrap, skip
        return obj

    collected = collect_recursively(obj, torch.Tensor)

    def replace_func(obj):
        if not isinstance(obj, TracingTensor):
            wrapped = obj.as_subclass(TracingTensor)
            if annotate:
                set_torch_tensor_id(wrapped, get_torch_tensor_identifier(obj))
            return wrapped
        else:
            return obj

    replace_dict = {id(c): replace_func(c) for c in collected}
    return deepcopy(obj, memo=replace_dict)


def unwrap_torch_tensors_recursively(obj, annotate=True):
    collected = collect_recursively(obj, torch.Tensor)

    def replace_func(obj):
        if isinstance(obj, TracingTensor):
            wrapped = obj.as_subclass(torch.Tensor)
            if annotate:
                set_torch_tensor_id(wrapped, get_torch_tensor_identifier(obj))
            return wrapped
        else:
            return obj

    replace_dict = {id(c): replace_func(c) for c in collected}
    return deepcopy(obj, memo=replace_dict)


def traceable(func_to_trace: Callable):
    if Tracer.is_decorated(func_to_trace):
        return func_to_trace
    else:
        module_suffix = func_to_trace.__qualname__
        module_suffix = '.'.join(module_suffix.split('.')[:-1])
        return Tracer.op_tracing_decorator(func_to_trace, inspect.getmodule(func_to_trace), module_suffix=module_suffix)


def find_call_summary(call_stack: List[traceback.FrameSummary]) -> traceback.FrameSummary:
    call_stack_reversed = list(reversed(call_stack))

    last_idx = 0
    for i, summary in enumerate(call_stack_reversed):
        if '__torch_function__' in summary.name:
            last_idx = i
            break

    for summary in call_stack_reversed[last_idx + 1:]:
        if not (summary.name.endswith('_call_impl') or summary.name.endswith('__call__')):
            return summary


class Tracer:
    
    op_tracing_classes = [
        torch,
        torch.Tensor,
        torch.linalg,
        torch.fft,
        torch.special,
        torch.functional,
        torch.nn.functional,
        torchvision.transforms.functional,
        torchvision.utils,
        torchvision.ops,
        torchvision.ops.boxes,
        torchvision.ops.poolers,
    ]

    op_blacklist = [
        torch.Tensor.__init__,  # Breaks stuff
        torch._C._disabled_torch_dispatch_impl,  # Breaks stuff
        torch.nn.functional.handle_torch_function,  # Looks ugly in the log
        torch.Tensor.__getitem__,  # Breaks tensor creation, e.g. torch.tensor([torch.ones([1])])
        torch.Tensor.__torch_function__,  # Used to break stuff
        torch.Tensor._make_subclass,  # Used to break stuff
    ]

    op_trace_blacklist = [
        torch.Tensor.__format__,
        torch.Tensor.__str__,
        torch.Tensor.__repr__,
    ]

    op_whitelist = {
        (torch.Tensor, torch.Tensor.__setitem__),
    }

    _parent_list: List[WrappedOp] = []
    _node_list: List[PytorchNode] = []
    _tensor_storage: TensorStorage = TensorStorage()

    _tracing_enabled: bool = False
    _trace_shape: bool = False

    class tracing_set_allowed(object):
        def __init__(self, is_allowed):
            self.is_allowed = is_allowed

        def __enter__(self):
            self.tracing_enabled = Tracer._tracing_enabled
            Tracer._tracing_enabled &= self.is_allowed

        def __exit__(self, *args):
            Tracer._tracing_enabled = self.tracing_enabled

    class tracing_suspend(tracing_set_allowed):
        def __init__(self):
            super().__init__(False)

    class register_parent(object):
        def __init__(self, parent):
            self.parent = parent

        def __enter__(self):
            Tracer._parent_list.append(self.parent)

        def __exit__(self, *args):
            Tracer._parent_list = Tracer._parent_list[:-1]

    @staticmethod
    def is_tracing_enabled() -> bool:
        return Tracer._tracing_enabled

    @staticmethod
    def append_node(node: PytorchNode):
        return Tracer._node_list.append(node)

    @staticmethod
    def is_decorated(callable) -> bool:
        return hasattr(callable, '__undecorated_func__')

    @staticmethod
    def op_undecorate(callable: Callable) -> Callable:
        if Tracer.is_decorated(callable):
            return callable.__undecorated_func__
        else:
            return callable

    @staticmethod
    def all_tensors_equal(tensors1, tensors2):
        tensors1 = collect_recursively(tensors1, torch.Tensor)
        tensors2 = collect_recursively(tensors2, torch.Tensor)
        assert len(tensors1) == len(tensors2)
        return all(torch.equal(t1.cpu(), t2.cpu()) for t1, t2 in zip(tensors1, tensors2))

    @staticmethod
    def module_forward_tracing_decorator(forward_func):

        def forward(self, *args, **kwargs):
            if Tracer.is_tracing_enabled():
                with torch._C.DisableTorchFunctionSubclass():
                    with Tracer.tracing_suspend():
                        wrapped_op = WrappedOp(self)

                        # Protection from external modification
                        args_clone, kwargs_clone = clone_torch_tensors_recursively_with_cache((args, kwargs), Tracer._tensor_storage)

                        # Inner function may change the input structure, insure against that
                        args_inner, kwargs_inner = deepcopy((args, kwargs), memo={id(t): t for t in collect_recursively((args, kwargs), torch.Tensor)})

                        # Transform `torch.Tensor`s into `TracingTensor`s, just in case
                        args_inner, kwargs_inner = wrap_torch_tensors_recursively((args_inner, kwargs_inner))

                with Tracer.register_parent(wrapped_op):
                    outputs = forward_func(*args_inner, **kwargs_inner)

                with torch._C.DisableTorchFunctionSubclass():
                    with Tracer.tracing_suspend():
                        # Transform `torch.Tensor`s into `TracingTensor`s, just in case
                        outputs = wrap_torch_tensors_recursively(outputs)

                        # Protection from external modification
                        outputs_clone = clone_torch_tensors_recursively_with_cache(outputs, Tracer._tensor_storage)
                        is_inplace = not Tracer.all_tensors_equal((args, kwargs), (args_clone, kwargs_clone))

                        summary: traceback.FrameSummary = find_call_summary(traceback.extract_stack())
                        args_clone, kwargs_clone, outputs_clone = unwrap_torch_tensors_recursively((args_clone, kwargs_clone, outputs_clone))
                        node = PytorchNode(wrapped_op, self.__module__, Tracer._parent_list.copy(), self, args_clone, kwargs_clone, outputs_clone, is_inplace, summary)
                        Tracer.append_node(node)
                    return outputs
            else:
                outputs = forward_func(*args, **kwargs)
            return outputs

        forward.__undecorated_func__ = forward_func
        return forward

    @staticmethod
    def op_tracing_decorator(orig_method, op_cls, module_suffix=None, is_whitelist_op=False, need_trace_deeper=True):

        orig_method = Tracer.op_undecorate(orig_method)

        def decorator(*args, **kwargs):

            def is_duplicate(orig_method):
                if not Tracer._parent_list:
                    return False
                else:
                    parent = Tracer.op_undecorate(Tracer._parent_list[-1].op)
                    return parent == orig_method

            if Tracer.is_tracing_enabled() and not is_duplicate(orig_method):

                trace_deeper = need_trace_deeper

                with torch._C.DisableTorchFunctionSubclass():
                    with Tracer.tracing_suspend():
                        call_method = orig_method
                        call_op_cls = op_cls

                        if Tracer._trace_shape:
                            if orig_method == Tracer.op_undecorate(torch.Tensor.size) or orig_method == Tracer.op_undecorate(torch.Tensor.__getattribute__) and 'shape' in args:
                                call_method = Tracer.op_undecorate(nobuco.shape)
                                call_op_cls = nobuco
                                if 'shape' in args:
                                    args = args[:1]
                                trace_deeper = False

                        wrapped_op = WrappedOp(call_method)

                        # Protection from external modification
                        args_clone, kwargs_clone = clone_torch_tensors_recursively_with_cache((args, kwargs), Tracer._tensor_storage)

                        # Inner function may change the input structure, insure against that
                        args_inner, kwargs_inner = deepcopy((args, kwargs), memo={id(t): t for t in collect_recursively((args, kwargs), torch.Tensor)})

                        # Transform `torch.Tensor`s into `TracingTensor`s, just in case
                        # args_inner, kwargs_inner = wrap_torch_tensors_recursively((args_inner, kwargs_inner))

                        num_input_tensors = len(collect_recursively((args, kwargs), torch.Tensor))

                with Tracer.tracing_set_allowed(trace_deeper):
                    with Tracer.register_parent(wrapped_op):
                        outputs = call_method(*args_inner, **kwargs_inner)

                with torch._C.DisableTorchFunctionSubclass():
                    with Tracer.tracing_suspend():
                        # Transform `torch.Tensor`s into `TracingTensor`s, just in case
                        outputs = wrap_torch_tensors_recursively(outputs)

                        # __setitem__ method is sorta special
                        if orig_method == Tracer.op_undecorate(torch.Tensor.__setitem__):
                            outputs = args[0]

                        num_output_tensors = len(collect_recursively(outputs, torch.Tensor))

                        if is_whitelist_op or (num_input_tensors > 0 and num_output_tensors > 0):
                            if isinstance(call_op_cls, str):
                                module_name = call_op_cls
                            elif isinstance(call_op_cls, types.ModuleType):
                                module_name = call_op_cls.__name__
                            else:
                                module_name = f'{call_op_cls.__module__}.{call_op_cls.__name__}'

                            if module_suffix:
                                module_name += '.' + module_suffix

                            # Protection from external modification
                            outputs_clone = clone_torch_tensors_recursively_with_cache(outputs, Tracer._tensor_storage)

                            is_inplace = not Tracer.all_tensors_equal((args, kwargs), (args_clone, kwargs_clone))

                            summary: traceback.FrameSummary = find_call_summary(traceback.extract_stack())
                            args_clone, kwargs_clone, outputs_clone = unwrap_torch_tensors_recursively((args_clone, kwargs_clone, outputs_clone))
                            node = PytorchNode(wrapped_op, module_name, Tracer._parent_list.copy(), None, args_clone, kwargs_clone, outputs_clone, is_inplace, summary)
                            Tracer.append_node(node)
            else:
                outputs = orig_method(*args, **kwargs)
            return outputs

        decorator.__undecorated_func__ = orig_method
        return decorator

    @staticmethod
    def decorate_module():
        def decorated_new(cls, *args, **kwargs):
            self = object.__new__(cls)
            if not Tracer.is_decorated(self.forward):
                self.forward = types.MethodType(Tracer.module_forward_tracing_decorator(self.forward), self)
            return self
        nn.Module.__new__ = decorated_new

    @staticmethod
    def decorate_ops():
        torch.jit._state.disable()

        for op_cls in Tracer.op_tracing_classes:
            for name, method in inspect.getmembers(op_cls):
                if not (inspect.ismethod(method) or inspect.isfunction(method) or inspect.isbuiltin(method) or inspect.isroutine(method)) \
                        or method in Tracer.op_blacklist:
                    continue

                if not Tracer.is_decorated(method):
                    decorated = Tracer.op_tracing_decorator(method, op_cls, is_whitelist_op=False)
                    setattr(op_cls, name, decorated)

        for op_cls, op in Tracer.op_whitelist:
            if not Tracer.is_decorated(op):
                decorated = Tracer.op_tracing_decorator(op, op_cls, is_whitelist_op=True)
                setattr(op_cls, op.__name__, decorated)

    @staticmethod
    def decorate_all():
        Tracer.decorate_module()
        Tracer.decorate_ops()

    @staticmethod
    def build_hierarchy(node_list: Collection[PytorchNode]) -> PytorchNodeHierarchy:

        def build(node_list: Collection[PytorchNode]) -> List[PytorchNodeHierarchy]:
            hierarchies = []
            while len(node_list) > 0:
                node = node_list.pop(len(node_list) - 1)
                child_nodes = [c for c in node_list if node.wrapped_op in c.parent_list]
                node_list = [n for n in node_list if n not in child_nodes]

                child_hierarchies = build(child_nodes)
                hierarchy = PytorchNodeHierarchy(node, child_hierarchies)
                hierarchies.append(hierarchy)
            return list(reversed(hierarchies))

        hierarchies = build(node_list)
        # assert len(hierarchies) == 1
        return hierarchies[-1]

    @staticmethod
    def trace(module_or_function: nn.Module | Callable, trace_shape: bool, args, kwargs) -> PytorchNodeHierarchy:
        ### Module tracing routines
        def apply_module_tracing_recursively(module):
            for child in module.children():
                apply_module_tracing_recursively(child)

            if not Tracer.is_decorated(module.forward):
                module.forward = types.MethodType(Tracer.module_forward_tracing_decorator(module.forward), module)
            return module

        ### Initiate tracing
        Tracer.decorate_all()

        if isinstance(module_or_function, nn.Module):
            apply_module_tracing_recursively(module_or_function)
        else:
            module_or_function = traceable(module_or_function)

        args, kwargs = wrap_torch_tensors_recursively((args, kwargs))

        Tracer._trace_shape = trace_shape
        Tracer._tracing_enabled = True
        with torch.no_grad():
            module_or_function(*args, **kwargs)
        Tracer._tracing_enabled = False
        Tracer._trace_shape = False

        hierarchy = Tracer.build_hierarchy(Tracer._node_list)

        Tracer._node_list = []
        Tracer._parent_list = []
        Tracer._tensor_storage = TensorStorage()

        return hierarchy
