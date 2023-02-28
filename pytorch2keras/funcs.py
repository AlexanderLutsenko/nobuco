from torch import Tensor

from pytorch2keras.trace.trace import Tracer


@Tracer.traceable()
def force_tensorflow_order(input: Tensor) -> Tensor:
    return input


@Tracer.traceable()
def force_pytorch_order(input: Tensor) -> Tensor:
    return input
