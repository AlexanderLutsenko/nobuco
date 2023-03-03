from nobuco.trace.trace import Tracer


@Tracer.traceable()
def force_tensorflow_order(inputs):
    return inputs


@Tracer.traceable()
def force_pytorch_order(inputs):
    return inputs
