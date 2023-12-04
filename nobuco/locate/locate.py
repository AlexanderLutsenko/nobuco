import inspect

from nobuco.locate.link import get_link_to_obj
from nobuco.trace.trace import Tracer
from nobuco.converters.node_converter import CONVERTER_DICT


def locate_converter(pytorch_node):
    pytorch_node = Tracer.op_undecorate(pytorch_node)
    node_converter = CONVERTER_DICT.get(pytorch_node, None)
    if node_converter is not None:
        convert_func = node_converter.convert_func
        location_link = get_link_to_obj(convert_func)
        source_code = inspect.getsource(convert_func)
    else:
        location_link, source_code = None, None
    return location_link, source_code
