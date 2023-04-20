import inspect
from nobuco.trace.trace import Tracer
from nobuco.converters.node_converter import CONVERTER_DICT


def get_link(file=None, line=None):
    """ Print a link in PyCharm to a line in file.
        Defaults to line where this function was called. """
    if file is None:
        file = inspect.stack()[1].filename
    if line is None:
        line = inspect.stack()[1].lineno
    string = f'File "{file}", line {max(line, 1)}'.replace("\\", "/")
    return string


def get_link_to_obj(obj):
    """ Print a link in PyCharm to a module, function, class, method or property. """
    if isinstance(obj, property):
        obj = obj.fget
    file = inspect.getfile(obj)
    line = inspect.getsourcelines(obj)[1]
    return get_link(file=file, line=line)


def locate_converter(pytorch_node):
    pytorch_node = Tracer.op_unwrap(pytorch_node)
    node_converter = CONVERTER_DICT.get(pytorch_node, None)
    if node_converter is not None:
        convert_func = node_converter.convert_func
        location_link = get_link_to_obj(convert_func)
        source_code = inspect.getsource(convert_func)
    else:
        location_link, source_code = None, None
    return location_link, source_code
