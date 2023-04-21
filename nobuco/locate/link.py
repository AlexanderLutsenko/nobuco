import inspect


# https://stackoverflow.com/a/64945941/4850610

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
    try:
        if isinstance(obj, property):
            obj = obj.fget
        file = inspect.getfile(obj)
        line = inspect.getsourcelines(obj)[1]
        return get_link(file=file, line=line)
    except Exception:
        return None
