
class UnimplementedOpStub:
    def __init__(self, original_node):
        self.original_node = original_node

    def __call__(self, *args, **kwargs):
        raise Exception(f'Unimplemented op: {self.original_node}')

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.original_node})"


class FailedConversionStub:
    def __init__(self, original_node):
        self.original_node = original_node

    def __call__(self, *args, **kwargs):
        raise Exception(f'Failed conversion: {self.original_node}')

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.original_node})"