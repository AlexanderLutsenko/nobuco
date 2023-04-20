from typing import Collection

from nobuco.converters.validation import ValidationResult, ConversionResult
from nobuco.entity.pytorch import PytorchNode


class KerasConvertedNode:
    def __init__(self, keras_op, pytorch_node: PytorchNode,
                 validation_result: ValidationResult, conversion_result: ConversionResult,
                 input_names: Collection[int], output_names: Collection[int],
                 children: Collection):
        self.keras_op = keras_op
        self.pytorch_node = pytorch_node
        self.validation_result = validation_result
        self.conversion_result = conversion_result
        self.input_names = input_names
        self.output_names = output_names
        self.children = children
