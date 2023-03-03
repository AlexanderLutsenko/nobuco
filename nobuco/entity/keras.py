from typing import Collection

from nobuco.convert.validation import ValidationResult, ValidationStatus, ConversionResult
from nobuco.entity.pytorch import PytorchNode
from sty import bg, ef, fg, rs


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

    # def __str__(self, tier=0, tier_colors=None, validation_result_dict=None) -> str:
    #     if tier_colors is None:
    #         tier_colors = []
    #
    #     if validation_result_dict is None:
    #         validation_result_dict = {}
    #
    #     def to_str(obj):
    #         if isinstance(obj, slice):
    #             return f"{str(obj.start) if obj.start is not None else ''}:{obj.stop if obj.stop is not None else ''}{f':{obj.step}' if obj.step is not None else ''}"
    #         elif isinstance(obj, list):
    #             return '[' + ', '.join([to_str(el) for el in obj]) + ']'
    #         elif isinstance(obj, tuple):
    #             return '(' + ', '.join([to_str(el) for el in obj]) + ')'
    #         elif isinstance(obj, dict):
    #             return '{' + ', '.join([f'{to_str(k)}: {to_str(v)}' for k, v in obj.items()]) + '}'
    #         else:
    #             return str(obj)
    #
    #     validation_result = self.validation_result
    #
    #     if validation_result is not None:
    #         status = validation_result.status
    #         converted_manually = validation_result.converted_manually
    #     else:
    #         status = None
    #         converted_manually = None
    #
    #     if status == ValidationStatus.SUCCESS:
    #         color = fg.green
    #     elif status == ValidationStatus.FAIL:
    #         color = fg.red
    #     elif status == ValidationStatus.INACCURATE:
    #         color = fg.yellow
    #     else:
    #         color = rs.all
    #
    #     if converted_manually:
    #         color += ef.bold
    #
    #     result = ''
    #     if status == ValidationStatus.INACCURATE:
    #         result += f'{"".join([f"{c}│{rs.all}   " for c in tier_colors])}' + \
    #                   color + ef.inverse + f' (!) Max diff {validation_result.diff} ' + rs.all + '\n'
    #
    #     result += \
    #         f'{"".join([f"{c}│{rs.all}   " for c in tier_colors])}' + \
    #         color + \
    #         f'{self.keras_op.__class__.__name__}' + \
    #         rs.all + \
    #         f' args={to_str(self.input_names)}' + \
    #         f' out={to_str(self.output_names)}' + \
    #         '\n'
    #
    #     for child in self.children:
    #         result += child.__str__(tier + 1, tier_colors + [color], validation_result_dict)
    #     return result