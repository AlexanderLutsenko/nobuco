from sty import ef, fg, rs

from nobuco.converters.validation import ValidationStatus
from nobuco.commons import ConnectivityStatus
from nobuco.util import get_torch_tensor_identifier


class ConsoleStylizer:

    def postprocess(self, text):
        return text

    def stylize(self, text: str, style: str):
        return style + text + rs.all

    def validation_status_to_style(self, status: ValidationStatus, converted_manually):
        if status == ValidationStatus.SUCCESS:
            style = fg.green
        elif status == ValidationStatus.FAIL:
            style = fg.red
        elif status == ValidationStatus.INACCURATE:
            style = fg.yellow
        else:
            style = rs.all

        if converted_manually:
            style += ef.bold
        return style

    def connectivity_status_to_style(self, tensor, self_status: ConnectivityStatus, parent_status: ConnectivityStatus, is_input) -> str:
        style = ''
        if self_status is not None:
            if is_input:
                if get_torch_tensor_identifier(tensor) in self_status.unused_inputs:
                    style += fg.da_grey
            else:
                if get_torch_tensor_identifier(tensor) in self_status.unreached_outputs:
                    style += ef.inverse

        if parent_status is not None:
            if is_input:
                if get_torch_tensor_identifier(tensor) in parent_status.unprovided_inputs:
                    style += ef.underl
            else:
                if get_torch_tensor_identifier(tensor) in parent_status.unused_nodes:
                    style += fg.da_grey
        return style

    def styles_join(self, style1: str, style2: str) -> str:
        return style1 + style2

    style_not_implemented = fg.red + ef.inverse
    style_bold = ef.bold
    style_inverse = ef.inverse
    style_inplace = fg.blue + ef.inverse
    style_underl = ef.underl
    style_grey = fg.da_grey
