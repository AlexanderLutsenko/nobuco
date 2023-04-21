from typing import Dict

from nobuco.converters.validation import ValidationStatus
from nobuco.commons import ConnectivityStatus
from nobuco.util import get_torch_tensor_identifier


class HtmlStylizer:

    def postprocess(self, text):
        text = text.replace('\n', '<br>\n')
        text = \
        '<!DOCTYPE html>\n' + \
        '<html>\n' + \
        '<body style="font-family: monospace">\n' + \
        text + '\n' \
        '</body>\n' + \
        '</html>'
        return text

    def stylize(self, text: str, style: Dict):
        style = style
        css_style_dict = {}
        if 'color' in style and not style.get('inverse', False):
            css_style_dict['color'] = style['color']
        if style.get('inverse', False):
            color = style.get('color', 'black')
            css_style_dict['background-color'] = color
            css_style_dict['color'] = 'white'
        if style.get('bold', False):
            css_style_dict['font-weight'] = 'bold'
        if style.get('underline', False):
            css_style_dict['text-decoration'] = 'underline'

        css_str = ";".join(k + ":" + v for k, v in css_style_dict.items())
        text = text.replace(' ', '&nbsp;')
        return f'<text style="{css_str}">{text}</text>'

    def validation_status_to_style(self, status: ValidationStatus, converted_manually):
        if status == ValidationStatus.SUCCESS:
            style = {'color': 'green'}
        elif status == ValidationStatus.FAIL:
            style = {'color': '#ce0505'}
        elif status == ValidationStatus.INACCURATE:
            style = {'color': '#b28c00'}
        else:
            style = {}

        if converted_manually:
            style = self.styles_join(style, self.style_bold)
        return style

    def connectivity_status_to_style(self, tensor, self_status: ConnectivityStatus, parent_status: ConnectivityStatus, is_input) -> Dict:
        style = {}
        if self_status is not None:
            if is_input:
                if get_torch_tensor_identifier(tensor) in self_status.unused_inputs:
                    style = self.styles_join(style, self.style_grey)
            else:
                if get_torch_tensor_identifier(tensor) in self_status.unreached_outputs:
                    style = self.styles_join(style, self.style_inverse)

        if parent_status is not None:
            if is_input:
                if get_torch_tensor_identifier(tensor) in parent_status.unprovided_inputs:
                    style = self.styles_join(style, self.style_underl)
            else:
                if get_torch_tensor_identifier(tensor) in parent_status.unused_nodes:
                    style = self.styles_join(style, self.style_grey)
        return style

    def styles_join(self, style1: Dict, style2: Dict) -> Dict:
        style1 = style1.copy()
        style1.update(style2)
        return style1

    style_not_implemented = {'color': '#ce0505', 'inverse': True}
    style_bold = {'bold': True}
    style_inverse = {'inverse': True}
    style_inplace = {'color': '#063fdb', 'inverse': True}
    style_underl = {'underline': True}
    style_grey = {'color': '#656565'}
