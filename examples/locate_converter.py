import torch
import torch.nn.functional as F
from torch import nn
import nobuco

node = torch.Tensor.repeat
# node = F.relu_
# node = nn.Conv2d

location_link, source_code = nobuco.locate_converter(node)
print('Converter location:')
print(location_link)
print('Converter source code:')
print(source_code)