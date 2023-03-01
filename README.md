# Nobuco

**No** **Bu**llshit **Co**nverter is a tool that lets you translate pytorch models into tensorflow graphs without losing your mind.

- Supports a wide range of architectures
  - [x] Control flow ops (If, While)
  - [x] Recurrent layers (LSTM, GRU)
  - [x] Advanced tensor slicing (except for slice assign, as there's no equivalent in TF)
- Simple
- Easily extensible
- Sanity-preserving, with clear mistake messaging

### Essentials
There's two aspects to converting a graph from one framework to another
1) Faithfully reconstruct the graph structure i.e. how individual operations link together
2) Convert each operation to the target framework

Stage (1) is error-prone and tedious for a human, and that's where a machine comes to rescue. 

Let's say we've got a pytorch module like this one:



````python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1))
        self.act1 = nn.Hardsigmoid()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x
````

````python
dummy_image = torch.rand(size=(1, 3, 256, 256))
pytorch_module = MyModule().eval()

keras_model = pytorch_to_keras(
    pytorch_module, [dummy_image],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)
````

Nobuco traces

<code>
<div style="overflow-x:scroll; white-space: nowrap">
<text style="color:#ce0505">MyModule[__main__]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens12{1,32,256,256}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">BatchNorm2d[torch.nn.modules.batchnorm]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">&nbsp;├·</text> <text style="">batch_norm[torch.nn.functional]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="">tens1{3}</text>, <text style="">tens2{3}</text>, <text style="">tens3{3}</text>, <text style="">tens4{3}</text>, False, 0.1, 1e-05) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">&nbsp;└&nbsp;</text> <text style="">&nbsp;└·</text> <text style="">batch_norm[torch]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="">tens3{3}</text>, <text style="">tens4{3}</text>, <text style="">tens1{3}</text>, <text style="">tens2{3}</text>, False, 0.1, 1e-05, True) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens5{1,3,256,256}</text>) -> <text style="">tens8{1,16,256,256}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens5{1,3,256,256}</text>, <text style="text-decoration:underline">tens6{16,3,3,3}</text>, <text style="text-decoration:underline">tens7{16}</text>, (1, 1), (1, 1), (1, 1), 1) -> <text style="">tens8{1,16,256,256}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:#ce0505">Hardsigmoid[torch.nn.modules.activation]</text>(<text style="">tens8{1,16,256,256}</text>) -> <text style="">tens9{1,16,256,256}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:#ce0505">&nbsp;└·</text> <text style="background-color:#ce0505;color:white">hardsigmoid[torch.nn.functional]</text>(<text style="">tens8{1,16,256,256}</text>, False) -> <text style="">tens9{1,16,256,256}</text><br>
<text style="color:#ce0505">&nbsp;├·</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens9{1,16,256,256}</text>) -> <text style="">tens12{1,32,256,256}</text><br>
<text style="color:#ce0505">&nbsp;└&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens9{1,16,256,256}</text>, <text style="text-decoration:underline">tens10{32,16,3,3}</text>, <text style="text-decoration:underline">tens11{32}</text>, (1, 1), (1, 1), (1, 1), 1) -> <text style="">tens12{1,32,256,256}</text><br>
</div>
</code>

Conversion is done directly. No layers upon layers of abstraction, no obscure intermediate representation, no bullshit.

````python
@converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return tf.keras.activations.hard_sigmoid(input)
    return func
````

<code>
<div style="overflow-x:scroll; white-space: nowrap">
<text style="background-color:#b28c00;color:white">&nbsp;(!)&nbsp;Max&nbsp;diff&nbsp;0.028111&nbsp;</text> <br>
<text style="color:#b28c00">MyModule[__main__]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens12{1,32,256,256}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">BatchNorm2d[torch.nn.modules.batchnorm]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">&nbsp;├·</text> <text style="">batch_norm[torch.nn.functional]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="">tens1{3}</text>, <text style="">tens2{3}</text>, <text style="">tens3{3}</text>, <text style="">tens4{3}</text>, False, 0.1, 1e-05) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">&nbsp;└&nbsp;</text> <text style="">&nbsp;└·</text> <text style="">batch_norm[torch]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="">tens3{3}</text>, <text style="">tens4{3}</text>, <text style="">tens1{3}</text>, <text style="">tens2{3}</text>, False, 0.1, 1e-05, True) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens5{1,3,256,256}</text>) -> <text style="">tens8{1,16,256,256}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens5{1,3,256,256}</text>, <text style="text-decoration:underline">tens6{16,3,3,3}</text>, <text style="text-decoration:underline">tens7{16}</text>, (1, 1), (1, 1), (1, 1), 1) -> <text style="">tens8{1,16,256,256}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="background-color:#b28c00;color:white">&nbsp;(!)&nbsp;Max&nbsp;diff&nbsp;0.042823&nbsp;</text> <br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:#b28c00">Hardsigmoid[torch.nn.modules.activation]</text>(<text style="">tens8{1,16,256,256}</text>) -> <text style="">tens9{1,16,256,256}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="background-color:#b28c00;color:white;font-weight:bold">&nbsp;(!)&nbsp;Max&nbsp;diff&nbsp;0.042823&nbsp;</text> <br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:#b28c00">&nbsp;└·</text> <text style="color:#b28c00;font-weight:bold">hardsigmoid[torch.nn.functional]</text>(<text style="">tens8{1,16,256,256}</text>, False) -> <text style="">tens9{1,16,256,256}</text><br>
<text style="color:#b28c00">&nbsp;├·</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens9{1,16,256,256}</text>) -> <text style="">tens12{1,32,256,256}</text><br>
<text style="color:#b28c00">&nbsp;└&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens9{1,16,256,256}</text>, <text style="text-decoration:underline">tens10{32,16,3,3}</text>, <text style="text-decoration:underline">tens11{32}</text>, (1, 1), (1, 1), (1, 1), 1) -> <text style="">tens12{1,32,256,256}</text><br>
</div>
</code>

There must be something wrong with the converter. Take a look at how [pytorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html) and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid) define hard sigmoid. See, the formulae do tot match! 
Good thing Nobuco clearly shows us where the problem originates.

````python
@converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: Tensor, inplace: bool = False):
    def func(input, inplace=False):
        return tf.clip_by_value(input/6 + 1/2, clip_value_min=0, clip_value_max=1)
    return func
````

<code>
<div style="overflow-x:scroll; white-space: nowrap">
<font style="font-family: monospace">
<text style="color:green">MyModule[__main__]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens12{1,32,256,256}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">BatchNorm2d[torch.nn.modules.batchnorm]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">&nbsp;├·</text> <text style="">batch_norm[torch.nn.functional]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="">tens1{3}</text>, <text style="">tens2{3}</text>, <text style="">tens3{3}</text>, <text style="">tens4{3}</text>, False, 0.1, 1e-05) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">&nbsp;└&nbsp;</text> <text style="">&nbsp;└·</text> <text style="">batch_norm[torch]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="">tens3{3}</text>, <text style="">tens4{3}</text>, <text style="">tens1{3}</text>, <text style="">tens2{3}</text>, False, 0.1, 1e-05, True) -> <text style="">tens5{1,3,256,256}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens5{1,3,256,256}</text>) -> <text style="">tens8{1,16,256,256}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens5{1,3,256,256}</text>, <text style="text-decoration:underline">tens6{16,3,3,3}</text>, <text style="text-decoration:underline">tens7{16}</text>, (1, 1), (1, 1), (1, 1), 1) -> <text style="">tens8{1,16,256,256}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">Hardsigmoid[torch.nn.modules.activation]</text>(<text style="">tens8{1,16,256,256}</text>) -> <text style="">tens9{1,16,256,256}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">hardsigmoid[torch.nn.functional]</text>(<text style="">tens8{1,16,256,256}</text>, False) -> <text style="">tens9{1,16,256,256}</text><br>
<text style="color:green">&nbsp;├·</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens9{1,16,256,256}</text>) -> <text style="">tens12{1,32,256,256}</text><br>
<text style="color:green">&nbsp;└&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens9{1,16,256,256}</text>, <text style="text-decoration:underline">tens10{32,16,3,3}</text>, <text style="text-decoration:underline">tens11{32}</text>, (1, 1), (1, 1), (1, 1), 1) -> <text style="">tens12{1,32,256,256}</text><br>
</font>
</div>
</code>