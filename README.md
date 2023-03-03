# Nobuco

---

**No** **Bu**llshit **Co**nverter is a tool that lets you translate pytorch models into tensorflow graphs without losing your mind.

- Supports a wide range of architectures
  - [x] Control flow ops (If, While)
  - [x] Recurrent layers (LSTM, GRU)
  - [x] Arbitrary torch functions (except for slice assign, as there's no equivalent in tensorflow)
- Simple
- Easily extensible
- Sanity-preserving, with clear mistake messaging

### Essentials

---

Suppose we want to convert a pytorch module similar to this one:

````python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = nn.Hardsigmoid()(x)
        x = 1 - x[:, ::2] * x[:, 1::2]
        return x
````
The process is exactly what you would expect. Instantiate the module, create dummy inputs and call the magic function:

````python
dummy_image = torch.rand(size=(1, 3, 256, 256))
pytorch_module = MyModule().eval()

keras_model = pytorch_to_keras(
    pytorch_module, [dummy_image],
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)
````

Aaaand done! That's all it takes to... wait, what's that?

<code>
<div style="overflow-x:scroll; white-space: nowrap">
<font style="font-family: monospace">
<text style="color:#ce0505">MyModule[__main__]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens8{1,8,128,128}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens3{1,16,128,128}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="text-decoration:underline">tens1{16,3,3,3}</text>, <text style="text-decoration:underline">tens2{16}</text>, (2, 2), (1, 1), (1, 1), 1) -> <text style="">tens3{1,16,128,128}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:#ce0505">Hardsigmoid[torch.nn.modules.activation]</text>(<text style="">tens3{1,16,128,128}</text>) -> <text style="">tens4{1,16,128,128}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:#ce0505">&nbsp;└·</text> <text style="background-color:#ce0505;color:white">hardsigmoid[torch.nn.functional]</text>(<text style="">tens3{1,16,128,128}</text>, False) -> <text style="">tens4{1,16,128,128}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__getitem__[torch.Tensor]</text>(<text style="">tens4{1,16,128,128}</text>, (:, ::2)) -> <text style="">tens5{1,8,128,128}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__getitem__[torch.Tensor]</text>(<text style="">tens4{1,16,128,128}</text>, (:, 1::2)) -> <text style="">tens6{1,8,128,128}</text><br>
<text style="color:#ce0505">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__mul__[torch.Tensor]</text>(<text style="">tens5{1,8,128,128}</text>, <text style="">tens6{1,8,128,128}</text>) -> <text style="">tens7{1,8,128,128}</text><br>
<text style="color:#ce0505">&nbsp;└·</text> <text style="color:green;font-weight:bold">__rsub__[torch.Tensor]</text>(<text style="">tens7{1,8,128,128}</text>, 1) -> <text style="">tens8{1,8,128,128}</text><br>
</font>
</div>
</code>

It says it doesn't know how to handle hard sigmoid.
Apparently, it's our job to provide a node converter for either `F.hardsigmoid` or the enclosing `Hardsigmoid` module (or the entire `MyModule`, but that makes little sense). Here, we'll go for the former.

Conversion is done directly. No layers upon layers of abstraction, no obscure intermediate representation. A node converter is just a `Callable` that takes in the same arguments as the corresponding node and outputs an equivalent node in tensorflow. The converted node preserves the original node's signature, but pytorch tensors replaced with tensorflow counterparts (be that `tf.Tensor`, `KerasTensor`, `tf.Variable`, or `ResourceVariable`).

This should do the trick:

````python
@converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: torch.Tensor, inplace: bool = False):
    return lambda input, inplace=False: tf.keras.activations.hard_sigmoid(input)
````

<code>
<div style="overflow-x:scroll; white-space: nowrap">
<font style="font-family: monospace">
<text style="background-color:#b28c00;color:white">&nbsp;(!)&nbsp;Max&nbsp;diff&nbsp;0.026256&nbsp;</text> <br>
<text style="color:#b28c00">MyModule[__main__]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens8{1,8,128,128}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens3{1,16,128,128}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="text-decoration:underline">tens1{16,3,3,3}</text>, <text style="text-decoration:underline">tens2{16}</text>, (2, 2), (1, 1), (1, 1), 1) -> <text style="">tens3{1,16,128,128}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="background-color:#b28c00;color:white">&nbsp;(!)&nbsp;Max&nbsp;diff&nbsp;0.040743&nbsp;</text> <br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:#b28c00">Hardsigmoid[torch.nn.modules.activation]</text>(<text style="">tens3{1,16,128,128}</text>) -> <text style="">tens4{1,16,128,128}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="background-color:#b28c00;color:white;font-weight:bold">&nbsp;(!)&nbsp;Max&nbsp;diff&nbsp;0.040743&nbsp;</text> <br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:#b28c00">&nbsp;└·</text> <text style="color:#b28c00;font-weight:bold">hardsigmoid[torch.nn.functional]</text>(<text style="">tens3{1,16,128,128}</text>, False) -> <text style="">tens4{1,16,128,128}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__getitem__[torch.Tensor]</text>(<text style="">tens4{1,16,128,128}</text>, (:, ::2)) -> <text style="">tens5{1,8,128,128}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__getitem__[torch.Tensor]</text>(<text style="">tens4{1,16,128,128}</text>, (:, 1::2)) -> <text style="">tens6{1,8,128,128}</text><br>
<text style="color:#b28c00">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__mul__[torch.Tensor]</text>(<text style="">tens5{1,8,128,128}</text>, <text style="">tens6{1,8,128,128}</text>) -> <text style="">tens7{1,8,128,128}</text><br>
<text style="color:#b28c00">&nbsp;└·</text> <text style="color:green;font-weight:bold">__rsub__[torch.Tensor]</text>(<text style="">tens7{1,8,128,128}</text>, 1) -> <text style="">tens8{1,8,128,128}</text><br>
</font>
</div>
</code>

It works, but the outputs don't quite match. Perhaps we should check on how [pytorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html) and [tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid) define hard sigmoid. 
And sure enough, their implementations differ. Have to type in the formula manually I guess...


````python
@converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: torch.Tensor, inplace: bool = False):
    return lambda input, inplace=False: tf.clip_by_value(input/6 + 1/2, clip_value_min=0, clip_value_max=1)
````

<code>
<div style="overflow-x:scroll; white-space: nowrap">
<font style="font-family: monospace">
<text style="color:green">MyModule[__main__]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens8{1,8,128,128}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">Conv2d[torch.nn.modules.conv]</text>(<text style="">tens0{1,3,256,256}</text>) -> <text style="">tens3{1,16,128,128}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">conv2d[torch.nn.functional]</text>(<text style="">tens0{1,3,256,256}</text>, <text style="text-decoration:underline">tens1{16,3,3,3}</text>, <text style="text-decoration:underline">tens2{16}</text>, (2, 2), (1, 1), (1, 1), 1) -> <text style="">tens3{1,16,128,128}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">Hardsigmoid[torch.nn.modules.activation]</text>(<text style="">tens3{1,16,128,128}</text>) -> <text style="">tens4{1,16,128,128}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">hardsigmoid[torch.nn.functional]</text>(<text style="">tens3{1,16,128,128}</text>, False) -> <text style="">tens4{1,16,128,128}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__getitem__[torch.Tensor]</text>(<text style="">tens4{1,16,128,128}</text>, (:, ::2)) -> <text style="">tens5{1,8,128,128}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__getitem__[torch.Tensor]</text>(<text style="">tens4{1,16,128,128}</text>, (:, 1::2)) -> <text style="">tens6{1,8,128,128}</text><br>
<text style="color:green">&nbsp;│&nbsp;</text> <text style="color:green;font-weight:bold">__mul__[torch.Tensor]</text>(<text style="">tens5{1,8,128,128}</text>, <text style="">tens6{1,8,128,128}</text>) -> <text style="">tens7{1,8,128,128}</text><br>
<text style="color:green">&nbsp;└·</text> <text style="color:green;font-weight:bold">__rsub__[torch.Tensor]</text>(<text style="">tens7{1,8,128,128}</text>, 1) -> <text style="">tens8{1,8,128,128}</text><br>
</font>
</div>
</code>

And the happy result:

<div style="overflow-y:scroll; white-space:nowrap; height:500px; width:300px">
<img src="docs/tutorial.png">
</div>

The example above is artificial but it illustrates the point.
It's not feasible to provide a node converter for every existing pytorch op. There's literally [thousands](https://dev-discuss.pytorch.org/t/where-do-the-2000-pytorch-operators-come-from-more-than-you-wanted-to-know/) of them! 
Best we can do without the converter constantly lacking essential functionality, being riddled with bugs, doing weird stuff and breaking apart with every other PT/TF release 
is to keep the system simple and customizable, make it clear where a problem comes from and let the _user_ sort things out.
Usually it's easy for a human to translate an isolated operation from one framework to another.
Reproducing the graph structure is a different matter entirely. Good thing Nobuco has you covered.

### Channel ordering

---
So we've seen that `channel_ordering_strategy` argument, what does it mean, exactly? 
You know the drill, 

- `FORCE_TENSORFLOW_ORDER`
  - Input tensors will be coerced to tensorflow channel order.
  - Convenient for converting channel-aware operations (convolution, batchnorm).
- `FORCE_PYTORCH_ORDER`
  - Input tensors entering the node will look exactly as they do in the original pytorch graph. 
  - Use it when the node does not interpret its input tensors as having a channel dimension (matmul). 
- `MINIMUM_TRANSPOSITIONS`
  - The channel order is decided by a majority vote (whichever prevails among the inputs). This way the number of coercions (i.e. tensor transpositions) is kept to the minimum.
  It also means whenever there's only one input, it will be left untouched.
  - Best choice for element-wise ops (most of activations).
- `MANUAL`
  - You are on your own. In exchange for unrestricted freedom, you take responsibility to coerce input tensors to suitable channel order and to annotate output tensors with their order.


`force_tensorflow_order` and `force_pytorch_order`. 

In case you are curious, the implementation is trivial:

````python
@Tracer.traceable()
def force_tensorflow_order(inputs):
    return inputs

@converter(force_tensorflow_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def force_tensorflow_order(inputs):
    return lambda inputs: inputs
````

`force_pytorch_order` is defined analogously.

### :construction: Advanced usage :construction:

---

This section is under construction, see examples.