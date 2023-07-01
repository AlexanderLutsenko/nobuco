<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/nobuco.png">
<sup><a href="https://www.behance.net/diliajl">diliajl</a></sup>
</p>

**No** **Bu**llshit **Co**nverter is a tool that helps you translate pytorch models into tensorflow graphs without losing your mind.

- Supports a wide range of architectures
  - [x] Control flow ops (If, While)
  - [x] Recurrent layers (LSTM, GRU)
  - [x] Arbitrary torch functions
- Simple
- Flexible
- Efficient
- Sanity-preserving, with clear mistake messaging

<!-- toc -->

## Installation <img src="https://img.shields.io/pypi/v/nobuco?color=blue&style=flat-square">
<img src="https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch"> <img src="https://img.shields.io/badge/TensorFlow-2.10-FF6F00.svg?style=flat&logo=tensorflow">

```bash
pip install -U nobuco
```

## Table of Contents
- [Essentials](#essentials)
- [Channel order wizardry](#channel-order-wizardry)
- [In-place operations](#in-place-operations)
- [Going dynamic](#going-dynamic)
  - [Control flows](#control-flows)
  - [Dynamic shapes](#dynamic-shapes)
- [So we put a converter inside your converter](#so-we-put-a-converter-inside-your-converter)
- [But was it worth it?](#but-was-it-worth-it)
- [Nobuco knowledge base](#nobuco-knowledge-base)

<!-- tocstop -->

## Essentials

Suppose we want to convert a pytorch module similar to this one:

````python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))

    def forward(self, x):
        x = self.conv(x)
        x = nn.Hardsigmoid()(x)
        x = 1 - x[:, ::2] * x[:, 1::2]
        return x
````

The process is exactly what you would expect. Instantiate the module, create dummy inputs and call the magic function:

```python
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
```

````python
dummy_image = torch.rand(size=(1, 3, 256, 256))
pytorch_module = MyModule().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[dummy_image], kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW
)
````

Aaaand done! That's all it takes to... wait, what's that?

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/essentials1.svg" width="100%">

Nobuco says it doesn't know how to handle hard sigmoid.
Apparently, it's our job to provide a node converter for either `F.hardsigmoid` or the enclosing `Hardsigmoid` module (or the entire `MyModule`, but that makes little sense). Here, we'll go for the former.

Conversion is done directly. No layers upon layers of abstraction, no obscure intermediate representation. 
A node converter is just a `Callable` that takes in the same arguments as the corresponding node in pytorch and outputs an equivalent node in tensorflow. 
The converted node preserves the original node's signature, but pytorch tensors replaced with tensorflow counterparts (be that `tf.Tensor`, `KerasTensor`, `tf.Variable`, or `ResourceVariable`).

This should do the trick:

````python
@nobuco.converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: torch.Tensor, inplace: bool = False):
    return lambda input, inplace=False: tf.keras.activations.hard_sigmoid(input)
````

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/essentials2.svg" width="100%">

It works, but the outputs don't quite match. Perhaps we should check on how [pytorch](https://pytorch.org/https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/stable/generated/torch.nn.functional.hardsigmoid.html) and [tensorflow](https://www.tensorflow.org/api_https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/python/tf/keras/activations/hard_sigmoid) define hard sigmoid. 
And sure enough, their implementations differ. Have to type in the formula manually, I guess...

````python
@nobuco.converter(F.hardsigmoid, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def hardsigmoid(input: torch.Tensor, inplace: bool = False):
    return lambda input, inplace=False: tf.clip_by_value(input/6 + 1/2, clip_value_min=0, clip_value_max=1)
````

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/essentials3.svg" width="100%">

And the happy result:

<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/tutorial.png" width="30%">
</p>

The example above is artificial but it illustrates the point.
It's not feasible to provide a node converter for every existing pytorch op. There's literally [thousands](https://dev-discuss.pytorch.org/t/where-do-the-2000-pytorch-operators-come-from-more-than-you-wanted-to-know/) of them! 
Best we can do without the converter constantly lacking essential functionality, being riddled with bugs, doing weird stuff and breaking apart with every other PT/TF release 
is to keep the tool simple and customizable, make it clear where a problem comes from and let the _user_ sort things out.
Usually it's easy for a human to translate an isolated operation from one framework to another.
Reproducing the graph structure is a different matter entirely. For that, Nobuco has you covered!

https://user-images.githubusercontent.com/2457934/233740603-cc11acc5-cd6b-48c8-b089-ff3ead772dd0.mp4

<p align="center"><em>
To ease debugging, Nobuco lets you jump right where the node was [I]nvoked, [D]efined and [C]onverted
</em></p>

## Channel order wizardry

Some operations assume its input tensors have a channel dimension. 
And as you probably know, pytorch and tensorflow do not agree on the layout of such tensors.
Pytorch adopts channel-first layout (_B**C**H_, _B**C**HW_, etc.) 
while tensorflow works efficiently with channel-last tensors (_BH**C**_, _BHW**C**_, ...).
Transposing tensors between the two layouts incurs non-trivial overhead as generally, tensor data must be physically rearranged.
In an effort to keep that overhead to the minimum, Nobuco does layout coercions _lazily_. 
A couple of things are needed to make it possible:

- Tensorflow tensors are augmented with an additional property which stores their channel order, either pytorch (channel first) or tensorflow (channel last) style.
- Node converters have requirements on what channel order their inputs must have. Said requirements are expressed with `channel_ordering_strategy` argument. 

Channel ordering strategies are
- `FORCE_TENSORFLOW_ORDER`
  - Input tensors will be coerced to tensorflow channel order.
  - Convenient for converting channel-aware operations (convolution, batchnorm).
- `FORCE_PYTORCH_ORDER`
  - Input tensors entering the node will look exactly as they do in the original pytorch graph. 
  - Use it when the node does not interpret its input tensors as having a channel dimension (linear, matmul). 
- `MINIMUM_TRANSPOSITIONS`
  - The channel order is decided by a majority vote (whichever prevails among the inputs). This way the number of coercions (i.e. tensor transpositions) is kept to the minimum.
  It also means whenever there's only one input, it will be left untouched.
  - Best choice for element-wise ops (most activations).
- `MANUAL`
  - You are on your own. In exchange for unrestricted freedom, you take responsibility to coerce input tensors to suitable channel order and to also annotate output tensors with their order.

The simple lazy approach makes wonders in most situations, but sometimes it produces suboptimal graphs.
Consider the code below. Imagine this is some sort of text processing network. 
It first applies a GRU layer which assumes the inputs do not have a channel dimension, so its input/output layouts are the same in both pytorch and tensorflow.
But then, the outputs are passed to a couple of 1D convolutions which are channel-aware. 
Because of that, a transpose op must be put in the converted graph.

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(32, 128, num_layers=1, batch_first=True, bidirectional=False)
        self.conv1 = nn.Conv1d(12, 40, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(12, 60, kernel_size=1, padding=0)

    def forward(self, x):
        x, hx = self.gru(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return x1, x2

pytorch_module = MyModule().eval()

inputs = [
    torch.normal(0, 1, size=(1, 12, 32)),
]
You can
keras_model = nobuco.pytorch_to_keras(
    pytorch_module, inputs,
    inputs_channel_order=ChannelOrder.PYTORCH,
)
```

The laziness shoots us in the foot here, and we get not one transpose but two:

<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/channel_ordering.png" width="30%">
</p>

For such occasions, there's two brethren functions: `force_tensorflow_order` and `force_pytorch_order`.

```python
x, hx = self.gru(x)
x = nobuco.force_tensorflow_order(x)
x1 = self.conv1(x)
x2 = self.conv2(x)
```

<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/channel_ordering_forced.png" width="30%">
</p>

In case you are curious, the implementation is trivial:

```python
@nobuco.traceable
def force_tensorflow_order(inputs):
    return inputs


@nobuco.converter(force_tensorflow_order, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_force_tensorflow_order(inputs):
    return lambda inputs: inputs
```

`force_pytorch_order` is defined analogously.

## In-place operations

Nobuco can handle most situations where tensors are modified in-place. For instance, these will work just fine:

```python
class MyModule(nn.Module):
    def forward(self, x):
        x[:, 1:2, 16:25, 8::2] *= 2
        torch.relu_(x)
        return x
```

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/inplace1.svg" width="100%">

However, applying in-place operation to a slice yields incorrect result. What gives?

```python
class MyModule(nn.Module):
    def forward(self, x):
        torch.relu_(x[:, 1:2, 16:25, 8::2])
        return x
```

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/inplace2.svg" width="100%">

You see, tensorflow graphs (and many other formats like ONNX) do not support in-place ops.
So when we take slice (`x[:, 1:2, 16:25, 8::2]`) in TF/ONNX, the result is not a view of the original tensor but a copy. 
This copy is then passed to `relu` (which is not in-place either), and its result is not used anywhere. 
As you can see above, the output tensors of `__getitem__` and `relu_` are <span style="color:gray">grayed out</span>, and these operations are excluded from the graph.
In fact, it's empty:

<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/inplace_empty.png" width="30%">
</p>

The easiest way of fixing this is to explicitly assign the result to the slice.
Conveniently enough, most standard in-place operations in pytorch do return their modified arguments as outputs.

```python
class MyModule(nn.Module):
    def forward(self, x):
        x[:, 1:2, 16:25, 8::2] = torch.relu_(x[:, 1:2, 16:25, 8::2])
        return x
```

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/inplace3.svg" width="100%">

## Going dynamic

### Control flows
Introducing python control flow statements into the compute graph is no easy feat.
Tensorflow can do so via `tf.autograph`, but at a cost of [system's complexity](https://www.youtube.com/watch?v=NIEgzljyDyI) and with some notable [limitations](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md).
Stuff like that is way above Nobuco's paygrade, so the following module cannot be properly handled without human intervention.

```python
class ControlIf(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = nn.Conv2d(3, 16, kernel_size=(1, 1))
        self.conv_true = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.conv_false = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.conv_shared = nn.Conv2d(32, 32, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_pre(x)
        if x.mean() > 0:
            x = self.conv_true(x)
            x = torch.tanh(x)
            x = self.conv_shared(x)
            x = x + 1
        else:
            x = self.conv_false(x)
            x = torch.sigmoid(x)
            x = self.conv_shared(x)
            x = x - 1
        x = self.conv_shared(x)
        return x
```

Of course, it's possible to translate the dynamic module into a tensorflow layer
(don't forget to decorate it with `@tf.function` for autograph to kick in).
But what if it contains inner modules, do you replicate them in tensorflow all by hand?
Not unless you want to! 
Just convert them separately and use the resulting graphs inside the parent layer.

```python
class ControlIfKeras(tf.keras.layers.Layer):
    def __init__(self, conv_pre, conv_true, conv_false, conv_shared, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_pre = conv_pre
        self.conv_true = conv_true
        self.conv_false = conv_false
        self.conv_shared = conv_shared

    def get_config(self):
        config = super().get_config()
        config.update({
            "conv_pre": self.conv_pre,
            "conv_true": self.conv_true,
            "conv_false": self.conv_false,
            "conv_shared": self.conv_shared,
        })
        return config

    @tf.function
    def call(self, x):
        x = self.conv_pre(x)
        if tf.reduce_mean(x) > 0:
            x = self.conv_true(x)
            x = tf.tanh(x)
            x = self.conv_shared(x)
            x = x + 1
        else:
            x = self.conv_false(x)
            x = tf.sigmoid(x)
            x = self.conv_shared(x)
            x = x - 1
        x = self.conv_shared(x)
        return x


@nobuco.converter(ControlIf, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
def converter_ControlIf(self, x):
    order = ChannelOrder.TENSORFLOW
    kwargs = {'inputs_channel_order': order, 'outputs_channel_order': order, 'return_outputs_pt': True}
    
    conv_pre, out_pre = nobuco.pytorch_to_keras(self.conv_pre, [x], **kwargs)
    conv_true, out_true = nobuco.pytorch_to_keras(self.conv_true, [out_pre], **kwargs)
    conv_false, out_false = nobuco.pytorch_to_keras(self.conv_false, [out_pre], **kwargs)
    conv_shared, _ = nobuco.pytorch_to_keras(self.conv_shared, [out_true], **kwargs)
    layer = ControlIfKeras(conv_pre, conv_true, conv_false, conv_shared)
    return layer
```

<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/control_if.png" width="25%">
</p>

See [examples](/examples) for other ways to convert control flow ops.

### Dynamic shapes

What if we wanted our module to accept images of arbitrary height and width?
Can we have that? Let's try:

```python
class DynamicShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)

        # Produces static shape
        b, c, h, w = x.shape

        x = x[:, :, h//3:, w//3:]
        return x


input = torch.normal(0, 1, size=(1, 3, 128, 128))
pytorch_module = DynamicShape().eval()

keras_model = nobuco.pytorch_to_keras(
    pytorch_module,
    args=[input],
    input_shapes={input: (None, 3, None, None)}, # Annotate dynamic axes with None
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
)
```

Something's not right. We don't see shape extraction ops in the debug output or the graph:

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/dynamic_shape1.svg" width="100%">

<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/dynamic_shape1.png" width="15%">
</p>

That's not surprising, actually. 
In pytorch, tensor shape is a tuple of regular integers, not tensors, so it's quite difficult to track them.
`nobuco.shape` solves this problem.
This function returns tensors, much like [`tf.shape`](https://www.tensorflow.org/api_https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/python/tf/shape) does:

```python
# Allows for dynamic shape
b, c, h, w = nobuco.shape(x)
```

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/dynamic_shape2.svg" width="100%">

<p align="center">
<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/dynamic_shape2.png" width="30%">
</p>

Also, take a moment to appreciate how elegant the solution is. 
`nobuco.shape` is a one-liner that simply replaces each integer in shape tuple with a scalar tensor.
No special treatment required!

```python
@traceable
def shape(x: Tensor):
    return tuple(torch.tensor(d, dtype=torch.int32) for d in x.shape)


@converter(shape, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_shape(x: Tensor):
    def func(x):
        shape = tf.unstack(tf.shape(x))
        if get_channel_order(x) == ChannelOrder.TENSORFLOW:
            shape = permute_keras2pytorch(shape)
        return tuple(shape)
    return func
```

It's also possible to automatically substitute every `.shape` call with `nobuco.shape` during the tracing phase by setting `trace_shape` flag:

```python
keras_model = nobuco.pytorch_to_keras(
  # ...
  trace_shape=True
)
```

## So we put a converter inside your converter

As we've learned, Nobuco gets confused when in-place operation is applied to a slice.
There's a way to fix that, but let's not do it now.
Instead, we'll use it as an excuse to explain the concept of nested converters.
So, for this module, conversion will give us incorrect result:

```python
class SliceReLU(nn.Module):
    def forward(self, x):
        # Gives incorrect result after conversion
        torch.relu_(x[:, 1:2, 16:25, 8::2])
        # That's the recommended approach, but we're not going for it now
        # x[:, 1:2, 16:25, 8::2] = torch.relu_(x[:, 1:2, 16:25, 8::2])
        return x


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        SliceReLU()(x)
        return x
```

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/converter_inside_converter1.svg" width="100%">

In the previous section, we've seen it's possible to invoke a Nobuco converter inside another Nobuco converter.
Can we embed some third-party converter? You bet! Why? Because it might just do what we need.
Let's consider the standard route: pytorch -> onnx -> tensorflow, with the latter step done with [onnx-tf](https://github.com/onnx/onnx-tensorflow).
This library likes transposing stuff so much, converting the whole graph with it may introduce intolerable inference overhead. Nonetheless, it does the job.
A sensible tradeoff would be to wrap the problematic operation into its own `nn.Module` and give it a special treat, while handling everything else with Nobuco.

```python
import onnx
from onnx_tf.backend import prepare


@nobuco.converter(SliceReLU, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER, reusable=False)
def converter_SliceReLU(self, x):
    model_path = 'slice_relu'
    onnx_path = model_path + '.onnx'

    # NB: onnx.export in implemented via tracing i.e. it may modify the inputs!
    torch.onnx.export(self, (x,), onnx_path, opset_version=12, input_names=['input'],
                      dynamic_axes={'input': [0, 1, 2, 3]}
                      )
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(model_path)
    model = tf.keras.models.load_model(model_path)
    return keras.layers.Lambda(lambda x: model(input=x))
```

<img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/converter_inside_converter2.svg" width="100%">

## But was it worth it?

Let's cut to the chase, here's the numbers.

**mobilenet_v3_large** (26.8 Mb)

|                   | nobuco  | onnx_tf  | speedup |
|-------------------|---------|----------|---------|
| x86 (XNNPACK)     | 11.1 ms | 14.7 ms  | 1.3x    |
| Arm CPU (XNNPACK) | 24.3 ms | 40.3 ms  | 1.6x    |
| Arm GPU (OpenCL)  | 21.3 ms | 192.6 ms | 9x      |

**deeplabv3_resnet50** (158.5 Mb)

|                   | nobuco | onnx_tf | speedup |
|-------------------|--------|---------|---------|
| x86 (XNNPACK)     | 1.25 s | 1.34 s  | 1.07x   |
| Arm CPU (XNNPACK) | 2.0 s  | 2.7 s   | 1.35x   |
| Arm GPU (OpenCL)  | 1.6 s  | 2.6 s   | 1.62x   |

As we can see, redundant transpositions may completely ruin the performance, especially on a GPU.
But that's not the only issue.
Let's test this:

```python
class SliceReLU(nn.Module):
    def forward(self, x):
        x[:, 1:2, 16:25, 8::2] = torch.relu_(x[:, 1:2, 16:25, 8::2])
        return x
```

|                   | nobuco     | onnx_tf    | speedup |
|-------------------|------------|------------|---------|
| x86 (XNNPACK)     | 0.40 ms    | 1.57 ms    | 3.9x    |
| Arm CPU           | 4.6 ms     | **2.9** ms | 0.6x    |
| Arm CPU (XNNPACK) | **2.1** ms | FAIL       | —       |
| Arm GPU (OpenCL)  | 21.8 ms    | FAIL       | —       |

Again, the graph obtained with `onnx_tf` is much slower on x86 CPU.
Worse yet, on mobile processor, optimized TFLite delegates for both GPU and CPU failed.
No transpose ops were added this time, so who's to blame?
It suffices to see what `torch.onnx.export` gives us:

<p align="center">
  <img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/slice_relu_onnx.png" width="100%">
  <b>slice_relu.onnx</b>
</p>

`onnx_tf` does a fair job optimizing the graph it's given,
but combining consecutive `slice` ops seems to be too much to ask.
It also leaves out garbage nodes sometimes (note the free-floating `While` in this example).

Nobuco evades these types of problems by simply not dealing with `onnx`.

<table align="center">
  <tr>
    <th>slice_relu_nobuco</th>
    <th>slice_relu_onnxtf</th>
  </tr>
  <tr>
    <td>
      <p align="center">
        <img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/slice_relu_nobuco.png" width="60%">
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://raw.githubusercontent.com/AlexanderLutsenko/nobuco/master/docs/slice_relu_onnxtf.png" width="60%">
      </p>
    </td>
  </tr>
</table>

## Nobuco knowledge base

Don't want to convert anything but looking for a tensorflow equivalent of a certain pytorch node (operation or module)?
Nobuco already implements quite a few node converters, most written in concise and (hopefully) understandable way.
These are located in [nobuco/node_converters](https://github.com/AlexanderLutsenko/nobuco/tree/master/nobuco/node_converters),
and there's a utility function to help you find what you need:


```python
node = torch.Tensor.repeat
# node = F.relu_
# node = nn.LSTM

location_link, source_code = nobuco.locate_converter(node)
print('Converter location:')
print(location_link)
print('Converter source code:')
print(source_code)
```

```console
Converter location:
File "/home/user/anaconda3/envs/nb/lib/python3.9/site-packages/nobuco/node_converters/tensor_manipulation.py", line 141

Converter source code:
@converter(torch.Tensor.repeat, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def converter_repeat(self, *sizes):
    def func(self, *sizes):
        if get_channel_order(self) == ChannelOrder.TENSORFLOW:
            sizes = permute_pytorch2keras(sizes)
        return tf.tile(self, sizes)
    return func
```

---

### Acknowledgements

Slice assign converter is based on [Zaccharie Ramzi's tf-slice-assign script](https://github.com/zaccharieramzi/tf-slice-assign).
