# Nobuco

**No** **Bu**llshit **Co**nverter is a tool that lets you translate pytorch models into tensorflow graphs without losing your mind.


### How?
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
Nobuco traces

````
MyModule[__main__](tens0{1,3,256,256}) -> tens12{1,32,256,256}
 │  BatchNorm2d[torch.nn.modules.batchnorm](tens0{1,3,256,256}) -> tens5{1,3,256,256}
 │   ├· batch_norm[torch.nn.functional](tens0{1,3,256,256}, tens1{3}, tens2{3}, tens3{3}, tens4{3}, False, 0.1, 1e-05) -> tens5{1,3,256,256}
 │   └   └· batch_norm[torch](tens0{1,3,256,256}, tens3{3}, tens4{3}, tens1{3}, tens2{3}, False, 0.1, 1e-05, True) -> tens5{1,3,256,256}
 │  Conv2d[torch.nn.modules.conv](tens5{1,3,256,256}) -> tens8{1,16,256,256}
 │   └· conv2d[torch.nn.functional](tens5{1,3,256,256}, tens6{16,3,3,3}, tens7{16}, (1, 1), (1, 1), (1, 1), 1) -> tens8{1,16,256,256}
 │  Hardsigmoid[torch.nn.modules.activation](tens8{1,16,256,256}) -> tens9{1,16,256,256}
 │   └· hardsigmoid[torch.nn.functional](tens8{1,16,256,256}, False) -> tens9{1,16,256,256}
 ├· Conv2d[torch.nn.modules.conv](tens9{1,16,256,256}) -> tens12{1,32,256,256}
 └   └· conv2d[torch.nn.functional](tens9{1,16,256,256}, tens10{32,16,3,3}, tens11{32}, (1, 1), (1, 1), (1, 1), 1) -> tens12{1,32,256,256}
````



Now, for the second part. 

-- Conversion is done directly. No layers upon layers of abstraction, no obscure intermediate representation, no bullshit.
