#!/bin/python3.11

## note
# 
# 单输入多输出高阶导数

import numpy as np
from mindspore import ops, Tensor
import mindspore.nn as nn
import mindspore as ms


class Net(nn.Cell):
    """前向网络模型"""

    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x):
        out1 = self.sin(x)
        out2 = self.cos(x)
        return out1, out2


x_train = Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = ms.grad(net)
secondgrad = ms.grad(firstgrad)
output = secondgrad(x_train)

# 打印结果
result = np.around(output.asnumpy(), decimals=2)
print(result)
