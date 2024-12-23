#!/bin/python3.11

## note
# 
# 多输入多输出高阶导数

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

    def construct(self, x, y):
        out1 = self.sin(x) - self.cos(y)
        out2 = self.cos(x) - self.sin(y)
        return out1, out2


x_train = Tensor(np.array([3.1415926]), dtype=ms.float32)
y_train = Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = ms.grad(net, grad_position=(0, 1))
secondgrad = ms.grad(firstgrad, grad_position=(0, 1))
output = secondgrad(x_train, y_train)

# 打印结果
print(np.around(output[0].asnumpy(), decimals=2))
print(np.around(output[1].asnumpy(), decimals=2))
