#!/bin/python3.11

## note
# 

# 单输入单输出高阶导数
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms


class Net(nn.Cell):
    """前向网络模型"""

    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()

    def construct(self, x):
        out = self.sin(x)
        return out


x_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = ms.grad(net)
secondgrad = ms.grad(firstgrad)
output = secondgrad(x_train)

# 打印结果
result = np.around(output.asnumpy(), decimals=2)
print(result)
