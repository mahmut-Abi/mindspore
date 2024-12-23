#!/bin/python3.11

## note
# MindSpore框架提供了丰富的Functional接口，这些接口定义在 mindspore.ops 下，以函数的形式直接定义了操作或计算过程，无需显式创建算子类实例。
# Functional接口提供了包括神经网络层函数、数学运算函数、Tensor操作函数、Parameter操作函数、微分函数、调试函数等类型的接口，
# 这些接口可以直接在 Cell 的 construct 方法中使用，也可以作为独立的操作在数据处理或模型训练中使用。
# MindSpore在 Cell 里使用Functional接口的流程如下所示： 

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class MyCell(nn.Cell):
    def construct(self, x, y):
        output = ops.add(x, y)
        return output

net = MyCell()
x = ms.Tensor([1, 2, 3], ms.float32)
y = ms.Tensor([4, 5, 6], ms.float32)
output = net(x, y)
print(output)