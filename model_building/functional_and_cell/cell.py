#!/bin/python3.11

## note
# MindSpore框架中的核心构成单元 mindspore.nn.Cell 是构建神经网络的基本模块，负责定义网络的计算逻辑。
# Cell 不仅支持动态图（PyNative模式）下作为网络的基础组件，也能够在静态图（GRAPH模式）下被编译成高效的计算图执行。
# Cell 通过其 construct 方法定义了前向传播的计算过程，并可通过继承机制扩展功能，实现自定义的网络层或复杂结构。
# 通过 set_train 方法，Cell 能够灵活地在训练与推理模式间切换，以适应不同算子在两种模式下的行为差异。
# 此外，Cell 还提供了丰富的API，如混合精度、参数管理、梯度设置、Hook功能、重计算等，以支持模型的优化与训练。
# MindSpore 基本的 Cell 搭建过程如下所示：

import mindspore.nn as nn
import mindspore.ops as ops

class MyCell(nn.Cell):
    def __init__(self, forward_net):
        super(MyCell, self).__init__(auto_prefix=True)
        self.net = forward_net
        self.relu = ops.ReLU()

    def construct(self, x):
        y = self.net(x)
        return self.relu(y)

inner_net = nn.Conv2d(120, 240, 4, has_bias=False)
my_net = MyCell(inner_net)
print(my_net.trainable_params())