#!/bin/python3.11

## note
# 

import numpy as np
from mindspore import ops, Tensor
import mindspore.nn as nn
import mindspore as ms

# 首先定义网络模型Net、输入x和输入y
# 定义输入x和y
x = Tensor([3.0], dtype=ms.float32)
y = Tensor([5.0], dtype=ms.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        out = x * x * y * self.z
        return out

# 对输入x, y进行求导，需要将grad_position设置成(0, 1)：
net = Net()
grad_fn = ms.grad(net, grad_position=(0, 1))
gradients = grad_fn(x, y)
print(gradients)

# 对权重进行求导
# 对权重z进行求导，这里不需要对输入求导，将grad_position设置成None：
params = ms.ParameterTuple(net.trainable_params())

output = ms.grad(net, grad_position=None, weights=params)(x, y)
print(output)


net = nn.Dense(10, 1)
loss_fn = nn.MSELoss()


def forward(inputs, labels):
    logits = net(inputs)
    loss = loss_fn(logits, labels)
    return loss, logits

# 返回辅助变量
# 同时对输入和权重求导，其中只有第一个输出参与求导，示例代码如下：
inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
labels = Tensor(np.random.randn(16, 1).astype(np.float32))
weights = net.trainable_params()

# Aux value does not contribute to the gradient.
grad_fn = ms.grad(forward, grad_position=0, weights=None, has_aux=True)
inputs_gradient, (aux_logits,) = grad_fn(inputs, labels)
print(len(inputs_gradient), aux_logits.shape)


# 停止计算梯度
class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        out1 = x * y
        out2 = x * y
        out2 = ops.stop_gradient(out2)  # 停止计算out2算子的梯度
        out = out1 + out2
        return out


net = Net()
grad_fn = ms.grad(net)
output = grad_fn(x, y)
print(output)

