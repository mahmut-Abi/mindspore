#!/bin/python3.11

## note

import mindspore
from mindspore import Tensor, ops
import mindspore.numpy as mnp

# 手动向量化
# 首先，我们先构造一个简单的卷积函数，适用于一维向量场景：
x = mnp.arange(5).astype('float32')
w = mnp.array([1., 2., 3.])

def convolve(x, w):
    output = []
    for i in range(1, len(x) - 1):
        output.append(mnp.dot(x[i - 1 : i + 2], w))
    return mnp.stack(output)

output = convolve(x, w)
print("convolve:", output)

# 当我们期望该函数运用于计算一批一维的卷积运算时，我们很自然地会想到调用for循环进行批处理：
x_batch = mnp.stack([x, x, x])
w_batch = mnp.stack([w, w, w])

def manually_batch_conv(x_batch, w_batch):
    output = []
    for i in range(x_batch.shape[0]):
        output.append(convolve(x_batch[i], w_batch[i]))
    return mnp.stack(output)

output = manually_batch_conv(x_batch, w_batch)
print("manually_batch_conv:\n",output)

# 很显然，通过这种实现方式我们能够得到正确的计算结果，但效率并不高。 当然，您也可以通过自己手动重写函数实现更高效率的向量化计算逻辑，但这将涉及对数据的索引、轴等信息的处理。
def manually_vectorization_conv(x_batch, w_batch):
    output = []
    for i in range(1, x_batch.shape[-1] - 1):
        output.append(mnp.sum(x_batch[:, i - 1 : i + 2] * w_batch, axis=1))
    return mnp.stack(output, axis=1)

output = manually_vectorization_conv(x_batch, w_batch)
print("manually_vectorization_conv:\n",output)