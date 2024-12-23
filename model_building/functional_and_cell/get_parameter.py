#!/bin/python3.11

## note
#mindspore.nn.Cell 使用 parameters_dict 、get_parameters 和 trainable_params 接口获取 Cell 中的 Parameter 。
# parameters_dict：获取网络结构中所有Parameter，返回一个以key为Parameter名，value为Parameter值的OrderedDict。
# get_parameters：获取网络结构中的所有Parameter，返回Cell中Parameter的迭代器。
# trainable_params：获取Parameter中requires_grad为True的属性，返回可训练的Parameter的列表。
# 在定义优化器时，使用net.trainable_params()获取需要进行Parameter更新的Parameter列表。

import mindspore.nn as nn

net = nn.Dense(2, 1, has_bias=True)
print(net.trainable_params())

for param in net.trainable_params():
    param_name = param.name
    if "bias" in param_name:
        param.requires_grad = False
print(net.trainable_params())
