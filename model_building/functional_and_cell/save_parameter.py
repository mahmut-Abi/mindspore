#!/bin/python3.11

## note
# MindSpore提供了load_checkpoint和save_checkpoint方法用来Parameter的保存和加载，需要注意的是Parameter保存时，
# 保存的是Parameter列表，Parameter加载时对象必须是Cell。 在Parameter加载时，可能Parameter名对不上，
# 需要做一些修改，可以直接构造一个新的Parameter列表给到load_checkpoint加载到Cell。

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn

net = nn.Dense(2, 1, has_bias=True)
for param in net.get_parameters():
    print(param.name, param.data.asnumpy())

ms.save_checkpoint(net, "dense.ckpt")
dense_params = ms.load_checkpoint("dense.ckpt")
print(dense_params)
new_params = {}
for param_name in dense_params:
    print(param_name, dense_params[param_name].data.asnumpy())
    new_params[param_name] = ms.Parameter(ops.ones_like(dense_params[param_name].data), name=param_name)

ms.load_param_into_net(net, new_params)
for param in net.get_parameters():
    print(param.name, param.data.asnumpy())


