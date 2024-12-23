#!/bin/python3.11

## note
# 此示例将使用transforms模块中通用Transform，对给定数据进行变换。
# 通用Transform的Eager模式支持numpy.array类型的数据作为入参。

import numpy as np
import mindspore.dataset.transforms as trans

# Apply Fill to input immediately
data = np.array([1, 2, 3, 4, 5])
fill = trans.Fill(0)
data = fill(data)
print("Fill result: ", data)

# Apply OneHot to input immediately
label = np.array(2)
onehot = trans.OneHot(num_classes=5)
label = onehot(label)
print("OneHot result: ", label)
