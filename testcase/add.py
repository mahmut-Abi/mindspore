import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="CPU") # 需指定CPU为后端
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))