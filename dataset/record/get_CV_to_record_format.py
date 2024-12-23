#!/bin/python3.11

## note
# 通过MindDataset接口读取MindSpore Record文件格式。

from mindspore.dataset import MindDataset
from mindspore.dataset.vision import Decode

file_name = "test_vision.mindrecord"

# 读取MindSpore Record文件格式
data_set = MindDataset(dataset_files=file_name)
decode_op = Decode()
data_set = data_set.map(operations=decode_op, input_columns=["data"], num_parallel_workers=2)

# 样本计数
print("Got {} samples".format(data_set.get_dataset_size()))