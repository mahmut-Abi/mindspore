#!/bin/python3.11

## note
# 本示例首先创建一个包含100条记录的MindSpore Record文件格式，其样本包含八个字段，均为整型数组，然后使用MindDataset接口读取该MindSpore Record文件。
# 生成100条文本数据，并转换成MindSpore Record文件格式。

import numpy as np
from mindspore.mindrecord import FileWriter

# 输出的MindSpore Record文件完整路径
file_name = "test_text.mindrecord"

# 定义样本数据包含的字段
nlp_schema = {"source_sos_ids": {"type": "int64", "shape": [-1]},
              "source_sos_mask": {"type": "int64", "shape": [-1]},
              "source_eos_ids": {"type": "int64", "shape": [-1]},
              "source_eos_mask": {"type": "int64", "shape": [-1]},
              "target_sos_ids": {"type": "int64", "shape": [-1]},
              "target_sos_mask": {"type": "int64", "shape": [-1]},
              "target_eos_ids": {"type": "int64", "shape": [-1]},
              "target_eos_mask": {"type": "int64", "shape": [-1]}}

# 声明MindSpore Record文件格式
writer = FileWriter(file_name, shard_num=1, overwrite=True)
writer.add_schema(nlp_schema, "Preprocessed nlp dataset.")

# 创建虚拟数据集
data = []
for i in range(100):
    sample = {"source_sos_ids": np.array([i, i + 1, i + 2, i + 3, i + 4], dtype=np.int64),
              "source_sos_mask": np.array([i * 1, i * 2, i * 3, i * 4, i * 5, i * 6, i * 7], dtype=np.int64),
              "source_eos_ids": np.array([i + 5, i + 6, i + 7, i + 8, i + 9, i + 10], dtype=np.int64),
              "source_eos_mask": np.array([19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int64),
              "target_sos_ids": np.array([28, 29, 30, 31, 32], dtype=np.int64),
              "target_sos_mask": np.array([33, 34, 35, 36, 37, 38], dtype=np.int64),
              "target_eos_ids": np.array([39, 40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64),
              "target_eos_mask": np.array([48, 49, 50, 51], dtype=np.int64)}
    data.append(sample)

    if i % 10 == 0:
        writer.write_raw_data(data)
        data = []

if data:
    writer.write_raw_data(data)

writer.commit()
