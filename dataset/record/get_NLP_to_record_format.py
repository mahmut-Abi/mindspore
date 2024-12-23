#!/bin/python3.11

## note
# 通过MindDataset接口读取MindSpore Record文件格式。

from mindspore.dataset import MindDataset

file_name = "test_text.mindrecord"

# 读取MindSpore Record文件格式
data_set = MindDataset(dataset_files=file_name, shuffle=False)

# 样本计数
print("Got {} samples".format(data_set.get_dataset_size()))

# 打印部分数据
count = 0
for item in data_set.create_dict_iterator(output_numpy=True):
    print("source_sos_ids:", item["source_sos_ids"])
    count += 1
    if count == 10:
        break
