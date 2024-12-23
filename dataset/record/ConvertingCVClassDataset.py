#!/bin/python3.11

## note
# 本示例主要以包含100条记录的CV数据集并将其转换为MindSpore Record格式为例子，介绍如何将CV类数据集转换成MindSpore Record文件格式，并使用MindDataset接口读取。
# 首先，需要创建100张图片的数据集并对齐进行保存，其样本包含file_name（字符串）、label（整型）、 data（二进制）三个字段，然后使用MindDataset接口读取该MindSpore Record文件。
# 生成100张图像，并转换成MindSpore Record文件格式。
# 下面示例运行无报错说明数据集转换成功。

from PIL import Image
from io import BytesIO
from mindspore.mindrecord import FileWriter

file_name = "test_vision.mindrecord"
# 定义包含的字段
cv_schema = {"file_name": {"type": "string"},
             "label": {"type": "int32"},
             "data": {"type": "bytes"}}

# 声明MindSpore Record文件格式
writer = FileWriter(file_name, shard_num=1, overwrite=True)
writer.add_schema(cv_schema, "it is a cv dataset")
writer.add_index(["file_name", "label"])

# 创建数据集
data = []
for i in range(100):
    sample = {}
    white_io = BytesIO()
    Image.new('RGB', ((i+1)*10, (i+1)*10), (255, 255, 255)).save(white_io, 'JPEG')
    image_bytes = white_io.getvalue()
    sample['file_name'] = str(i+1) + ".jpg"
    sample['label'] = i+1
    sample['data'] = white_io.getvalue()

    data.append(sample)
    if i % 10 == 0:
        writer.write_raw_data(data)
        data = []

if data:
    writer.write_raw_data(data)

writer.commit()
