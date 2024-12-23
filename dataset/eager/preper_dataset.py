#!/bin/python3.11

## note
# 以下示例代码将图片数据下载到指定位置。

import os
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/banana.jpg"
download(url, './banana.jpg', replace=True)
