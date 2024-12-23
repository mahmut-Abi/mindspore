#!/bin/python3.11
from download import download

## note
# 下载 cifar 数据集，大约170M
# 下好了自己解压

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz"

path = download(url, "./", kind="tar.gz", replace=True)