#!/bin/python3.11

## note
# 当聊到回调Callback的时候，大部分用户都会觉得很难理解，是不是需要堆栈或者特殊的调度方式，实际上我们简单的理解回调：
# 假设函数A有一个参数，这个参数是个函数B，当函数A执行完以后执行函数B，那么这个过程就叫回调。
# Callback是回调的意思，MindSpore中的回调函数实际上不是一个函数而是一个类，用户可以使用回调机制来观察训练过程中网络内部的状态和相关信息，或在特定时期执行特定动作。
# 例如监控损失函数Loss、保存模型参数ckpt、动态调整参数lr、提前终止训练任务等。下面我们继续以手写体识别模型为例，介绍常见的内置回调函数和自定义回调函数。

import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore.train import Model

# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)


def datapipe(path, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

train_dataset = datapipe('MNIST_Data/train', 64)
test_dataset = datapipe('MNIST_Data/test', 64)

trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})


