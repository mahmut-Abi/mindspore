#!/bin/python3.11

## note
# 

import mindspore as ms
from mindspore import Tensor, Parameter, ops

# ms.jit_class为MindSpore即时编译修饰器，可以将普通的Python类作为可编译计算图使用。
@ms.jit_class
class Accumulator():
    def __init__(self, optimizer, accumulate_step, clip_norm=1.0):
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = Parameter(Tensor(1, ms.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step
        self.map = ops.HyperMap()

    def __call__(self, grads):
        # 将单步获得的梯度累加至Accumulator的inner_grads
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        if self.counter % self.accumulate_step == 0:
            # 如果达到累加步数，进行参数优化更新
            self.optimizer(self.inner_grads)
            # 完成参数优化更新后，清零inner_grads
            self.map(ops.partial(ops.assign), self.inner_grads, self.zeros)
        # 计算步数加一
        ops.assign_add(self.counter, Tensor(1, ms.int32))

        return True

# 使用快速入门中手写数字识别模型验证梯度累加的效果。
from mindspore import nn
from mindspore import value_and_grad
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset

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
    label_transform = transforms.TypeCast(ms.int32)

    dataset = MnistDataset(path)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

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


accumulate_step = 2

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)
accumulator = Accumulator(optimizer, accumulate_step)

def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    # loss除以累加步数accumulate_step
    return loss / accumulate_step


grad_fn = value_and_grad(forward_fn, None, model.trainable_params())

@ms.jit
def train_step(data, label):
    loss, grads = grad_fn(data, label)
    accumulator(grads)
    return loss


def train_loop(model, dataset, loss_fn, optimizer):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
            

def test_loop(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
train_dataset = datapipe('MNIST_Data/train', 32)
test_dataset = datapipe('MNIST_Data/test', 32)

epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, train_dataset, loss_fn, optimizer)
    test_loop(model, test_dataset, loss_fn)
print("Done!")