#!/bin/python3.11

## note
# 通常情况下，定义训练和评估网络并直接运行，已经可以满足基本需求。
# 一方面，Model可以在一定程度上简化代码。例如：无需手动遍历数据集；在不需要自定义nn.TrainOneStepCell的场景下，可以借助Model自动构建训练网络；
# 可以使用Model的eval接口进行模型评估，直接输出评估结果，无需手动调用评价指标的clear、update、eval函数等。
# 另一方面，Model提供了很多高阶功能，如数据下沉、混合精度等，在不借助Model的情况下，使用这些功能需要花费较多的时间仿照Model进行自定义。
# 本文档首先对MindSpore的Model进行基本介绍，然后重点讲解如何使用Model进行模型训练、评估和推理。

# Model基本介绍
# # Model是MindSpore提供的高阶API，可以进行模型训练、评估和推理。其接口的常用参数如下：
# # network：用于训练或推理的神经网络。
# # loss_fn：所使用的损失函数。
# # optimizer：所使用的优化器。
# # metrics：用于模型评估的评价函数。
# # eval_network：模型评估所使用的网络，未定义情况下，Model会使用network和loss_fn进行封装。
# # Model提供了以下接口用于模型训练、评估和推理：
# # fit：边训练边评估模型。
# # train：用于在训练集上进行模型训练。
# # eval：用于在验证集上进行模型评估。
# # predict：用于对输入的一组数据进行推理，输出预测结果。

import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
context.set_context(runtime_num_threads=32)  # 设置并行工作的数量（即使用的CPU核心数）
context.set_context(max_device_memory='256GB') # 设置设备可用的最大内存。格式为"xxGB"。默认值： 1024GB
context.set_context(mempool_block_size='256GB') # 关闭虚拟内存下生效，设置设备内存池的块大小。格式为"xxGB"。默认值： 1GB
context.set_context(memory_offload='OFF') # 是否开启Offload功能，在内存不足场景下将空闲数据临时拷贝至Host侧内存。


# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

# 下载并处理数据集
# 使用download库下载数据集，通过 vison.Rescale 接口对图片进行缩放， vision.Normalize 接口对输入图片进行归一化处理， vision.HWC2CHW 接口对数据格式进行转换。
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

train_dataset = datapipe('MNIST_Data/train', 64)
test_dataset = datapipe('MNIST_Data/test', 64)

# 
# Define model
# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),  # 输入大小为 28x28 的图像被展平成 784 维向量
            nn.ReLU(),             # 激活函数
            nn.Dense(512, 512),    # 第二个全连接层
            nn.ReLU(),             # 激活函数
            nn.Dense(512, 10)      # 输出层，假设是针对 10 类分类问题
        )

    def construct(self, x):
        x = self.flatten(x)                    # 展平输入数据
        logits = self.dense_relu_sequential(x) # 通过一系列全连接层和激活函数
        return logits                          # 返回未归一化的预测值 (logits)

model = Network()  # 实例化模型

# 定义损失函数和优化器
# 要训练神经网络模型，需要定义损失函数和优化器函数。
# 损失函数这里使用交叉熵损失函数CrossEntropyLoss。
# 优化器这里使用SGD。
# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)


# 训练及保存模型
# 在开始训练之前，MindSpore需要提前声明网络模型在训练过程中是否需要保存中间过程和结果，.
# 因此使用ModelCheckpoint接口用于保存网络模型和参数，以便进行后续的Fine-tuning（微调）操作。

steps_per_epoch = train_dataset.get_dataset_size()
config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)

ckpt_callback = ModelCheckpoint(prefix="mnist", directory="./checkpoint", config=config)
loss_callback = LossMonitor(steps_per_epoch)

# 通过MindSpore提供的model.fit接口可以方便地进行网络的训练与评估，LossMonitor可以监控训练过程中loss值的变化。
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})

trainer.fit(10, train_dataset, test_dataset, callbacks=[ckpt_callback, loss_callback])

# 训练过程中会打印loss值，loss值会波动，但总体来说loss值会逐步减小，精度逐步提高。每个人运行的loss值有一定随机性，不一定完全相同。
# 通过模型运行测试数据集得到的结果，验证模型的泛化能力：
# 使用model.eval接口读入测试数据集。
# 使用保存后的模型参数进行推理。
acc = trainer.eval(test_dataset)
acc
