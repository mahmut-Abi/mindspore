
import mindspore
from mindspore import nn, context
from mindspore.nn import ExponentialDecayLR
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor, SummaryCollector


context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
context.set_context(runtime_num_threads=96)

# 数据集下载
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)


def datapipe(path, batch_size):
    # 定义图像预处理操作
    image_transforms = [
        # 对图像的像素值进行缩放和偏移，以改变其数值范围
        # 第一个参数是缩放因子（scale factor）
        # 第二个参数是偏移量（shift）
        # output = x * scale + shift
        vision.Rescale(1.0 / 255.0, 0),
        # 对图像的像素值进行标准化
        # 减小不同图像之间统计特性（如亮度和对比度）的差异
        # 去均值化（Mean Subtraction）：从每个像素值中减去一个全局平均值（mean）。这一步使得数据集的平均像素值趋近于零，即数据被中心化。
        # 除以标准差（Division by Standard Deviation）：将每个像素值除以一个全局标准差（std）。这一步使得数据集的像素值分布具有单位方差，从而进一步规范了数据的尺度。
        # output = (x - mean) / std
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        # 从“高度-宽度-通道”（Height-Width-Channel, HWC）格式转换为“通道-高度-宽度”（Channel-Height-Width, CHW）格式
        # HWC 格式：这是大多数图像读取库（如 OpenCV、PIL 等）默认输出的格式
        # CHW 格式：许多深度学习框架（包括 PyTorch 和 MindSpore）期望输入的图像数据是以 CHW 格式组织的张量
        vision.HWC2CHW()
    ]

    # 定义标签转换操作
    # 数据集中的标签（labels）转换为int32数据类型
    label_transform = transforms.TypeCast(mindspore.int32)

    # 加载MNIST数据集
    dataset = MnistDataset(path)
    # 应用图像转换操作到数据集的每个样本
    dataset = dataset.map(image_transforms, 'image')
    # 应用标签转换操作到数据集的每个样本
    dataset = dataset.map(label_transform, 'label')
    # 数据集中的样本组合成小批量
    # “小批量随机梯度下降”（mini-batch stochastic gradient descent, mini-batch SGD）
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe('MNIST_Data/train', 64)
test_dataset = datapipe('MNIST_Data/test', 64)

# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        # 将输入张量展平为一个一维向量
        self.flatten = nn.Flatten()
        # 这是一个容器，用来顺序地堆叠多个层
        # 
        self.dense_relu_sequential = nn.SequentialCell(
            # nn.Dense 层（也称为全连接层）
            # 输入 28*28 的向量，输出 512 的向量
            nn.Dense(28*28, 512),
            # ReLU 激活函数
            nn.ReLU(),
            # 512输入， 512 输出， 中间有权重和偏移
            nn.Dense(512, 512),
            nn.ReLU(),
            # 这是因为要 0-9 十个数，所以输入 上一层的 512, 输出为10
            nn.Dense(512, 10)
            # 最后一个 nn.Dense 层没有跟随激活函数，因为它的输出是分类任务的对数几率（logits）
        )

    # 前向传播方法
    def construct(self, x):
        # 调用 Flatten 层将输入图像数据展平为一维向量。
        x = self.flatten(x)
        # 将展平后的输入传递给由 SequentialCell 定义的层序列，依次通过每一层进行前向传播，最后返回未经过激活函数的输出（logits）。
        logits = self.dense_relu_sequential(x)
        return logits


# 实例化网络
model = Network()

# 数据集的大小，即一个 epoch 中的总步数（steps）
# 对于批处理的数据集来说，它等于总的样本数除以批次大小（batch size）
steps_per_epoch = train_dataset.get_dataset_size()

# 定义学习率衰减策略
decay_rate = 0.95  # 学习率衰减速率
decay_steps = steps_per_epoch * 2  # 每两个 epoch 衰减一次
lr_scheduler = ExponentialDecayLR(learning_rate=1e-1, decay_rate=decay_rate, decay_steps=decay_steps)

# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
# 随机梯度下降
# 通过迭代更新模型参数来最小化损失函数
# 1e-2 是学习率, 较高的学习率可能导致更快的收敛但可能不稳定；较低的学习率则更加稳定但收敛速度较慢。
optimizer = nn.SGD(model.trainable_params(), learning_rate=lr_scheduler)

# 定义检查点保存的各种配置选项
# save_checkpoint_steps=steps_per_epoch：这个参数指定了每隔多少步保存一次检查点
# 设置为 steps_per_epoch，意味着每个 epoch 结束后会保存一个检查点
# 确保每个 epoch 的训练状态都被记录下来，便于后续分析或恢复训练。
config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)

# 回调会在满足条件时自动保存模型的状态
# prefix="mnist"：指定保存的检查点文件名前缀
ckpt_callback = ModelCheckpoint(prefix="mnist", directory="./checkpoint", config=config)
# LossMonitor 回调
# LossMonitor 会根据这个参数定期输出当前的损失值。
loss_callback = LossMonitor(steps_per_epoch)

# SummaryCollector 是一个回调函数，用于收集训练过程中产生的日志信息并保存到指定目录。
# summary_collector = SummaryCollector(summary_dir="./tensorboard_logs", collect_specified_data={'collect_metric': True})

# model：定义的神经网络模型实例。继承自 nn.Cell 或其子类。
# loss_fn：损失函数对象，用于衡量模型预测值与真实标签之间的差异。
# optimizer：优化器对象，负责根据损失函数的梯度更新模型参数。
# metrics：训练和评估过程中监控的评估指标。在这里，{'accuracy'} 表示你希望计算模型的准确率
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})

# epoch：总训练轮数
# train_dataset：用于训练的数据集迭代器。
# valid_dataset：用于评估模型的数据集。
# valid_frequency：指定验证的频率。
# callbacks：回调函数列表，用于在训练的不同阶段执行特定操作，如保存检查点、监控损失等。
# initial_epoch：开始训练的起始 epoch。
trainer.fit(epoch=100, train_dataset=train_dataset, valid_dataset=test_dataset, callbacks=[ckpt_callback, loss_callback], initial_epoch=0)

# 在测试集上评估模型性能
eval_results = trainer.eval(test_dataset)

mindspore.export(model, model, file_name="model", file_format="MINDIR")

# 打印评估结果
print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
