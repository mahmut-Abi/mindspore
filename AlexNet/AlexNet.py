#!/bin/python3

import mindspore
import numpy as np
from datasets import load_dataset
import mindspore.dataset as ds
from mindspore import nn, context, Tensor
from mindspore.train import Model, LossMonitor
from mindspore.dataset import vision, transforms
from mindspore.common.initializer import Normal
from mindspore.dataset import GeneratorDataset
from PIL import Image
from io import BytesIO

context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
context.set_context(runtime_num_threads=32)

def create_mindspore_dataset_from_parquet(file_path, batch_size=64):
    
    def load_parquet():
        """Load a Parquet dataset using the Hugging Face datasets library."""
        hf_dataset = load_dataset("parquet", data_files=file_path)
        return hf_dataset["train"]

    def dataset_generator(hf_dataset):
        """Generator function to yield image and label data from the Hugging Face dataset."""
        for data in hf_dataset:
            image_data = data["image"]["bytes"]  # Assuming 'image' column contains bytes of image data
            image_label = data["labels"]         # Assuming 'labels' column contains labels

            # Decode image from bytes
            image = Image.open(BytesIO(image_data))
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Debug: Print original image mode and size
            
            yield (image, image_label)

    # Step 1: Load the Parquet dataset
    hf_dataset = load_parquet()

    # Step 2: Define the generator based on the loaded dataset
    generator = dataset_generator(hf_dataset)

    # Step 3: Create the GeneratorDataset
    dataset = GeneratorDataset(generator, ["image", "label"])

    # Step 4: Apply transformations to the dataset
    transform_img = [
            vision.Resize((227, 227)),  # Resize to match AlexNet input size
            vision.ToTensor(),          # Convert PIL Image to Tensor and change from HWC to CHW format
            vision.Rescale(1.0 / 255.0, 0),  # Rescale pixel values to [0, 1]
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)  # Normalize with ImageNet mean/std
        ]
    
    dataset = dataset.map(operations=transform_img, input_columns=["image"])
    dataset = dataset.map(operations=transforms.TypeCast(mindspore.int32), input_columns=["label"])
    
    # Step 5: Batch and shuffle the dataset
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return dataset

class AlexNet(nn.Cell):
    def __init__(self, num_classes=2, dropout=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell([
            nn.Conv2d(3, 64, kernel_size=11, stride=4, pad_mode='valid', weight_init=Normal(0.01)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, pad_mode='same', weight_init=Normal(0.01)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, pad_mode='same', weight_init=Normal(0.01)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, pad_mode='same', weight_init=Normal(0.01)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode='same', weight_init=Normal(0.01)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])
        self.classifier = nn.SequentialCell([
            nn.Dropout(p=dropout),
            nn.Dense(in_channels=256 * 6 * 6, out_channels=4096, weight_init=Normal(0.01)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Dense(in_channels=4096, out_channels=4096, weight_init=Normal(0.01)),
            nn.ReLU(),
            nn.Dense(in_channels=4096, out_channels=num_classes, weight_init=Normal(0.01))
        ])

    def construct(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)  # Flatten the output of convolutional layers
        x = self.classifier(x)
        return x

# 初始化AlexNet模型
net = AlexNet(num_classes=2)

# 定义损失函数和优化器
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.9)

# 创建Model实例
model = Model(net, loss_fn=loss, optimizer=opt, metrics={"accuracy"})

# 数据预处理和加载
train_ds = create_mindspore_dataset_from_parquet('./datasets/train/train-00000-of-00001.parquet', batch_size=64)

# 打印数据集大小以验证
print(f"Train Dataset size: {train_ds.get_dataset_size()}")

train_ds.save("ds_has_been_normalzied")
# 训练模型
model.train(epoch=10, train_dataset=train_ds, callbacks=[LossMonitor(300)])

# 测试模型（同样需要根据实际情况编写）
# val_ds = create_mindspore_dataset_from_parquet('datasets/test-00000-of-00001.parquet')
# acc = model.eval(val_ds)
# print("Validation Accuracy:", acc)