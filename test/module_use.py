
import mindspore
from mindspore import nn
from mindspore.train import load_checkpoint, load_param_into_net
from PIL import Image
from mindspore import Tensor
from mindspore.dataset import vision
import mindspore.dataset.transforms as transforms
from PIL import Image
import numpy as np


def preprocess_image(image_path):

    # 打开图像文件并转换为灰度图
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    
    # 定义图像预处理操作，使用 vision 模块中的操作，并使用 Compose 组合
    transform_pipeline = transforms.Compose([
        vision.Resize([28, 28]),          # 调整大小到指定尺寸
        vision.Rescale(1.0 / 255.0, 0),   # 缩放像素值到 [0.0, 1.0]
        vision.ToTensor(),                # 将图像转换为 Tensor
        vision.Normalize(mean=[0.1307], std=[0.3081])  # 归一化
    ])
    
    # 应用预处理操作
    transformed_img = transform_pipeline(img)
    
    # 取第一个元素（假设只有一个输出）
    transformed_img = transformed_img[0]
    
    #  NumPy 数组，转换为 MindSpore Tensor
    transformed_img = Tensor(transformed_img, dtype=mindspore.float32)
    
    # 增加 batch 维度和通道维度
    img_tensor = transformed_img.expand_dims(0)  # 添加 batch 维度

    return img_tensor


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

# 加载参数
param_dict = load_checkpoint(ckpt_file_name="checkpoint/mnist_1-10_938.ckpt")
load_param_into_net(model, param_dict)

# 将模型设置为评估模式
model.set_train(False)

# 执行推理
img_tensor = preprocess_image(image_path= './4.png')
print("Input tensor shape:", img_tensor.shape)
print("Input tensor min/max values:", img_tensor.min().asnumpy(), img_tensor.max().asnumpy())

output = model(img_tensor)

# 验证模型输出
print("Model output logits:", output.asnumpy())

predicted_class = np.argmax(output.asnumpy(), axis=1)
print(f"Predicted class: {predicted_class}")
