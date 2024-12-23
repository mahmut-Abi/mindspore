#!/bin/python3.11

## note
# mindspore.nn.Cell 中可定义其他Cell实例作为子模块。这些子模块是网络中的组成部分，
# 自身也可能包含可学习的Parameter（如卷积层的权重和偏置）和其他子模块。这种层次化的模块结构允许用户构建复杂且可重用的神经网络架构。
# mindspore.nn.Cell 提供 cells_and_names 、 insert_child_to_cell 等接口实现子模块管理功能。

from mindspore import nn

class MyCell(nn.Cell):
    def __init__(self):
        super(MyCell, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 使用insert_child_to_cell添加子模块
        self.insert_child_to_cell('conv3', nn.Conv2d(64, 128, 3, 1))

        self.sequential_block = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU()
        )

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sequential_block(x)
        return x

module = MyCell()

# 使用cells_and_names遍历所有子模块（包括直接和间接子模块）
for name, cell_instance in module.cells_and_names():
    print(f"Cell name: {name}, type: {type(cell_instance)}")
