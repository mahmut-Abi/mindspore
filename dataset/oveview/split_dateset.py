#!/bin/python3.11

import mindspore.dataset as ds

data = [1, 2, 3, 4, 5, 6]
dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1"], shuffle=False)

train_dataset, eval_dataset = dataset.split([4, 2])

print(">>>> train dataset >>>>")
for item in train_dataset.create_dict_iterator():
    print(item)