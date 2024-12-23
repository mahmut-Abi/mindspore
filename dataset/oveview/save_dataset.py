#!/bin/python3.11

import os
import mindspore.dataset as ds

ds.config.set_seed(1234)

data = [1, 2, 3, 4, 5, 6]
dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1"])
if os.path.exists("./train_dataset.mindrecord"):
    os.remove("./train_dataset.mindrecord")
if os.path.exists("./train_dataset.mindrecord.db"):
    os.remove("./train_dataset.mindrecord.db")
dataset.save("./train_dataset.mindrecord")