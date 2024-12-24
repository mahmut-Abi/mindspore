#!/bin/python3

from datasets import load_dataset
import numpy as np
from mindspore.dataset import GeneratorDataset
from mindspore.mindrecord import FileWriter

def create_mindspore_dataset_from_parquet(file_path, output_file_name="train.mindrecord", num_files=1):
    """Create a MindSpore dataset from a Parquet file and save it as a MindRecord file.
    
    Args:
        file_path (str): Path to the Parquet file.
        output_file_name (str): Name of the output MindRecord file.
        num_files (int): Number of MindRecord files to split the data into.
    """
    
    def load_parquet():
        """Load a Parquet dataset using the Hugging Face datasets library."""
        hf_dataset = load_dataset("parquet", data_files=file_path)
        return hf_dataset["train"]

    def dataset_generator(hf_dataset):
        """Generator function to yield image and label data from the Hugging Face dataset."""
        for data in hf_dataset:
            image_data = data["image"]["bytes"]  # Assuming 'image' column contains bytes of image data
            image_label = data["labels"]         # Assuming 'labels' column contains labels
            yield (image_data, image_label)

    # Step 1: Load the Parquet dataset
    hf_dataset = load_parquet()

    # Step 2: Define the generator based on the loaded dataset
    generator = dataset_generator(hf_dataset)

    # Step 3: Create the GeneratorDataset
    dataset = GeneratorDataset(generator, ["image", "label"])
    
    # Print the size of the dataset for verification
    print(f"Dataset size: {dataset.get_dataset_size()}")

    # Step 4: Define schema for MindRecord file
    schema_json = {
        "image": {"type": "bytes"},
        "label": {"type": "int32"}
    }

    # Step 5: Initialize FileWriter and write the dataset to MindRecord file
    writer = FileWriter(output_file_name, num_files)
    writer.add_schema(schema_json, "mindspore_dataset")
    
    for item in dataset.create_dict_iterator(num_epochs=1):
        writer.write_raw_data([{
            "image": item["image"].asnumpy().tobytes(),
            "label": int(item["label"])
        }])

    writer.commit()

# 使用示例
if __name__ == "__main__":
    create_mindspore_dataset_from_parquet(
        file_path="./datasets/test/test-00000-of-00001.parquet",
        output_file_name="test.mindrecord",
        num_files=1
    )