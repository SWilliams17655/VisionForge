import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes
from tqdm import tqdm
import torch.utils.data
import json
from PIL import Image

NEW_DICTIONARY = {0: 'Maritime Vessel',
                  1: 'Motorboat',
                  2: 'Sailboat',
                  3: 'Tugboat',
                  4: 'Barge',
                  5: 'Fishing Vessel',
                  6: 'Ferry',
                  7: 'Yacht',
                  8: 'Container Ship',
                  9: 'Oil Tanker'}

#**********************************************************************************************************************

class SatImageDataset(Dataset):

    def __init__(self, file_name, error_logger = None):
        """Dataset holding satellite images for training an object detection algorithm in pytorch.
        @:param filename -- The relative filename location for where the JSON is saved.
        @:param error_logger -- Logger to log errors during training.
        """
        self.error_logger = error_logger

        # Transform used for the tensor
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5),
            v2.ToDtype(torch.float16, scale=True),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        ])

        # Open and read the JSON file into a pandas dataframe
        with open(file_name, 'r') as infile:
            data = json.load(infile)
            feature_list = data['FEATURES']

            columns = ['IMAGE_ID', 'TYPE_ID', 'BBOX']
            output_list = []
            for feature in tqdm(feature_list):
                img_id = feature['IMAGE_ID']
                type_id = feature['TYPE_ID']
                bbox = feature['BBOX']

                if type_id <10: #Only adds valid type identification to the training data.
                    one_row = [img_id, type_id, bbox]  # Creates a row out of the data.
                    output_list.append(one_row)

            self.training_data = pd.DataFrame(output_list, columns=columns)
            self.image_list = self.training_data["IMAGE_ID"].unique()  # List of unique images
            print(f"Value Counts: \n")
            for i, value_count in enumerate(self.training_data["TYPE_ID"].value_counts().sort_index()):
                print(f"{i} {NEW_DICTIONARY[i]}: {value_count}")
            print(f"Loaded {len(self.image_list)} images with {len(self.training_data)} bounding boxes\n")
#***********************************************************************************************************************

    def __len__(self):
        """Returns the number of images in the dataset."""

        return len(self.image_list)

#***********************************************************************************************************************

    def __getitem__(self, index):
        """ Returns the image at that index with all associated bounding boxes.

        :param index: Index of the bounding box.
        :return: tuple with image_tensor & target formatted per pytorch documentation
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset.
        """

        # Isolates all bounding boxes associated with this image.
        subset_df = self.training_data[self.training_data["IMAGE_ID"] == self.image_list[index]]

        # Reads image then converts to tensor per
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset.
        image = Image.open(f"Training Data/train/{subset_df.iloc[0]["IMAGE_ID"]}")
        img_array = np.array(image)
        height_image = img_array.shape[0]
        width_image = img_array.shape[1]
        image = self.transform(img_array)
        image_tensor = torchvision.tv_tensors.Image(image)

        # Formats in a pytorch bounding box per
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset.
        boxes = subset_df["BBOX"]
        boxes_array = np.array([np.array(item, dtype=np.float32) for item in boxes])
        boxes_tensor = BoundingBoxes(boxes_array,
                                     format='XYXY',
                                     canvas_size=(height_image, width_image),
                                     dtype=torch.float16)

        # Sets labels equal to classification then converts to a tensor.
        labels = np.array(subset_df["TYPE_ID"])
        label_tensor = torch.tensor(labels, dtype=torch.int64)

        # Generates a unique identifier for all images.
        image_id = torch.tensor([index], dtype=torch.int16)

        # Calculates area.
        area = [((box[2]-box[0]) * (box[3]-box[1])) for box in boxes_tensor]
        area_tensor = torch.tensor(area, dtype=torch.float16)

        # Does not handle crowds so sets all to False.
        is_crowd_tensor = torch.zeros(len(subset_df), dtype=torch.uint8)

        # Converts to dictionary per pytorch documentation
        # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset
        target = {"boxes": boxes_tensor,
                  "labels": label_tensor,
                  "image_id": image_id,
                  "area": area_tensor,
                  "is_crowd": is_crowd_tensor
                  }

        return image_tensor, target