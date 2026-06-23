import os
import torch
from .utils import DatasetBase, read_split
from torchvision import datasets, transforms

template = ['a photo of a {}.']

print('preparing Places365 dataset')

# Define the transform using CLIP's preprocessing
# transform = preprocess

# Load the Places365 dataset using torchvision
# places_data = datasets.Places365(
#     root="./data",
#     split='val',  # use 'train-standard' for training data or 'val' for validation data
#     small=True,   # use the smaller version of the dataset
#     download=True,
#     transform=transform
# )

# Load the class names from the 'categories_places365.txt' file
# def load_places365_classnames():
#     file_path = places_data.root + "/categories_places365.txt"
#     class_names = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             class_name = line.strip().split(' ')[0].split('/')[-1]
#             class_names.append(class_name)
#     return class_names

# Get the class names
# class_names = load_places365_classnames()

class Places365:
   def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 batch_size_eval=128,
                 num_workers=24,
                 classnames='openai'):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers        
        self.template = template
        self.templates = template
        
        places_data = datasets.Places365(
            root=location,
            split='val',  # use 'train-standard' for training data or 'val' for validation data
            small=True,   # use the smaller version of the dataset
            download=True,
            transform=preprocess
        )

        file_path = places_data.root + "/categories_places365.txt"
        class_names = []
        with open(file_path, 'r') as f:
            for line in f:
                class_name = line.strip().split(' ')[0].split('/')[-1]
                class_name = class_name.replace('_', ' ')
                class_names.append(class_name)
        self._classnames = class_names
        print('Class names:', self._classnames)
        
        places_data = datasets.Places365(
            root=location,
            split='val',  # use 'train-standard' for training data or 'val' for validation data
            small=True,   # use the smaller version of the dataset
            download=False,
            transform=preprocess
        )

        file_path = places_data.root + "/categories_places365.txt"
        class_names = []
        with open(file_path, 'r') as f:
            for line in f:
                class_name = line.strip().split(' ')[0].split('/')[-1]
                class_names.append(class_name)
        self._classnames = class_names
        self.classnames = class_names
        print('Class names:', self._classnames)

        self.test_dataset = places_data
        self.test = self.test_dataset
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            shuffle=False, pin_memory=True
        )
