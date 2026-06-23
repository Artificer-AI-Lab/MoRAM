from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .places365 import Places365
from .collections import (
    CIFAR100,
    MNIST,
)

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "aircraft": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "cifar100": CIFAR100,
                "imagenet": ImageNet,
                "places365": Places365,
                }


def build_dataset(dataset, root_path, shots, preprocess=None):
    if dataset == 'imagenet' or dataset == 'places365':
        return dataset_list[dataset](preprocess=preprocess, location=root_path)
    else:
        return dataset_list[dataset](root_path, shots)
