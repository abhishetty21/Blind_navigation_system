 # augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.5),
        ToTensorV2()
    ])

