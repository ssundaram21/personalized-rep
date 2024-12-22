import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable
from util.constants import IMG_SIZE

class TripletDataset(Dataset):
    def __init__(self, csv_path: str, preprocess: Callable, augment: bool = True, **kwargs):
        """
        Args:
            csv_path (str): Path to the CSV file containing dataset information.
            preprocess (Callable): Preprocessing function to apply to the images.
            augment (bool): Whether to apply classic data augmentation.
            **kwargs: Additional keyword arguments.
        """
        self.csv_path = csv_path
        self.csv = pd.read_csv(self.csv_path)
        self.preprocess_fn = preprocess
        self.do_augment = augment
        self.img_size = IMG_SIZE

        print(f"Augment: {self.do_augment}")

        # Define augmentation transformations
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(size=(self.img_size, self.img_size), scale=(0.8, 1.0))  # Adjust size and scale as needed
        ])

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.csv)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: (img_ref, img_0, img_1, label, id)
        """
        id = self.csv.iloc[idx, 0]
        label = self.csv.iloc[idx, 1].astype(np.float32)  # Which of 0 or 1 is the positive pair
        img_ref = self.preprocess_fn(Image.open(self.csv.iloc[idx, 2]))
        img_0 = self.preprocess_fn(Image.open(self.csv.iloc[idx, 3]))
        img_1 = self.preprocess_fn(Image.open(self.csv.iloc[idx, 4]))

        if self.do_augment:
            img_ref = self.augment(img_ref)
            img_0 = self.augment(img_0)
            img_1 = self.augment(img_1)
        
        return img_ref, img_0, img_1, label, id
