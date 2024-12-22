import os
from collections import defaultdict
from PIL import Image, ImageFile
import torch
from torchvision import transforms
from util.data_util import get_paths

def pil_loader(path):
    """
    Load an image from the given path and convert it to RGB format.

    Args:
        path (str): Path to the image file.

    Returns:
        PIL.Image.Image: PIL Image in RGB format.
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        """
        Initialize the dataset.

        Args:
            root (str): Dataset root directory. Should follow the structure class1/0.jpg...n.jpg, class2/0.jpg...n.jpg
            class_to_idx (dict): Dictionary mapping the class names to integers.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.transform = transform
        self.label_to_idx = class_to_idx

        self.paths = []
        self.labels = []

        for cls in class_to_idx:
            cls_paths = get_paths(os.path.join(root, cls))
            self.paths += cls_paths
            self.labels += [int(self.label_to_idx[cls]) for _ in cls_paths]

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: Tuple containing the image, label, image path, and optionally the depth image.
        """
        im_path, label = self.paths[idx], self.labels[idx]
        img = pil_loader(im_path)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label, im_path