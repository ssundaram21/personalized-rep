from torchvision import transforms
import glob
import os
import torch

norms = {
    "dino": transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "mae": transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
}

dreamsim_transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

dinov2_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            norms['dino'],
        ])

mae_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            norms['mae'],
        ])


clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    lambda x: x.convert('RGB'),
    transforms.ToTensor(),
    norms['clip'],
])

def get_transform(model_type):
    """
    Get the appropriate transform for the given model type.

    Args:
        model_type (str): The type of model for which to get the transform.

    Returns:
        torchvision.transforms.Compose: The composed transform for the specified model type.

    Raises:
        ValueError: If the model_type is not supported.
    """
    if "dinov2" in model_type:
        return dinov2_transform
    elif "mae" in model_type:
        return mae_transform
    elif "clip" in model_type:
        return clip_transform
    else:
        raise ValueError(f"{model_type} not supported.")




