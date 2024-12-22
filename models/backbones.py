import torch
from torchvision.transforms import transforms
from models.load_mae_as_vit import load_mae_as_vit
from util.transforms import mae_transform, dinov2_transform, get_transform
from models.load_clip_as_dino import load_clip_as_dino
import os
import yaml

def clip_vitb16(load_dir="./cache", device="cuda", **kwargs):
    """
    Load the CLIP ViT-B/16 model.

    Args:
        load_dir (str): Directory to load the model from.
        device (str): Device to load the model on.
        **kwargs: Additional arguments.

    Returns:
        model: The loaded CLIP model.
        preprocess: The preprocessing function for the model.
    """
    import clip
    model, preprocess = clip.load("ViT-B/16", device=device, download_root=load_dir)
    if 'custom_model' in kwargs and kwargs['custom_model']: 
        model, _ = load_clip_as_dino(16, load_dir)        
    return model, preprocess

def dinov2_vitb14(load_dir="./cache", device="cuda", **kwargs):
    """
    Load the DINOv2 ViT-B/14 model.

    Args:
        load_dir (str): Directory to load the model from.
        device (str): Device to load the model on.
        **kwargs: Additional arguments.

    Returns:
        model: The loaded DINOv2 model.
        preprocess: The preprocessing function for the model.
    """
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    return model, dinov2_transform

def mae_vitb16(load_dir="./cache", device="cuda", **kwargs):
    """
    Load the MAE ViT-B/16 model.

    Args:
        load_dir (str): Directory to load the model from.
        device (str): Device to load the model on.
        **kwargs: Additional arguments.

    Returns:
        model: The loaded MAE model.
        preprocess: The preprocessing function for the model.
    """
    model = load_mae_as_vit("mae_vitb16", None, load_dir=".")
    return model, mae_transform

class ModelFactory:
    def __init__(self, load_dir, device="cuda") -> None:
        """
        Initialize the ModelFactory with the given load directory and device.

        Args:
            load_dir (str): Directory to load the models from.
            device (str): Device to load the models on.
        """
        self.model_library = {
            "clip_vitb16": clip_vitb16,
            "dinov2_vitb14": dinov2_vitb14,
            "mae_vitb16": mae_vitb16
        }
        self.load_dir = load_dir
        self.device = device
        torch.hub.set_dir(load_dir)

    def init_backbone(self, model_name, **kwargs):
        """
        Initialize the backbone model.

        Args:
            model_name (str): Name of the model to initialize.
            **kwargs: Additional arguments.

        Returns:
            model: The initialized model.
            preprocess: The preprocessing function for the model.
        """
        model, preprocess = self.model_library[model_name](**kwargs)
        model = model.to(self.device)
        return model, preprocess

    def get_global_fn(self, model, model_name, custom_model=False):
        """
        Get the global function for the model.

        Args:
            model: The model instance.
            model_name (str): Name of the model.
            custom_model (bool): Flag indicating if the model is custom.

        Returns:
            function: The global function for the model.
        """
        if "custom" in model_name:
            return model.global_fn
        elif not custom_model:
            if "clip" in model_name:
                return model.encode_image
            elif "mae" in model_name:
                return lambda x: model(x).squeeze()
            else:
                return model
        else:
            return model

    def get_dense_fn(self, model, model_name):
        """
        Get the dense function for the model.

        Args:
            model: The model instance.
            model_name (str): Name of the model.

        Returns:
            function: The dense function for the model.
        """
        if "custom" in model_name:
            return model.dense_fn
        else:
            return lambda x: model.get_intermediate_layers(x, return_class_token=True)

    def load_custom_model(self, checkpoint, load_epoch=None, set_eval=False, device="cuda", **kwargs):
        """
        Load a custom model from a checkpoint.

        Args:
            checkpoint (str): Path to the checkpoint directory.
            load_epoch (int, optional): Epoch number to load. Defaults to None.
            set_eval (bool, optional): Flag to set the model to evaluation mode. Defaults to False.
            device (str): Device to load the model on.
            **kwargs: Additional arguments.

        Returns:
            model: The loaded custom model.
        """
        from contrastive.train_custom import LightningPersonalModel
        print("Loading from ", checkpoint)
        with open(os.path.join(checkpoint, "config.yaml"), "r") as f:
            model_config = yaml.safe_load(f)
        model = LightningPersonalModel(**model_config)
        if load_epoch is None:
            sd = torch.load(os.path.join(checkpoint, f'checkpoints/last.ckpt'))
        else:
            sd = torch.load(os.path.join(checkpoint, f'checkpoints/epoch=0{load_epoch-1}.ckpt'))
        model.load_state_dict(sd["state_dict"])
        if set_eval:
            model.personal_model.model.eval().requires_grad_(False)
            model.personal_model.mlp.eval().requires_grad_(False)
        model = model.personal_model.to(device)
        return model  

    def load(self, model_name, checkpoint=None, **kwargs):
        """
        Load a model by name or from a checkpoint.

        Args:
            model_name (str): Name of the model to load.
            checkpoint (str, optional): Path to the checkpoint directory. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            model: The loaded model.
            preprocess: The preprocessing function for the model.
        """
        if checkpoint is None:
            model, preprocess = self.init_backbone(model_name, **kwargs)
        else:
            model = self.load_custom_model(checkpoint, **kwargs)
            preprocess = get_transform(model_name)
        return model, preprocess