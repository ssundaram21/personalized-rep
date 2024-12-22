import torch
import torch.nn.functional as F
from models.backbones import ModelFactory
from util.constants import EMBED_DIMS, PATCH_SIZES

class PersonalModel(torch.nn.Module):
    def __init__(self, 
                 model_type: str = "dinov2_vitb14", 
                 hidden_size: int = 512, 
                 lora: bool = False, 
                 patch_tokens: bool = False,
                 load_dir: str = "./models", 
                 device: str = "cuda", 
                 **kwargs):
        """
        Initializes a personal model container for training and eval.
        Args:
            model_type (str): Base backbone model.
            hidden_size (int): Dimension of the final MLP hidden layer that extracted ViT features are passed to.
            lora (bool): True means finetuning with LoRA. Replaces the MLP with an identity function.
            patch_tokens (bool): Whether to use patch tokens for training.
            load_dir (str): Path to pretrained backbone checkpoints.
            device (str): Device to load the model on.
            **kwargs: Additional keyword arguments.
        """
        print(f"Initializing personalized model with {model_type}")
        super().__init__()
        self.config = None
        self.model_type = model_type
        self.patch_tokens = patch_tokens
        self.lora = lora

        self.model_factory = ModelFactory(load_dir=load_dir, device=device)
        self.model, self.global_fn, self.dense_fn = self._initialize_backbone(load_dir, device)
        self.embed_size = self._get_embed_dims()        
        self.hidden_size = hidden_size
        
        if self.lora:
            self.mlp = torch.nn.Identity()
        else:
            print("Initializing an MLP")
            self.mlp = MLP(in_features=self.embed_size, hidden_size=self.hidden_size)        
            self.model = self.model.eval().to(device)

    def forward_global(self, img):
        """
        Forward pass for global features.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Global features.
        """
        return self.global_fn(img)

    def forward_dense(self, img):
        """
        Forward pass for dense (spatial) features.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Dense features.
        """
        return self.dense_fn(img)

    def forward(self, img):
        """
        Forward pass for the model.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Embedded features.
        """
        if self.patch_tokens:
            feats = self.forward_dense(img)
            patch_tokens, cls_token = feats[0][0], feats[0][1]
            n = patch_tokens.shape[0]
            s = int(patch_tokens.shape[1] ** 0.5)
            patch_tokens_pooled = F.adaptive_avg_pool2d(patch_tokens.reshape(n, s, s, -1).permute(0, 3, 1, 2), (1, 1)).squeeze()
            cls_token = cls_token.reshape(patch_tokens_pooled.shape)
            feats = torch.cat((cls_token, patch_tokens_pooled), dim=-1)
        else:
            feats = self.forward_global(img)
        embed = self.mlp(feats)
        return embed
        
    def _initialize_backbone(self, load_dir, device):
        """
        Initializes the backbone model.

        Args:
            load_dir (str): Path to pretrained ViT checkpoints.
            device (str): Device to load the model on.

        Returns:
            tuple: Model, global function, and dense function.
        """
        model, _ = self.model_factory.load(self.model_type, custom_model=True)
        global_fn = self.model_factory.get_global_fn(model, self.model_type, custom_model=True)
        dense_fn = self.model_factory.get_dense_fn(model, self.model_type)
        return model, global_fn, dense_fn

    def _get_embed_dims(self):
        """
        Gets the embedding dimensions.

        Returns:
            int: Embedding dimensions.
        """
        if not self.patch_tokens:
            return EMBED_DIMS[self.model_type]
        else:
            return EMBED_DIMS[self.model_type] * 2

class MLP(torch.nn.Module):
    """
    MLP head with a single hidden layer and residual connection.
    """
    def __init__(self, in_features: int, hidden_size: int = 512):
        """
        Initializes the MLP.

        Args:
            in_features (int): Number of input features.
            hidden_size (int): Number of hidden units.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(in_features, self.hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(self.hidden_size, in_features, bias=True)

    def forward(self, img):
        """
        Forward pass for the MLP.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor after applying MLP and residual connection.
        """
        x = self.fc1(img)
        x = F.relu(x)
        return self.fc2(x) + img
