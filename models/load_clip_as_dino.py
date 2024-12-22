import torch
from .vision_transformer import vit_base, VisionTransformer
import os


class QuickGELU(torch.nn.Module):
    """
    QuickGELU activation function.
    """

    def forward(self, x: torch.Tensor):
        """
        Forward pass for QuickGELU.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying QuickGELU.
        """
        return x * torch.sigmoid(1.702 * x)


def load_clip_as_dino(patch_size, load_dir="./cache", l14=False):
    """
    Load CLIP model as DINO.

    Args:
        patch_size (int): Patch size for the Vision Transformer.
        load_dir (str): Directory to load the model weights from.
        l14 (bool): Flag to indicate if the L14 model should be loaded.

    Returns:
        tuple: Loaded Vision Transformer model and projection weights.
    """
    if l14:
        sd = torch.load(os.path.join(load_dir, 'clipl14_as_dino_vitl.pth.tar'), map_location='cpu', weights_only=True)
        dino_vit = VisionTransformer(**sd['kwargs'])
        sd = sd['state_dict']
    else:
        sd = torch.load(os.path.join(load_dir, f'clip_vitb{patch_size}_pretrain.pth.tar'), weights_only=True)['state_dict']
        dino_vit = vit_base(patch_size=patch_size)

    dino_vit.pos_drop = torch.nn.LayerNorm(dino_vit.embed_dim)
    proj = sd.pop('proj')
    dino_vit.load_state_dict(sd)

    # GeLU -> QuickGeLU
    for blk in dino_vit.blocks:
        blk.mlp.act = QuickGELU()

    # LN eps 1e-6 -> 1e-5
    for m in dino_vit.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.eps = 1e-5

    return dino_vit, proj