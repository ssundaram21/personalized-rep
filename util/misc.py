import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
import logging
from torchmetrics import Metric
from pytorch_lightning import seed_everything
from util import pidfile
import torchvision.transforms as T

def embed_stack(embeds, labels, paths):
    """
    Combine embeddings, labels, and paths into one torch tensor each.

    Args:
        embeds (list): List of embedding tensors.
        labels (list): List of label tensors.
        paths (list): List of paths.

    Returns:
        tuple: Combined embeddings, labels, and paths as numpy arrays.
    """
    embeds = torch.vstack(embeds).numpy()
    labels = torch.concatenate(labels).numpy()
    return embeds, labels, np.concatenate(paths)


def load_features_simple(embed_file):
    """
    Load features from a presaved embedding file and returns them without formatting.

    Args:
        embed_file (str): Path to the embedding file.

    Returns:
        tuple: Normalized embeddings, labels, and paths as tensors.
    """
    embed_dict = np.load(embed_file)
    embeds, labels, paths = embed_dict['embeds'], embed_dict['labels'], embed_dict['paths']
    embeds = torch.nn.functional.normalize(torch.Tensor(embeds), dim=1, p=2).numpy()
    return torch.Tensor(embeds), labels, paths


def load_features_as_dict(embed_file):
    """
    Load features from an embedding file and return as a dictionary.

    Args:
        embed_file (str): Path to the embedding file.

    Returns:
        dict: Dictionary with paths as keys and normalized embeddings as values.
    """
    embed_dict = np.load(embed_file)
    embeds, labels, paths = embed_dict['embeds'], embed_dict['labels'], embed_dict['paths']
    embeds = torch.nn.functional.normalize(torch.Tensor(embeds), dim=1, p=2).numpy()
    embed_dict = {p: torch.Tensor(e) for p, e in zip(list(paths), list(embeds))}
    return embed_dict


def feature_extract_and_save(dataloader, model, save_path, cache=False, device="cuda"):
    """
    Extract features using a model and save them to a file.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        model (nn.Module): Model to use for feature extraction.
        save_path (str): Path to save the extracted features.
        cache (bool): Whether to cache the features.
        device (str): Device to use for computation.

    Returns:
        tuple: Normalized embeddings, labels, and paths as numpy arrays.
    """
    embeds = []
    labels = []
    paths = []
    for im, label, path in tqdm(dataloader):
        im = im.to(device)
        paths.append(path)
        with torch.no_grad():
            embeds.append(model(im).to("cpu"))
            labels.append(label)
    embeds, labels, paths = embed_stack(embeds, labels, paths)
    if cache:
        np.savez(save_path, embeds=embeds, labels=labels, paths=paths)
    embeds = torch.nn.functional.normalize(torch.Tensor(embeds), dim=1, p=2).numpy()
    return embeds, labels, paths


def set_all_seeds(seed):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def cosine_similarity(embed1, embed2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        embed1 (Tensor): First embedding tensor.
        embed2 (Tensor): Second embedding tensor.

    Returns:
        Tensor: Cosine similarity between the embeddings.
    """
    return F.cosine_similarity(embed1, embed2, dim=-1)


def l2_dist(embed1, embed2):
    """
    Compute L2 distance between two embeddings.

    Args:
        embed1 (Tensor): First embedding tensor.
        embed2 (Tensor): Second embedding tensor.

    Returns:
        Tensor: L2 distance between the embeddings.
    """
    return torch.sqrt(torch.sum((embed1 - embed2) ** 2, dim=-1))


def load_features(loader, model, embed_path, cache=False):
    """
    Load features from a file or compute and save them.

    Args:
        loader (DataLoader): DataLoader for the dataset.
        model (nn.Module): Model to use for feature extraction.
        embed_path (str): Path to save or load the features.
        cache (bool): Whether to cache the features.

    Returns:
        tuple: Normalized embeddings, labels, and paths as numpy arrays.
    """
    if cache and os.path.exists(embed_path):
        print(f"Loading features from {embed_path}")
        embed_dict = np.load(embed_path)
        embeds, labels, paths = embed_dict['embeds'], embed_dict['labels'], embed_dict['paths']
        embeds = torch.nn.functional.normalize(torch.Tensor(embeds), dim=1, p=2).numpy()
        return embeds, labels, paths
    else:
        print(f"Computing features")
        if cache:
            print(f"Saving to {embed_path}")
        return feature_extract_and_save(loader, model, embed_path, cache=cache)


def load_features_from_combined_embeds(synthetic_train_pathfile, combined_embedding_file):
    """
    Load features from combined embeddings file based on a synthetic train pathfile.

    Args:
        synthetic_train_pathfile (str): Path to the synthetic train pathfile.
        combined_embedding_file (str): Path to the combined embedding file.

    Returns:
        tuple: Normalized embeddings, labels, and paths as numpy arrays.
    """
    combined_embeds, combined_labels, combined_paths = load_features_simple(combined_embedding_file)
    with open(synthetic_train_pathfile, "r") as f:
        pathfile = json.load(f)
    concat_pathfile = np.concatenate(list(pathfile.values()))

    print(f"Loading embeddings for {synthetic_train_pathfile} from {combined_embedding_file}")
    mask = np.isin(combined_paths, concat_pathfile)
    embeds = combined_embeds[mask]
    labels = combined_labels[mask]
    paths = combined_paths[mask]
    embeds, labels, paths = np.array(embeds), np.array(labels), np.array(paths)
    embeds = torch.nn.functional.normalize(torch.Tensor(embeds), dim=1, p=2).numpy()
    return embeds, labels, paths


def seed_worker(worker_id):
    """
    Seed worker for reproducibility in DataLoader.

    Args:
        worker_id (int): Worker ID.
    """
    clear_logging()
    worker_seed = torch.initial_seed() % 2 ** 32
    seed_everything(worker_seed)


def clear_logging():
    """
    Clear logging for specific loggers.
    """
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if 'lightning' in name]
    for logger in loggers:
        logger.setLevel(logging.ERROR)

class Mean(Metric):
    """
    Compute the mean of values.
    """
    def __init__(self):
        super().__init__()
        self.add_state("vals", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denom", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, val, denom):
        """
        Update the mean with new values.

        Args:
            val (Tensor): Value to add.
            denom (Tensor): Denominator to add.
        """
        self.vals = self.vals + val
        self.denom = self.denom + denom

    def compute(self):
        """
        Compute the mean.
        """
        return self.vals / self.denom


def make_exp_name(args):
    """
    Create an experiment name based on the arguments.

    Args:
        args (Namespace): Arguments for the experiment.

    Returns:
        str: Generated experiment name.
    """
    tag = args.tag if len(args.tag) > 0 else ""
    training_method = "lora" if args.use_lora else "mlp"
    if args.synthetic_train_pathfile is not None:
        dataset_name = args.synthetic_train_pathfile.split("/")[-1].split(".json")[0]
    if args.synthetic_train_root is not None:
        dataset_name = "_".join(args.synthetic_train_root.split("/")[-2:])
    name = (f'{tag}_ds_{dataset_name}_model_{str(args.train_model_type)}_train_{str(training_method)}_'
            f'synth_num_{args.num_synthetic}_triplet_num_{args.num_triplets}_ref_{args.ref_type}_'
            f'loss_{args.train_loss_fn}_aug_{args.train_augment}')
    if args.patch_tokens:
        name += f'_patch_tokens'
    name += f'_seed_{str(args.seed)}'
    return name


def make_triplet_dirname(args):
    """
    Create a directory name for triplets based on the arguments.

    Args:
        args (Namespace): Arguments for the experiment.

    Returns:
        str: Generated triplet directory name.
    """
    return os.path.join(args.output_path, "triplets")


def make_checkpoint_dirname(args):
    """
    Create a directory name for checkpoints based on the arguments.

    Args:
        args (Namespace): Arguments for the experiment.

    Returns:
        str: Generated checkpoint directory name.
    """
    exp_dir = os.path.join(args.output_path, "train_outputs")
    print(f"exp dir: {exp_dir}")
    return exp_dir

def checkpoint_exists(ckpt_dir, epoch):
    """
    Check if a checkpoint exists for a given epoch.

    Args:
        ckpt_dir (str): Directory containing checkpoints.
        epoch (int): Epoch number to check.

    Returns:
        tuple: Boolean indicating if checkpoint exists and checkpoint path.
    """
    lightning_ckpts = os.path.join(ckpt_dir, "lightning_logs")
    if os.path.exists(lightning_ckpts):
        ckpt_versions = os.listdir(lightning_ckpts)
        ckpt_versions_sorted = sorted(
            (os.path.join(lightning_ckpts, ckpt) for ckpt in os.listdir(lightning_ckpts) if 'version' in ckpt),
            key=os.path.getmtime, reverse=True
        )
        ckpt_versions = [os.path.basename(ckpt) for ckpt in ckpt_versions_sorted]
        for version in ckpt_versions:
            if "checkpoints" in os.listdir(os.path.join(lightning_ckpts, version)):
                if f'epoch={(int(epoch)-1):02}.ckpt' in os.listdir(os.path.join(lightning_ckpts, version, "checkpoints")):
                    return True, os.path.join(lightning_ckpts, version)
    return False, None


def training_in_progress(ckpt_dir):
    """
    Check if training is in progress based on checkpoints.

    Args:
        ckpt_dir (str): Directory containing checkpoints.

    Returns:
        bool: True if training is in progress, False otherwise.
    """
    lightning_ckpts = os.path.join(ckpt_dir, "lightning_logs")
    ckpt_versions_sorted = []
    if os.path.exists(lightning_ckpts):
        ckpt_versions = os.listdir(lightning_ckpts)
        ckpt_versions_sorted = sorted(
            (os.path.join(lightning_ckpts, ckpt) for ckpt in os.listdir(lightning_ckpts) if 'version' in ckpt),
            key=os.path.getmtime, reverse=True
        )
    if len(ckpt_versions_sorted) > 0:
        return pidfile.check_if_job_taken(ckpt_versions_sorted[0])
    else:
        return False