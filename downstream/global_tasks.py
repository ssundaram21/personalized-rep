import torch
import sys
from torchvision import transforms
from util.transforms import get_transform
import argparse
from PIL import Image
import os
from tqdm import tqdm
import torch.nn.functional as F
from util.misc import load_features
from downstream.classification import classification
from downstream.retrieval import retrieval
from dataset.image_dataset import EvalDataset
import numpy as np

def run_global_tasks(args, model_fn, model_name, preprocess, class_to_idx, checkpoint_dir=None, device="cuda"):
    """
    Run global tasks for evaluation.

    Args:
        args (argparse.Namespace): Arguments for the task.
        model_fn (callable): Model function to generate embeddings.
        model_name (str): Name of the model.
        preprocess (callable): Preprocessing function for the dataset.
        class_to_idx (dict): Mapping from class names to indices.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to None.
        device (str, optional): Device to run the tasks on. Defaults to "cuda".

    Returns:
        dict: Results of the classification and retrieval tasks.
    """
    torch.cuda.empty_cache()
    all_classes = [int(x) for x in class_to_idx.values()]
    positive_cls = int(args.class_id)
    # train_class_subset = [int(x) for x in args.train_class_subset.split(",")]
    print(f"{len(class_to_idx)} total classes. Eval'ing {positive_cls} classifiers")

    test_results = {}
    # Set up datasets   
    train_dataset = EvalDataset(
        root=args.real_train_root, 
        class_to_idx=class_to_idx,
        transform=preprocess
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.downstream_batch_size, 
        shuffle=True,
        num_workers=args.downstream_workers, 
        pin_memory=True
    )
    test_dataset = EvalDataset(
        root=args.test_root, 
        class_to_idx=class_to_idx, 
        transform=preprocess
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.downstream_batch_size, 
        shuffle=False,
        num_workers=args.downstream_workers, 
        pin_memory=True
    )
    
    # Get embeddings
    cache_embeds = not ("custom" in model_name)
    train_save_path, test_save_path = None, None
    if cache_embeds:
        train_save_path = os.path.join(args.embed_path, f"{args.dataset}_{model_name}_global_train.npz")
        test_save_path = os.path.join(args.embed_path, f"{args.dataset}_{model_name}_global_test.npz")

    print(f"Caching embeddings: {cache_embeds}")
    train_embeds, train_labels, train_paths = load_features(train_loader, model_fn, train_save_path, cache=cache_embeds)
    train_embeds = train_embeds.astype(np.float32)
    test_embeds, test_labels, test_paths = load_features(test_loader, model_fn, test_save_path, cache=cache_embeds)
    test_embeds = test_embeds.astype(np.float32)
    
    print(f"{train_embeds.shape, train_labels.shape} real train embeddings total for {len(np.unique(train_labels))} unique classes")
    print(f"{test_embeds.shape, test_labels.shape} real test embeddings total for {len(np.unique(test_labels))} unique classes")

    # Run global tasks
    classification_results = classification(
        train_embeds, 
        train_labels, 
        test_embeds, 
        test_labels, 
        test_paths, 
        positive_cls, 
        save_extra=True
    )
    retrieval_results = retrieval(
        train_embeds, 
        train_labels, 
        test_embeds, 
        test_labels, 
        test_paths, 
        positive_cls 
    )
    test_results[args.pos_class_name] = {**classification_results, **retrieval_results}

    return test_results

    print("done :)")

if __name__ == "__main__":
    args = parse_args()
    classify(args)
