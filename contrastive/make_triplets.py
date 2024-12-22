import pandas as pd
import glob
import os
import random
from util.data_util import get_paths, load_pathfile, load_paths_from_root

def sample_triplet(triplet):
    """
    Randomize positive/negative ordering.

    Args:
        triplet (tuple): A tuple containing reference, positive, and negative samples.

    Returns:
        tuple: A tuple containing the randomized triplet and the label indicating the position of the positive sample.
    """
    ref, pos, neg = triplet

    # Randomly rearrange the positive and negative
    if random.randint(1, 2)==1:
        return (ref, pos, neg), 0
    else:
        return (ref, neg, pos), 1

def get_positives(synthetic_paths, real_paths, ref_type):
    """
    Get positive samples based on the reference type.

    Args:
        synthetic_paths (list): List of synthetic paths.
        real_paths (list): List of real paths.
        ref_type (str): Type of reference ('real', 'synthetic', or 'both').

    Returns:
        list: A list containing the reference and positive samples.
    """
    if ref_type == "real":
        ref_paths = real_paths
    elif ref_type == "synthetic":
        ref_paths = synthetic_paths
    elif ref_type == "both":
        ref_paths = real_paths + synthetic_paths
    
    ref = random.sample(ref_paths, 1)[0] 
    pos = random.sample(synthetic_paths, 1)[0]
    positives = [ref, pos]
    return positives 
    
def get_negative(negative_paths):
    """
    Sample a negative sample from a list containing negative samples 
    """
    return random.sample(negative_paths, 1)

def setup_triplets(
    cls, 
    cls_name, 
    dataset_name,
    synthetic_pathfile, 
    synthetic_root,
    num_synthetic,
    negatives_root, 
    real_train_root, 
    num_triplets, 
    save_path, 
    ref_type="real",
    verbose=False
    ):
    """
    Sets up triplets for a given class by generating a dataset of triplets consisting of 
    reference, positive, and negative images. The triplets will be used for training 
    contrastive learning models.

    Args:
        cls (int): Class id to make the triplet dataset for.
        cls_name (str): Corresponding class name.
        dataset_name (str): Name of the dataset.
        synthetic_pathfile (str): Pathfile form of synthetic data (i.e., {class_id: [path list], ...}).
        synthetic_root (str): Root directory of synthetic data.
        num_synthetic (int): Number of synthetic positives to use.
        negatives_root (str): Root directory of negative samples.
        real_train_root (str): Root directory of real training data.
        num_triplets (int): Number of triplets to generate per class.
        save_path (str): Path to save the generated dataset CSV.
        ref_type (str, optional): Type of anchor image in each triplet, either "real" or "synthetic". Defaults to "real".
    Raises:
        ValueError: If both synthetic_pathfile and synthetic_root are provided.
        ValueError: If neither synthetic_pathfile nor synthetic_root are provided.
    """
    
    print()
    print(f"Running on class {cls}: {cls_name}")
    print(f"Using {num_synthetic} synthetic positives to make {num_triplets} triplets.")
    print("Saving to ", save_path)

    if dataset_name == "pods":
        obj = cls_name.split("_")[0]
        negatives_root = os.path.join(negatives_root, f"{obj}_negatives")
        print(f"Getting negatives from {negatives_root}")

    # Get positive and negative paths
    negative_train_paths = get_paths(negatives_root)
    real_train_paths = get_paths(os.path.join(real_train_root, cls_name))

    if synthetic_pathfile is not None and synthetic_root is not None:
        raise ValueError("Cannot use both a pathfile and a root")

    if synthetic_pathfile is not None:
        synthetic_train_paths = load_pathfile(synthetic_pathfile)[cls]
    elif synthetic_root is not None:
        synthetic_train_paths = load_paths_from_root(synthetic_root, cls_name)
    else:
        raise ValueError("No synthetic paths given.")

    # Sample triplets
    random.shuffle(synthetic_train_paths)
    synthetic_train_paths = synthetic_train_paths[:num_synthetic]

    if len(synthetic_train_paths) == 0:
        print("WARNING: 0 synthetic paths, replacing with real")
        synthetic_train_paths = real_train_paths
    elif len(synthetic_train_paths) < num_synthetic:
        print(f"WARNING: {len(synthetic_train_paths)} synthetic but supposed to be at least {num_synthetic}")

    print(f"{len(synthetic_train_paths)} synthetic positives")
    print(f"{len(negative_train_paths)} synthetic negatives")

    csv_dict = []
    train_triplets = []

    print("Making train triplets")
    progress = 0
    while len(train_triplets) < num_triplets:
        negative = get_negative(negative_train_paths)
        positives = get_positives(synthetic_train_paths, real_train_paths, ref_type)
        train_triplets.append(positives + negative)

        new_progress = int(len(train_triplets) * 100 // num_triplets)
        if new_progress % 10 == 0 and new_progress != progress and verbose:
            progress = new_progress
            print(f"{progress} % generated")

    for i in range(num_triplets):
        (ref, left, right), label = sample_triplet(train_triplets[i])
        csv_dict.append({
            "id": i,
            "label": label,
            "ref_path": ref,
            "left_path": left,
            "right_path": right,
        })

    csv = pd.concat([pd.DataFrame(csv_dict)])
    csv.to_csv(save_path, index=False)
    print("done :)")