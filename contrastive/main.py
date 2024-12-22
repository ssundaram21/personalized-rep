import configargparse
import torch
from contrastive.make_triplets import setup_triplets
from contrastive.train_custom import train
from downstream.run_downstream import run_downstream
from util.misc import (
    make_exp_name, make_triplet_dirname, make_checkpoint_dirname, 
    checkpoint_exists, training_in_progress, set_all_seeds
)
from util.data_util import makedirs
import json
import os
import copy
import sys

def parse_args():
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

    ### Run options
    parser.add_argument('--seeds', type=str, default=None, help='String-separated list of random seeds to sweep over. Overrides args.seed.')
    parser.add_argument('--seed', type=int, default=1234, help='Single random seed for all randomness.')
    parser.add_argument('--tag', type=str, default='', help='Experiment name (prepended to output directory).')
    parser.add_argument("--debug", action="store_true", default=False, help="If on, the experiment does not lock output folder")

    ### Shared settings
    # Save paths
    parser.add_argument('--output_path', type=str, default="./outputs", help='Path to save model checkpoints, logs, and results.')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='Path to cache pretrained ViT checkpoints')
    parser.add_argument("--embed_path", type=str, default="./embeds",  help="Path to cache computed embeddings")
    
    # Dataset 
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--dataset_info", type=str, help="(optional) Path to dataset metadata file, containing a class-to-id mapping.")
    parser.add_argument("--pos_class_name", type=str, help="Positive class name to run the pipeline on. The personalized model will be trained on this class.")
    parser.add_argument("--real_data_root", type=str, default="./real",
                        help="Root for real data containing train, test, and mask folders (See the dataset README for details on dataset setup.)")

    # Mask dataset information - for Dense Evaluation
    parser.add_argument("--mask_ext", type=str, default="jpg", help="File extension for masks.")

    ### Contrastive learning options
    # Dataset
    parser.add_argument("--synthetic_train_pathfile", type=str, default=None, 
                        help="Pathfile for synthetic data: A JSON file mapping each class to a list of image paths (i.e., {class_id: [path list], ...}).")
    parser.add_argument("--synthetic_train_root", type=str, default=None, 
                        help="Root for synthetic data (see dataset README for expected structure). Cannot pass both a synthetic pathfile and a synthetic root.")
    parser.add_argument("--negatives_root", type=str, default=None, help="Root for negative images.") 
    
    # Triplet Composition
    parser.add_argument("--use_existing_triplets", action="store_true", help="Reuses existing triplet dataset (if it exists)")
    parser.add_argument("--num_synthetic", type=int, default=None, help="Number of synthetic positives to draw triplets from.")
    parser.add_argument("--num_triplets", type=int, default=5000, help="Number of triplets to sample.")
    parser.add_argument("--ref_type", type=str, default="real", help="Anchor type for triplets: one of {real, synthetic, both}")

    # Model
    parser.add_argument('--train_model_type', type=str, default='dinov2_vitb14',
                        help='Which ViT model to finetune. Accepted models: [dinov2_vitb14, clip_vitb16, mae_vitb16]')
    parser.add_argument('--use_lora', action="store_true",
                        help='Whether to train with LoRA finetuning [True] or with an MLP head [False].')
    parser.add_argument('--patch_tokens', action="store_true",
                        help='Whether to train with CLS and avg pooled patch tokens [TRUE] or just CLS token [FALSE].')
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of the MLP hidden layer.')

    # Training
    parser.add_argument("--train_use_existing_checkpoint", action="store_true", help="Reuses existing checkpoint (if it exists)")
    parser.add_argument('--train_lr', type=float, default=0.0003, help='Learning rate for training.')
    parser.add_argument('--train_loss_fn', type=str, default="info_nce_fixed", help='Loss function for training.')
    parser.add_argument('--train_weight_decay', type=float, default=0.0, help='Weight decay for training.')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Dataset batch size.')
    parser.add_argument('--train_epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--train_margin', default=0.05, type=float, help='Margin for hinge loss')
    parser.add_argument('--train_lora_r', default=16, type=int)
    parser.add_argument('--train_lora_alpha', default=0.5, type=float)
    parser.add_argument('--train_lora_dropout', default=0.3, type=float)
    parser.add_argument('--train_num_workers', type=int, default=16)
    parser.add_argument('--train_augment', action="store_true")

    ### Evaluation options
    # Downstream task
    parser.add_argument("--downstream_tasks", type=str, default=None)
    parser.add_argument("--persam", action="store_true")

    # Model
    parser.add_argument("--eval_models", type=str, default="dinov2_vitb14", help="Models to test.")
    parser.add_argument("--eval_epoch", type=str, default="2", help="Epochs to evaluate trained models at -- can be comma separated if multiple epochs are being evaluated")
    
    # Experiment setup
    parser.add_argument("--downstream_batch_size", type=int, default=2, help="Batch size for computing embeddings")
    parser.add_argument("--downstream_workers", type=int, default=10)

    return parser.parse_args()

def main(args, device):
    """
    Main function to execute the training and evaluation pipeline.

    Args:
        args (Namespace): Parsed command-line arguments containing configuration for the experiment.
        device (torch.device): The device (CPU or GPU) to run the computations on.
        
    The function performs the following tasks:
        - Sets up the experiment directory structure and loads dataset information from a JSON file.
        - Generates or loads triplet datasets.
        - Finetunes a personalization representation if no existing checkpoint is found.
        - Runs downstream tasks using the trained model.

    Raises:
        Exception: If there is an error changing the permissions of the configuration file.
    """
    
    # Set seed for reproducibility
    print(f"\nSeeding with {args.seed}")
    set_all_seeds(args.seed)
    
    # Setup paths and save configuration
    args.exp_name = make_exp_name(args)
    args.output_path = os.path.join(args.output_path, args.exp_name, str(args.pos_class_name))
    makedirs(args.output_path)
    with open(os.path.join(args.output_path, "config.json"), "w") as f:
        json.dump(vars(args), f)
    
    # Load dataset information and paths
    args.real_train_root = f"{args.real_data_root}/train"
    if args.dataset_info:
        with open(args.dataset_info, "r") as f:
            class_info = json.load(f)
            class_info['class_to_idx'] = {k: int(v) for k, v in class_info['class_to_idx'].items()}
            class_info['idx_to_class'] = {int(v): k for k, v in class_info['class_to_idx'].items()}
    else:
        print("Extracting class information from data path")
        classes = os.listdir(args.real_train_root)
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        class_info = {
            'classes': classes,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }

    # Check for existing triplet dataset CSV, otherwise create one
    print("Checking for existing triplets")
    triplet_save_root = make_triplet_dirname(args)
    makedirs(triplet_save_root)
    args.class_id = class_info['class_to_idx'][args.pos_class_name]

    args.dataset_csv = os.path.join(triplet_save_root, f"{args.pos_class_name}.csv")
    if not os.path.exists(args.dataset_csv) or not args.use_existing_triplets:
        print("Generating triplets")
        setup_triplets(
            args.class_id, 
            args.pos_class_name,
            args.dataset,
            args.synthetic_train_pathfile,
            args.synthetic_train_root,
            args.num_synthetic,
            args.negatives_root, 
            args.real_train_root,
            args.num_triplets,
            args.dataset_csv,
            args.ref_type)
    else:
        print("Found existing triplets")
    
    # Check for existing trained model checkpoint, otherwise train a new one
    print("Checking for existing checkpoint")
    args.ckpt_dir = make_checkpoint_dirname(args)
    eval_epoch_list = args.eval_epoch.split(',')
    if args.downstream_tasks is not None:
        downstream_tasks = args.downstream_tasks.split(',') 
    else:
        downstream_tasks = None

    for eval_epoch in eval_epoch_list:
        eval_epoch = int(eval_epoch)

        # Check if training is in progress
        if training_in_progress(args.ckpt_dir) and not args.debug:
            print("Training in progress; exiting")
            sys.exit(0)

        print(f"Checking for a checkpoint for epoch {eval_epoch}")
        ckpt_exists, ckpt_version_dir = checkpoint_exists(args.ckpt_dir, eval_epoch)
        
        if not ckpt_exists or not args.train_use_existing_checkpoint:
            # Train new personalized model
            print("Training custom model")
            makedirs(args.ckpt_dir)
            train(args, device)
            ckpt_exists, ckpt_version_dir = checkpoint_exists(args.ckpt_dir, eval_epoch)
        else:
            print(f"Found existing checkpoint at {ckpt_version_dir}")
    
        # Run downstream tasks for this class/evaluation epoch
        assert ckpt_exists
        args.eval_epoch = eval_epoch
        args.checkpoint_dir = ckpt_version_dir
        print(args.checkpoint_dir)
        if downstream_tasks:
            for task in downstream_tasks:
                args.downstream_task = task 
                print(f"Running {args.downstream_task} at epoch {args.eval_epoch}")
                run_downstream(args, device, class_info)
        else:
            print("Skipping evaluation")
            
    print("All done :)")

if __name__ == '__main__':
    args = parse_args()
    print("Arguments: ", vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seeds is not None:
        all_seeds = [int(x) for x in args.seeds.split(',')]
        print(f"Running with {len(all_seeds)} seeds")
        original_args = copy.deepcopy(args)
        for seed in all_seeds:
            args = copy.deepcopy(original_args)
            args.seed = seed
            print(f"Running with seed {args.seed}")
            main(args, device)
    else:
        main(args, device)
