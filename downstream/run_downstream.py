import os
import json
import configargparse
from util.misc import set_all_seeds
from util.data_util import makedirs
from util import pidfile
from models.backbones import ModelFactory
from downstream.global_tasks import run_global_tasks
from downstream.dense_tasks import run_dense_tasks
import torch

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path')

    # Run settings
    parser.add_argument("--seed", type=int, default=1234, help="Single random seed for all randomness.")
    parser.add_argument("--debug", action="store_true", default=False, help="If on, the experiment does not lock output folder")

    # Save paths
    parser.add_argument("--output_path", type=str, default="./outputs", help="Path to save outputs")
    parser.add_argument("--embed_path", type=str, default="./embeds", help="Path to cache computed embeddings")

    # Data
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--dataset_info", type=str, help="(optional) Path to dataset metadata file, containing a class-to-id mapping.")
    parser.add_argument("--pos_class_name", type=str, default=None, help="Positive class name.")
    parser.add_argument("--real_data_root", type=str, default="./real", 
                        help="Root for real data containing train, test, and mask folders (See the dataset README for details on dataset setup.)")
    parser.add_argument("--mask_ext", type=str, default="jpg", help="File extension for masks.")

    # Model
    parser.add_argument("--eval_models", type=str, default="dinov2_vitb14", help="Models to test.")
    parser.add_argument("--eval_epoch", type=int, default=2, help="Epoch to evaluate at (if passing in a checkpoint)")
    parser.add_argument("--checkpoint_dir", type=str, default="Path to a checkpoint root.")
    parser.add_argument('--cache_dir', type=str, default="./cache", help='path to pretrained ViT checkpoints')

    # Downstream eval setup
    parser.add_argument("--downstream_tasks", type=str, default="global_tasks,dense_tasks")
    parser.add_argument("--downstream_batch_size", type=int, default=1, help="Batch size for computing embeddings")
    parser.add_argument("--downstream_workers", type=int, default=10)
    parser.add_argument("--persam", action="store_true")

    return parser.parse_args()

def run_downstream(args, device, class_info):
    """
    Run downstream tasks based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        device (torch.device): Device to run the model on.
        class_info (dict): Dictionary containing class information.
    """
    print()
    makedirs(args.embed_path)
    set_all_seeds(args.seed)

    output_path = os.path.join(args.output_path, f"epoch_{args.eval_epoch}_{args.downstream_task}_outputs")
    if args.persam:
        output_path = os.path.join(args.output_path, f"epoch_{args.eval_epoch}_{args.downstream_task}_persam_outputs")
    makedirs(output_path)
    if not args.debug:
        if pidfile.check_if_job_done(output_path):
            return

    all_models = args.eval_models.split(",")
    all_ckpts = args.checkpoint_dir.split(",") if len(args.checkpoint_dir) > 0 else []
    ckpt_index = 0

    all_results = {}
    model_factory = ModelFactory(args.cache_dir, device)
    for model_name in all_models:
        print("Using model ", model_name)
        ckpt_dir = all_ckpts[ckpt_index] if "custom" in model_name else None
        model, preprocess = model_factory.load(model_name, checkpoint=ckpt_dir, load_epoch=args.eval_epoch)
        model_fn = model_factory.get_global_fn(model, model_name)

        args.real_train_root = f"{args.real_data_root}/train"
        args.class_id = class_info["class_to_idx"][args.pos_class_name]
        if args.downstream_task == "global_tasks":
            args.test_root = f"{args.real_data_root}/test"
            result = run_global_tasks(args, model_fn, model_name, preprocess, class_info["class_to_idx"], device)
        elif args.downstream_task == "dense_tasks":
            if args.dataset == "pods":
                args.test_root = f"{args.real_data_root}/test_dense"
            else:
                args.test_root = f"{args.real_data_root}/test"
            result = run_dense_tasks(args, model, model_name, preprocess, args.class_id, args.pos_class_name, class_info["class_to_idx"], device, args.persam)

        all_results[model_name] = result
        if "custom" in model_name:
            ckpt_index += 1

    with open(os.path.join(output_path, f"results.json"), "w") as f:
        json.dump(all_results, f)

    if not args.debug:
        pidfile.mark_job_done(output_path)

    print("done :)")

if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.dataset_info:
        with open(args.dataset_info, "r") as f:
            class_info = json.load(f)
            class_info['class_to_idx'] = {k: int(v) for k, v in class_info['class_to_idx'].items()}
            class_info['idx_to_class'] = {int(v): k for k, v in class_info['class_to_idx'].items()}
    else:
        print("Extracting class information from data path")
        args.real_train_root = f"{args.real_data_root}/train"
        classes = os.listdir(args.real_train_root)
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        class_info = {
            'classes': classes,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    downstream_tasks = args.downstream_tasks.split(',') 
    for task in downstream_tasks:
        args.downstream_task = task 
        print(f"Running {args.downstream_task} at epoch {args.eval_epoch}")
        run_downstream(args, device, class_info)