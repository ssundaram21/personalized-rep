from diffusers import StableDiffusionPipeline
import torch
import os
import argparse
import sys
from util import pidfile
from util.data_util import makedirs, get_paths
from util.constants import IMAGE_EXTENSIONS
from pathlib import Path
import glob
from PIL import Image
import json
import torch.nn as nn
from util.transforms import dreamsim_transform
from downstream.dense_tasks import EmbeddingProcessor, dense_inference_batch
from downstream.perSAM import PerSAM, ModelConfig
from transformers import SamModel
from models.backbones import dinov2_vitb14
import torch.nn.functional as F
import numpy as np
import math

device = "cuda"

def crop(image, mask):
    """
    Crop the image based on the bounding box of the mask.

    Args:
        image (PIL.Image): The original image.
        mask (PIL.Image): The mask image.

    Returns:
        PIL.Image: Cropped image.
    """
    mask = mask.convert('RGB')
    mask_np = np.array(mask)[..., 0]

    # Find the coordinates where the mask is non-zero (object pixels)
    object_coords = np.argwhere(mask_np > 0)
    if object_coords.size == 0:
        raise ValueError("No object found in the mask.")

    # Get the bounding box (min and max row, min and max column)
    top_left = object_coords.min(axis=0)
    bottom_right = object_coords.max(axis=0)
    min_row, min_col = top_left
    max_row, max_col = bottom_right

    # Crop the image using the bounding box
    cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))
    return cropped_image


def filter(cropped_img, train_embeds, embed_model, device, thresh=0.75):
    """
    Filter the cropped image based on similarity with training embeddings.

    Args:
        cropped_img (PIL.Image): The cropped image.
        train_embeds (torch.Tensor): Training embeddings.
        embed_model (torch.nn.Module): Embedding model.
        device (str): Device to run the model on.
        thresh (float): Threshold for similarity score.

    Returns:
        float: Similarity score.
    """
    with torch.no_grad():
        input = dreamsim_transform(cropped_img).unsqueeze(0).to(device)
        embed = embed_model(input)
        embed = F.normalize(embed, dim=1, p=2)
    scores = F.cosine_similarity(train_embeds, embed, dim=1)
    return scores.mean().item()


def disable_safety_check(batch_size=1):
    """
    Disable the safety checker for the pipeline.

    Args:
        batch_size (int): Batch size for the pipeline.

    Returns:
        function: A lambda function that disables the safety checker.
    """
    return lambda images, **kwargs: (images, [False] * batch_size)

def prepare_persam(train_imgs, train_masks):
    """
    Prepare PerSAM model for mask estimation.

    Args:
        train_imgs (list): List of training images.
        train_masks (list): List of training masks.

    Returns:
        tuple: Embedding processor, PerSAM processor, and PerSAM model.
    """
    # Load dinov2 embedding processor for point prompt estimation
    backbone_name = "dinov2_vitb14"
    backbone_model, _ = dinov2_vitb14()
    backbone_model = backbone_model.to(device)
    emb_processor = EmbeddingProcessor(backbone_name=backbone_name, backbone_model=backbone_model, device=device)

    # Prepare PerSAM embedding processor for mask estimation
    backbone_name_sam = "sam-vit-huge"
    backbone_model_sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    persam_processor = EmbeddingProcessor(backbone_name=backbone_name_sam, backbone_model=backbone_model_sam, device=device)
    
    # Prepare and train PerSAM
    config = ModelConfig(device=device)
    persam = PerSAM(backbone_name=backbone_name, config=config)
    train_img = torch.Tensor(np.asarray(Image.open(train_imgs[0]).convert("RGB")))
    embedding_output = emb_processor.compute_embeddings(train_imgs, train_masks)
    persam._train_mask_weights(
        embedding_output.train_embeds[0],
        emb_processor._process_target_feat(embedding_output.train_masked_embeds[0]),
        embedding_output.train_masks[0],
        train_img,
        config
    )
    return emb_processor, persam_processor, persam

def run_dreambooth(args):
    """
    Generate images using StableDiffusionPipeline with DreamBooth and LLM prompts.

    Args:
        args (argparse.Namespace): Arguments for the function.

    Returns:
        None
    """

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    if args.prompts_path is not None:
        with open(args.prompts_path, "r") as f:
            prompt_dict = json.load(f)
            prompts = prompt_dict[args.class_name]
            batch_size = math.ceil(args.n / len(prompts))
    else:
        vanilla_prompt = f"photo of a <new1> {args.class_category}"
        batch_size = args.batch_size
        num_prompts = int(args.n / batch_size)
        prompts = [vanilla_prompt for _ in range(num_prompts)]

    if args.filter:
        from dreamsim import dreamsim
        os.makedirs(args.rej_path, exist_ok=True)

        # Get the training images and masks
        train_paths = glob.glob(os.path.join(args.train_img_folder, args.class_name, '*.jpg'))
        train_mask_paths = glob.glob(os.path.join(args.train_mask_folder, args.class_name, f'*.{args.mask_ext}'))
        train_imgs = [Image.open(x).convert('RGB') for x in train_paths]
        train_masks = [Image.open(x).convert('RGB') for x in train_mask_paths]

        # Load the filtering model
        model, _ = dreamsim(pretrained=True)
        model = model.to(device)
        embed_model = model.embed

        # Crop the training images and compute the embeddings
        cropped_train = []
        for i, train_img in enumerate(train_imgs):
            cropped_train.append(dreamsim_transform(crop(train_img, train_masks[i])))
        input = torch.stack(cropped_train).to(device)
        with torch.no_grad():
            train_embeds = embed_model(input)
            train_embeds = F.normalize(train_embeds, dim=1, p=2)

        # Finetune PerSAM for masks
        emb_processor, persam_processor, persam = prepare_persam(train_paths, train_mask_paths)
        embedding_output = emb_processor.compute_embeddings(train_imgs, train_masks)

        # Generate images
        img_count, accept, rej = 0, 0, 0
        max_rej_save = 15 # Save the first 15 rejected images for reference
        while img_count < args.max_image_count:
            for id, prompt in enumerate(prompts):
                image = pipe(prompt, num_inference_steps=args.inf_steps, guidance_scale=args.guidance, num_images_per_prompt=1).images[0]

                # Extract mask
                pixel_values = emb_processor._process_image(image).to(device)
                batch_embeddings = emb_processor._extract_embeddings(pixel_values)
                
                pixel_values_sam = persam_processor._process_image(image).to(device)
                batch_sam_embeddings = persam_processor._extract_embeddings(pixel_values_sam)
                results = dense_inference_batch(
                    "persam", [image], emb_processor, embedding_output,
                    device, batch_embeddings, persam, batch_sam_embeddings,
                    batch_size=1
                )
                if results[0][0] is None:
                    continue
                mask_image = Image.fromarray(results[0][0])
                cropped_img = crop(image, mask_image)
                
                # Get similarity score
                score = filter(cropped_img, train_embeds, embed_model, device)
                print(score)
                if score >= args.filter_thresh:
                    image.save(
                        f"{args.output_path}/{accept}_{prompt}.jpg"
                    )
                    accept += 1
                    if accept >= args.n:
                        img_count = args.max_image_count + 1
                        break
                else:
                    rej += 1
                    if rej <= max_rej_save:
                        image.save(
                            f"{args.rej_path}/{accept}_{prompt}.jpg"
                        )
                img_count += 1
    else:
        for i, prompt in enumerate(prompts):
            print(f"Generating prompt {prompt}")
            images = pipe(prompt, num_inference_steps=args.inf_steps, guidance_scale=args.guidance, num_images_per_prompt=batch_size).images
            for j, im in enumerate(images):
                im.save(
                    f"{args.output_path}/{i * batch_size + j}_{prompt}.jpg"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="Path to finetuned DreamBooth checkpoint.")
    parser.add_argument("--class_name", type=str, required=True, help="Class name.")
    parser.add_argument("--class_category", type=str, required=True, help="Class semantic category.")
    parser.add_argument("--output_path", type=str, default="./dreambooth_output", help="Directory to save output images.")

    # Run args
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=450)
    parser.add_argument("--inf_steps", type=int, default=50)
    
    # Prompts args
    parser.add_argument("--prompts_path", type=str, default=None)
    
    # Filtering args
    parser.add_argument("--filter", action="store_true", default=False, help="Filter at inference time")
    parser.add_argument("--filter_thresh", type=float, default=0.6)
    parser.add_argument("--mask_ext", type=str, default="jpg")
    parser.add_argument("--max_image_count", type=int, default=5000, help="Max # images to generate before stopping")
    parser.add_argument("--train_img_folder", type=str, help="Directory with training/reference images.")
    parser.add_argument("--train_mask_folder", type=str, help="Directory with masks of the images.")

    parser.add_argument("--debug", action="store_true", default=False, help="debug - if on, doesn't lock the output folder")
    args = parser.parse_args()

    print(f"Using class category {args.class_category}")

    args.model_path = os.path.join(args.model_path, args.class_name)
    method_name = "dreambooth"
    if args.prompts_path is not None:
        method_name += "_llm"
    if "masked" in args.model_path:
        method_name += "_masked"
    if args.filter:
        method_name += "_filtered_dinov2"
    
    args.rej_path = os.path.join(args.output_path, f"{method_name}_sd1.5", f"cfg_{args.guidance}", f"{args.class_name}_rej")
    args.output_path = os.path.join(args.output_path, f"{method_name}_sd1.5", f"cfg_{args.guidance}", args.class_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    pidfile.exit_if_job_done(args.output_path)
    existing_files = list(Path(args.output_path).rglob('*.jpg'))
    existing = len(existing_files)
    if existing >= args.n:
        print(f"{existing}, already done.")
        sys.exit()

    args.n = args.n - existing
    args.offset = existing
    
    print(f"Generating {args.n} images starting from id {args.offset}")

    run_dreambooth(args)

    pidfile.mark_job_done(args.output_path)