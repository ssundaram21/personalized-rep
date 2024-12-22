import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image
from typing import Tuple, List
from torch import nn

from transformers import AutoProcessor, SamModel
from PIL import Image
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, binary_dilation, binary_erosion, disk
from dataclasses import dataclass
from tqdm import tqdm

import matplotlib.pyplot as plt

from models.load_clip_as_dino import load_clip_as_dino
from downstream.perSAM import PerSAM, ModelConfig

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from collections import defaultdict

from util.data_util import get_paths
import os
from dataset.image_dataset import EvalDataset

@dataclass
class EmbeddingOutput:
    """Container for embedding computation results."""
    train_embeds: torch.Tensor
    train_masked_embeds: torch.Tensor
    train_bbox_masked_embeds: torch.Tensor
    train_masks: torch.Tensor

class EmbeddingProcessor:
    """Handles computation of patch embeddings and masked patch embeddings."""
    
    def __init__(self, backbone_name: str, backbone_model, device: str = "cuda"):
        """
        Initialize the EmbeddingProcessor.

        Args:
            backbone_name (str): Name of the backbone model.
            backbone_model: Backbone model instance.
            device (str): Device to run the model on (default: "cuda").
        """
        if backbone_name == 'clip_vitb16':
            backbone_model, _ = load_clip_as_dino(16, "./models")
            self.backbone_model = backbone_model.to(device)
        else:
            self.backbone_model = backbone_model
        self.device = device
        self.backbone_name = backbone_name
        self.processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")

    def load_image(self, image_input):
        """
        Load image from file path or PIL image.

        Args:
            image_input: File path (str) or PIL.Image.Image instance.

        Returns:
            PIL.Image.Image: Loaded image.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Input must be a file path (str) or a PIL.Image.Image instance.")
        return image
    
    def _process_image(self, image_input) -> torch.Tensor:
        """
        Process single image and return resized pixel values.

        Args:
            image_input: File path (str) or PIL.Image.Image instance.

        Returns:
            torch.Tensor: Resized pixel values.
        """
        image = self.load_image(image_input)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        size = (896, 896) if "vitb14" in self.backbone_name else (1024, 1024)
        return F.interpolate(inputs.pixel_values, size=size, mode="bilinear")

    def _extract_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings based on backbone model.

        Args:
            pixel_values (torch.Tensor): Pixel values of the image.

        Returns:
            torch.Tensor: Extracted embeddings.
        """
        batch_size = pixel_values.shape[0]
        with torch.no_grad():
            if "sam" in self.backbone_name:
                embeddings = self.backbone_model.get_image_embeddings(pixel_values)
            else:
                if "custom_lora" in self.backbone_name:
                    embeddings = self.backbone_model.model.model.get_intermediate_layers(pixel_values)[0].permute(0, 2, 1)
                else:
                    embeddings = self.backbone_model.get_intermediate_layers(pixel_values)[0].permute(0, 2, 1)
                if embeddings.shape[-1] % 2 == 1:  # cls token is attached
                    embeddings = embeddings[:, :, 1:]  # slice off only the patch token 
                embeddings = embeddings.reshape(batch_size, -1, 64, 64)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.squeeze()
            
    def _process_mask(self, mask_input) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process mask file.

        Args:
            mask_input: File path (str) or PIL.Image.Image instance.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Ground truth mask, processed mask, and bounding box mask tensor.
        """
        mask_img = self.load_image(mask_input)
        mask_tensor = (torch.tensor(np.asarray(mask_img))[:, :, 0] > 0).float()
        gt_mask = mask_tensor.float().unsqueeze(0).flatten(1)
        processed_mask = prepare_mask(np.asarray(mask_img))

        bbox = extract_bounding_box(mask_tensor)
        bbox_mask_tensor = torch.Tensor(create_bbox_mask(bbox, mask_tensor)[0][0])
        return gt_mask, processed_mask, bbox_mask_tensor

    def _process_masked_embeddings(self, embeddings, mask):
        """
        Extract masked embeddings.
        
        Args:
            embeddings (torch.Tensor): The embeddings tensor.
            mask (torch.Tensor): The mask tensor.
        
        Returns:
            torch.Tensor: Masked embeddings.
        """
        embeddings = embeddings.permute(1, 2, 0)
        if mask.ndim == 2:
            ref_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear").squeeze()
        else:
            ref_mask = F.interpolate(mask, size=(64, 64), mode="bilinear").squeeze()[0]
        masked_feat = embeddings[ref_mask > 0]
        return masked_feat

    def _process_target_feat(self, masked_feat):
        """
        Generate target features.
        
        Args:
            masked_feat (torch.Tensor): Masked features tensor.
        
        Returns:
            torch.Tensor: Normalized target features.
        """
        if "mae" in self.backbone_name:
            target_feat = masked_feat.mean(0)
        else:
            target_feat_mean = masked_feat.mean(0)
            target_feat_max = torch.max(masked_feat, dim=0)[0]
            target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
        target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
        return target_feat
    
    def compute_embeddings(
        self,
        image_paths: List[str],
        mask_paths: List[str]
    ) -> EmbeddingOutput:
        """
        Compute embeddings for all images and target features from the first image.
        
        Args:
            image_paths (List[str]): List of image file paths.
            mask_paths (List[str]): List of mask file paths.
        
        Returns:
            EmbeddingOutput: Container with computed embeddings and masks.
        """
        print("Computing embeddings...")
        embeddings_list, masked_embeddings_list, bbox_masked_embeddings_list = [], [], []
        masks_list = []
        
        # Process all images including the first one
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths)):
            pixel_values = self._process_image(img_path)
            embeddings = self._extract_embeddings(pixel_values)
            mask, processed_mask, bbox_mask = self._process_mask(mask_path)
            masked_embed = self._process_masked_embeddings(embeddings, processed_mask)
            bbox_masked_embed = self._process_masked_embeddings(embeddings, bbox_mask)
            
            embeddings_list.append(embeddings.detach().cpu())
            masked_embeddings_list.append(masked_embed.detach().cpu())
            bbox_masked_embeddings_list.append(bbox_masked_embed.detach().cpu())
            masks_list.append(mask.detach().cpu())
                
        return EmbeddingOutput(
            train_embeds=torch.stack(embeddings_list),
            train_masked_embeds=masked_embeddings_list,
            train_bbox_masked_embeds=bbox_masked_embeddings_list,
            train_masks=masks_list,
        )

class EvalContainer:
    """Container for evaluation, logging, and formatting of predictions."""

    def __init__(self):
        self.COCO = COCO
        self.COCOeval = COCOeval
        self.np = np
        self.gt_seg, self.gt_det = [], []
        self.thresh_seg, self.thresh_det, self.persam_seg, self.persam_det = [], [], [], []

    def calculate_mAP(self, ground_truths, detections, iou_type='bbox', iou_thresholds=None):
        """
        Calculate mAP and F1 scores using COCO metrics.

        Args:
            ground_truths (list): List of ground truth annotations.
            detections (list): List of detection annotations.
            iou_type (str): Type of IoU ('bbox' or 'segm').
            iou_thresholds (list, optional): List of IoU thresholds.

        Returns:
            tuple: mAP, mAP50, F1 score at IoU 50, and average F1 score over IoU 50:95.
        """
        coco_gt = self._create_coco_gt(ground_truths)
        coco_dt = coco_gt.loadRes(detections)
        coco_eval = self._evaluate_coco(coco_gt, coco_dt, iou_type, iou_thresholds)

        mAP, mAP50 = coco_eval.stats[0], coco_eval.stats[1]
        f1_at_iou50, f1_avg_iou50_95 = self._calculate_f1_score(coco_eval)
        return mAP, mAP50, f1_at_iou50, f1_avg_iou50_95

    def _create_coco_gt(self, ground_truths):
        """
        Create a COCO ground truth object.

        Args:
            ground_truths (list): List of ground truth annotations.

        Returns:
            COCO: COCO ground truth object.
        """
        coco_gt = self.COCO()
        coco_gt.dataset = {
            'images': [{'id': g['image_id'], 'height': g['height'], 'width': g['width']} for g in ground_truths],
            'annotations': ground_truths,
            'categories': [{'id': 1, 'name': 'object'}]
        }
        coco_gt.createIndex()
        return coco_gt

    def _evaluate_coco(self, coco_gt, coco_dt, iou_type, iou_thresholds):
        """
        Run COCO evaluation and set thresholds if provided.

        Args:
            coco_gt (COCO): COCO ground truth object.
            coco_dt (COCO): COCO detection object.
            iou_type (str): Type of IoU ('bbox' or 'segm').
            iou_thresholds (list, optional): List of IoU thresholds.

        Returns:
            COCOeval: COCO evaluation object.
        """
        coco_eval = self.COCOeval(coco_gt, coco_dt, iouType=iou_type)
        if iou_thresholds:
            coco_eval.params.iouThrs = self.np.array(iou_thresholds)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval

    def _calculate_f1_score(self, coco_eval):
        """
        Calculate F1 scores at IoU 50 and averaged over 50:95.

        Args:
            coco_eval (COCOeval): COCO evaluation object.

        Returns:
            tuple: F1 score at IoU 50 and average F1 score over IoU 50:95.
        """
        precision, recall = self._nan_safe(coco_eval.eval['precision']), self._nan_safe(coco_eval.eval['recall'])
        iou50_idx = self.np.where(coco_eval.params.iouThrs == 0.5)[0][0]
        f1_at_iou50 = self._mean_f1_score(precision[iou50_idx], recall[iou50_idx], axis=(1, 2, 3))
        f1_scores = []
        for t in range(precision.shape[0]):  # For each IoU threshold
            f1_t = self._mean_f1_score(precision[t], recall[t], axis=(1, 2, 3))
            f1_scores.append(f1_t)
        f1_avg_iou50_95 = float(self.np.mean(f1_scores))
        return f1_at_iou50, f1_avg_iou50_95

    def _nan_safe(self, array):
        """
        Replace -1 with NaN for safe averaging.

        Args:
            array (np.ndarray): Input array.

        Returns:
            np.ndarray: Array with -1 replaced by NaN.
        """
        return self.np.where(array == -1, self.np.nan, array)

    def _mean_f1_score(self, precision, recall, axis=None):
        """
        Compute mean F1 score.

        Args:
            precision (np.ndarray): Precision values.
            recall (np.ndarray): Recall values.
            axis (tuple, optional): Axis for averaging.

        Returns:
            float: Mean F1 score.
        """
        if precision.ndim > recall.ndim:
            recall = self.np.expand_dims(recall, axis=0)  # Add R dimension
            recall = self.np.broadcast_to(recall, precision.shape)
        avg_precision = self.np.nanmean(precision, axis=axis)
        avg_recall = self.np.nanmean(recall, axis=axis)
        return float(self.np.mean(2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)))

    def finalize_results(self):
        """
        Finalize and return results for segmentation and detection.

        Returns:
            dict: Results for segmentation and detection.
        """
        results = {}
        if self.thresh_seg:
            results["threshold_segmentation"] = self.evaluate_and_log(self.gt_seg, self.thresh_seg, "Segmentation (Threshold)", 'segm')
        if self.persam_seg:
            results["persam_segmentation"] = self.evaluate_and_log(self.gt_seg, self.persam_seg, "Segmentation (PerSAM)", 'segm')
        if self.thresh_det:
            results["threshold_detection"] = self.evaluate_and_log(self.gt_det, self.thresh_det, "Detection (Threshold)", 'bbox', True)
        if self.persam_det:
            results["persam_detection"] = self.evaluate_and_log(self.gt_det, self.persam_det, "Detection (PerSAM)", 'bbox', True)
        return results

    def evaluate_and_log(self, ground_truths, predictions, title, iou_type='bbox', save_preds=False):
        """
        Evaluate and log results, optionally saving predictions.

        Args:
            ground_truths (list): List of ground truth annotations.
            predictions (list): List of prediction annotations.
            title (str): Title for logging.
            iou_type (str): Type of IoU ('bbox' or 'segm').
            save_preds (bool, optional): Whether to save predictions.

        Returns:
            dict: Evaluation results.
        """
        print(title)
        mAP, mAP50, f1_at_iou50, f1_avg_iou50_95 = self.calculate_mAP(ground_truths, predictions, iou_type)
        print(f"mAP: {mAP} mAP50: {mAP50} f1: {f1_avg_iou50_95} f1_50: {f1_at_iou50}")

        result = {"mAP": mAP, "mAP50": mAP50, "f1": f1_avg_iou50_95, "f1_50": f1_at_iou50}
        if save_preds:
            result["preds"] = self.format_predictions(predictions)
        return result

    def format_predictions(self, predictions):
        """
        Standardize predictions with float bbox coordinates.

        Args:
            predictions (list): List of prediction annotations.

        Returns:
            list: Formatted predictions.
        """
        return [{'bbox': [float(x) for x in pred['bbox']], 'score': pred['score'], 'path': pred['path']} for pred in predictions]

    def process_ground_truth(self, img_id, gt_mask, label, positive_cls, height, width):
        """
        Store ground truth for segmentation and detection tasks.

        Args:
            img_id (int): Image ID.
            gt_mask (np.ndarray): Ground truth mask.
            label (int): Label of the image.
            positive_cls (int): Positive class label.
            height (int): Height of the image.
            width (int): Width of the image.
        """
        height, width = gt_mask.shape 
        gt_seg, gt_area_seg = extract_segmentation_mask(gt_mask)
        gt_det = extract_bounding_box(gt_mask)
        gt_area_det = gt_det[2] * gt_det[3]
        
        self.gt_seg.append(self.create_ground_truth(img_id, gt_seg, gt_area_seg, label, positive_cls, height, width, segmentation=True))
        self.gt_det.append(self.create_ground_truth(img_id, gt_det, gt_area_det, label, positive_cls, height, width, segmentation=False))

    def process_predictions(self, inference_mode, img_id, img_path, pred_mask, pred_bbox, score, height, width):
        """
        Process predictions based on presence of bounding box.

        Args:
            inference_mode (str): Inference mode ('threshold' or 'persam').
            img_id (int): Image ID.
            img_path (str): Path to the image.
            pred_mask (np.ndarray): Predicted mask.
            pred_bbox (list): Predicted bounding box.
            score (float): Confidence score.
            height (int): Height of the image.
            width (int): Width of the image.
        """
        if pred_bbox is None:
            self.store_negative_predictions(inference_mode, img_id, img_path, height, width)
        else:
            self.store_positive_predictions(inference_mode, img_id, img_path, pred_mask, pred_bbox, score, height, width)

    def store_negative_predictions(self, inference_mode, img_id, img_path, height, width):
        """
        Store negative predictions with zero score.

        Args:
            inference_mode (str): Inference mode ('threshold' or 'persam').
            img_id (int): Image ID.
            img_path (str): Path to the image.
            height (int): Height of the image.
            width (int): Width of the image.
        """
        empty_seg, _ = extract_segmentation_mask([[0, 0]])
        if inference_mode == "threshold":
            self.thresh_seg.append(self.create_prediction(img_id,  empty_seg, img_path, 0, height, width, True, True))
            self.thresh_det.append(self.create_prediction(img_id, [0, 0, 0, 0], img_path, 0, height, width, True, False))
        elif inference_mode == "persam":
            self.persam_seg.append(self.create_prediction(img_id, empty_seg, img_path, 0, height, width, True, True))
            self.persam_det.append(self.create_prediction(img_id, [0, 0, 0, 0], img_path, 0, height, width, True, False))
        else:
            print("Inference mode not supported")

    def store_positive_predictions(self, inference_mode, img_id, img_path, pred_mask, pred_bbox, score, height, width):
        """
        Store positive predictions with given scores.

        Args:
            inference_mode (str): Inference mode ('threshold' or 'persam').
            img_id (int): Image ID.
            img_path (str): Path to the image.
            pred_mask (np.ndarray): Predicted mask.
            pred_bbox (list): Predicted bounding box.
            score (float): Confidence score.
            height (int): Height of the image.
            width (int): Width of the image.
        """
        pred_mask, _ = extract_segmentation_mask(pred_mask)
        if inference_mode == "threshold":
            self.thresh_seg.append(self.create_prediction(img_id, pred_mask, img_path, score, height, width, False, True))
            self.thresh_det.append(self.create_prediction(img_id, pred_bbox, img_path, score, height, width, False, False))
        elif inference_mode == "persam":
            self.persam_seg.append(self.create_prediction(img_id, pred_mask, img_path, score, height, width, False, True))
            self.persam_det.append(self.create_prediction(img_id, pred_bbox, img_path, score, height, width, False, False))
        else:
            print("Inference mode not supported")

    def create_ground_truth(self, img_id, gt, area, label, positive_cls, height, width, segmentation=False):
        """
        Format ground truth data.

        Args:
            img_id (int): Image ID.
            gt (list): Ground truth annotation.
            area (float): Area of the ground truth.
            label (int): Label of the image.
            positive_cls (int): Positive class label.
            height (int): Height of the image.
            width (int): Width of the image.
            segmentation (bool, optional): Whether the annotation is for segmentation.

        Returns:
            dict: Formatted ground truth annotation.
        """
        if segmentation:
            annot = "segmentation"
            annotation = gt if (label == positive_cls) else [[0, 0]]  
        else:
            annot = "bbox"
            annotation = gt if (label == positive_cls) else [0, 0, 0, 0]
    
        return {
            'id': img_id,
            'image_id': img_id,
            f'{annot}': annotation,
            'iscrowd': 0,
            'area': area,
            'height': height,
            'width': width,
            'category_id': int(label == positive_cls)  # Ground truth class
        }

    def create_prediction(self, img_id, pred, path, confidence, height, width, negative=False, segmentation=False):
        """
        Create a formatted prediction dictionary.

        Args:
            img_id (int): Image ID.
            pred (list): Prediction annotation.
            path (str): Path to the image.
            confidence (float): Confidence score.
            height (int): Height of the image.
            width (int): Width of the image.
            negative (bool, optional): Whether the prediction is negative.
            segmentation (bool, optional): Whether the annotation is for segmentation.

        Returns:
            dict: Formatted prediction annotation.
        """
        annot = "segmentation" if segmentation else "bbox"
        return {
            'id': img_id,
            'image_id': img_id,
            annot: pred,
            'score': 0 if negative else confidence.item(),
            'height': height,
            'width': width,
            'path': path,
            'category_id': 0 if negative else 1
        }
        
def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.

    Args:
        oldh (int): Original height of the image.
        oldw (int): Original width of the image.
        long_side_length (int): Target length of the long side.

    Returns:
        Tuple[int, int]: New height and width of the image.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def preprocess(x: torch.Tensor, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], img_size=1024) -> torch.Tensor:
    """
    Normalize pixel values and pad to a square input.

    Args:
        x (torch.Tensor): Input image tensor.
        pixel_mean (list): Mean pixel values for normalization.
        pixel_std (list): Standard deviation of pixel values for normalization.
        img_size (int): Target image size after padding.

    Returns:
        torch.Tensor: Normalized and padded image tensor.
    """
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def prepare_mask(image, target_length=1024):
    """
    Prepares a mask from the given image by resizing it to the target length and preprocessing it.

    Args:
        image (numpy.ndarray): The input image to be processed.
        target_length (int, optional): The target length for resizing the image. Defaults to 1024.
        
    Returns:
        torch.Tensor: The preprocessed mask tensor.
    """
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    mask = np.array(resize(to_pil_image(image), target_size))

    input_mask = torch.as_tensor(mask)
    input_mask = input_mask.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_mask = preprocess(input_mask)

    return input_mask

def extract_segmentation_mask(mask_image, threshold=0.5):
    """
    Convert a mask image to the proper form for segmentation predictions.
    
    Parameters:
    - mask_image: numpy array of shape (H, W) or (H, W, C)
                  Values are expected to be in the range [0, 1] or [0, 255]
    - threshold: float, threshold to binarize the mask (default: 0.5)
    
    Returns:
    - A dictionary containing the processed mask in RLE format and its area
    """
    if not np.any(mask_image):
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        rle = mask_utils.encode(np.asfortranarray(empty_mask))
        return rle, 0
        
    # Ensure mask_image is 2D
    if mask_image.ndim == 3:
        mask_image = mask_image.mean(axis=2)
    
    # Normalize to [0, 1] if necessary
    if mask_image.max() > 1:
        mask_image = mask_image / 255.0
    
    # Threshold the mask
    binary_mask = (mask_image > threshold).astype(np.uint8)
    
    # Find contours to remove small noise and holes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Keep only the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(binary_mask)
        cv2.drawContours(clean_mask, [largest_contour], 0, 1, -1)
    else:
        clean_mask = binary_mask
    
    # Encode mask
    rle = mask_utils.encode(np.asfortranarray(clean_mask))

    # visualize 
    binary_mask = mask_utils.decode(rle).astype(np.uint8)

    # Compute area
    area = float(clean_mask.sum())
    return rle, area

def get_thresholding_mask(sim):
    """
    Apply thresholding to the similarity map to generate a binary mask.

    Args:
        sim (torch.Tensor): Similarity map.

    Returns:
        Tuple[np.ndarray, float]: Refined binary mask and threshold value.
    """
    sim = sim.cpu().numpy()
    normalized_mask = (sim - sim.min()) / (sim.max() - sim.min())
    threshold = threshold_otsu(normalized_mask)
    binary_mask = normalized_mask > threshold
    refined_mask = remove_small_objects(binary_mask, min_size=100)
    refined_mask = binary_closing(refined_mask)
    selem = disk(1)  # Adjust the radius as needed 
    refined_mask = binary_dilation(refined_mask, selem)
    refined_mask = binary_erosion(refined_mask, selem)
    return refined_mask, threshold

def load_ground_truth_mask(mask_path: str) -> Tuple[np.ndarray, int, int]:
    """
    Load ground truth mask and return it with image dimensions.

    Args:
        mask_path (str): Path to the mask file.

    Returns:
        Tuple[np.ndarray, int, int]: Ground truth mask and its dimensions (height, width).
    """
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    return mask, *mask.shape

def create_bbox_mask(bbox, mask):
    """
    Create a bounding box mask from the given bounding box coordinates.

    Args:
        bbox (list): Bounding box coordinates [xmin, ymin, width, height].
        mask (np.ndarray): Original mask.

    Returns:
        torch.Tensor: Resized bounding box mask tensor.
    """
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height
    bbox_mask = np.zeros_like(mask)
    bbox_mask[ymin:ymax+1, xmin:xmax+1] = 1
    return F.interpolate(torch.tensor(bbox_mask).float().unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")

def extract_bounding_box(mask):
    """
    Compute bounding box from the given mask.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        list: Bounding box coordinates [xmin, ymin, width, height] or None if no bounding box is found.
    """
    if mask is None:
        return None
    elif mask.shape == () and mask.item() is None:
        return None
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    xmin, xmax = x_indices.min(), x_indices.max()
    ymin, ymax = y_indices.min(), y_indices.max()
    return [xmin, ymin, xmax - xmin, ymax - ymin]

@torch.no_grad()
def dense_inference_batch(
    inference_mode, test_imgs, processor, embeds_output, device, 
    test_image_embeddings_pt, persam=None, test_image_embeddings_seg=None, batch_size=16
):
    """
    Run threshold or persam dense inference on a batch of images.

    Args:
        inference_mode (str): Inference mode ('threshold' or 'persam').
        test_imgs (list): List of test image paths.
        processor (EmbeddingProcessor): Processor for handling embeddings.
        embeds_output (EmbeddingOutput): Container with computed embeddings and masks.
        device (str): Device to run the model on.
        test_image_embeddings_pt (torch.Tensor): Test image embeddings.
        persam (PerSAM, optional): PerSAM model instance.
        test_image_embeddings_seg (torch.Tensor, optional): Test image embeddings for segmentation.
        batch_size (int, optional): Batch size for processing.

    Returns:
        list: List of tuples containing predicted mask, bounding box, and score.
    """
    results = []
    
    # Process target features once outside the loop
    target_feat = processor._process_target_feat(embeds_output.train_masked_embeds[0].to(device))
    
    # Pre-compute train bbox masked embeds for confidence scoring
    train_bbox_masked_embeds = torch.stack([x.mean(dim=0) for x in embeds_output.train_bbox_masked_embeds]).to(device)

    # Check dimensions of test_image_embeddings_pt and test_image_embeddings_seg
    if test_image_embeddings_pt.ndim == 3:
        test_image_embeddings_pt = test_image_embeddings_pt.unsqueeze(0)

    if test_image_embeddings_seg is not None and test_image_embeddings_seg.ndim == 3:
        test_image_embeddings_seg = test_image_embeddings_seg.unsqueeze(0)
    
    for i in range(0, len(test_imgs), batch_size):
        batch_imgs = test_imgs[i:i + batch_size]
        
        # Process batch of images
        batch_inputs = []
        batch_images = []
        for img_input in batch_imgs:
            img = processor.load_image(img_input)
            proc = processor.processor
            inputs = proc(images=img, return_tensors="pt")

            batch_images.append(img)
            batch_inputs.append(inputs)

        # Stack batch inputs
        stacked_inputs = {
            k: torch.cat([inp[k] for inp in batch_inputs], dim=0).to(device)
            for k in batch_inputs[0].keys()
        }
        
        # Process batch embeddings
        batch_embeddings = test_image_embeddings_pt[i:i + batch_size]
        B, C, h, w = batch_embeddings.shape
        
        # Normalize and reshape embeddings
        batch_test_feat = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
        batch_test_feat = batch_test_feat.reshape(B, C, h * w)
        
        # Compute similarities for whole batch
        batch_sim = torch.bmm(target_feat.unsqueeze(0).expand(B, -1, -1), batch_test_feat)
        batch_sim = batch_sim.reshape(B, 1, h, w)
        batch_sim = F.interpolate(batch_sim, scale_factor=4, mode="bilinear", align_corners=False)
        
        # Process each item in batch
        for j, sim in enumerate(batch_sim):
            sim = proc.post_process_masks(
                batch_sim[j:j+1].unsqueeze(1),
                original_sizes=[stacked_inputs["original_sizes"][j].tolist()],
                reshaped_input_sizes=[stacked_inputs["reshaped_input_sizes"][j].tolist()],
                binarize=False
            )[0]
            
            if inference_mode == "threshold":
                pred_mask, _ = get_thresholding_mask(sim.squeeze())
            elif inference_mode == "persam":
                pred_mask = persam.inference(
                    test_img=batch_images[j],
                    sim=sim.squeeze(),
                    image_embeddings=test_image_embeddings_seg[i + j] if test_image_embeddings_seg is not None else None,
                )
            else:
                print("Inference mode not supported")
                continue
                
            # Compute confidence score
            bbox = extract_bounding_box(pred_mask)
            if pred_mask is not None and bbox is not None:
                confidence_mask = create_bbox_mask(bbox, pred_mask)
                test_masked_embeds = batch_embeddings[j, :, confidence_mask[0][0] == 1].mean(dim=-1)
                score = F.cosine_similarity(
                    test_masked_embeds.unsqueeze(0),
                    train_bbox_masked_embeds,
                    dim=-1
                ).max()
                if inference_mode == "persam":
                    debug = False
                    if debug:
                        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                        axs[0].imshow(pred_mask, cmap='gray'); axs[0].set_title("Prediction Mask"); axs[0].axis('off')
                        axs[1].imshow(Image.open(batch_imgs[j]).convert("RGB")); axs[1].set_title("Test Image"); axs[1].axis('off')
                        fig.suptitle(f"Score: {score}")
                        plt.savefig(f"visualizations/{i}_{j}")
            else:
                pred_mask, bbox, score = None, None, None
            results.append((pred_mask, bbox, score))
    return results

def run_dense_tasks(args, backbone_model, backbone_name, preprocess, positive_cls, positive_cls_name, class_to_idx, device, persam=False):
    """
    Run dense tasks with memory-efficient batched processing.

    Args:
        args: Arguments containing configuration parameters.
        backbone_model: Backbone model instance.
        backbone_name (str): Name of the backbone model.
        preprocess: Preprocessing function.
        positive_cls (int): Positive class label.
        positive_cls_name (str): Name of the positive class.
        class_to_idx (dict): Mapping from class names to indices.
        device (str): Device to run the model on.
        persam (bool, optional): Whether to use PerSAM for inference.

    Returns:
        dict: Results for segmentation and detection.
    """
    # Get training image and mask paths
    train_imgs = get_paths(os.path.join(args.real_train_root, positive_cls_name))
    train_masks = [x.replace("/train/", "/train_masks/").replace(x.split('.')[-1], args.mask_ext) for x in train_imgs]
    
    # Initialize test dataset and dataloader
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
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize processors and compute embeddings
    emb_processor = EmbeddingProcessor(backbone_name=backbone_name, backbone_model=backbone_model, device=device)
    eval = EvalContainer()
    embedding_output = emb_processor.compute_embeddings(train_imgs, train_masks)
    
    if persam:
        # Initialize and train PerSAM once
        config = ModelConfig(device=device)
        persam = PerSAM(backbone_name=backbone_name, config=config)
        train_img = torch.Tensor(np.asarray(Image.open(train_imgs[0]).convert("RGB")))
        
        persam._train_mask_weights(
            embedding_output.train_embeds[0],
            emb_processor._process_target_feat(embedding_output.train_masked_embeds[0]),
            embedding_output.train_masks[0],
            train_img,
            config
        )
        
        # Initialize SAM model for PerSAM
        backbone_name_sam = "sam-vit-huge"
        backbone_model_sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        persam_processor = EmbeddingProcessor(backbone_name=backbone_name_sam, backbone_model=backbone_model_sam, device=device)
    
    print("Processing test data in batches")
    batch_idx = 0
    
    for i, (images, labels, paths) in tqdm(enumerate(test_loader)):
        # Process main embeddings for current batch
        pixel_values = torch.cat([emb_processor._process_image(p) for p in paths]).to(device)
        batch_embeddings = emb_processor._extract_embeddings(pixel_values)
        
        # Process SAM embeddings if using PerSAM
        if persam:
            pixel_values_sam = torch.cat([persam_processor._process_image(p) for p in paths]).to(device)
            batch_sam_embeddings = persam_processor._extract_embeddings(pixel_values_sam)
        
        # Process ground truth for batch
        for j, img_path in enumerate(paths):
            if args.dataset == "pods":
                mask_path = img_path.replace('/test_dense/', '/test_dense_masks/')
            else:
                mask_path = img_path.replace('/test/', '/test_masks/')
            gt_mask, height, width = load_ground_truth_mask(mask_path)
            eval.process_ground_truth(batch_idx + j, gt_mask, labels[j], positive_cls, height, width)
        
        # Threshold inference
        results = dense_inference_batch(
            "threshold", paths, emb_processor, embedding_output, 
            device, batch_embeddings
        )
        
        for j, (pred_mask, pred_bbox, score) in enumerate(results):
            eval.process_predictions(
                "threshold", batch_idx + j, paths[j], 
                pred_mask, pred_bbox, score, height, width
            )
        
        if persam:
            # PerSAM inference
            results = dense_inference_batch(
                "persam", paths, emb_processor, embedding_output,
                device, batch_embeddings, persam, batch_sam_embeddings
            )
            
            for j, (pred_mask, pred_bbox, score) in enumerate(results):
                eval.process_predictions(
                    "persam", batch_idx + j, paths[j],
                    pred_mask, pred_bbox, score, height, width
                )
        
        # Free up memory
        del batch_embeddings
        if persam:
            del batch_sam_embeddings
        torch.cuda.empty_cache()
        
        batch_idx += len(paths)
    
    results_dict = eval.finalize_results()
    return results_dict
