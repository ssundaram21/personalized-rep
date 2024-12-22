from transformers import AutoProcessor, SamModel

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from typing import Optional
from dataclasses import dataclass
from PIL import Image
from scipy.spatial.distance import cdist

from skimage import measure

class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

@dataclass
class ModelConfig:
    """Configuration for model parameters and training."""
    num_epochs: int = 2000
    log_epoch: int = 200
    learning_rate: float = 1e-3
    optimizer_eps: float = 1e-4
    device: str = "cuda"
    skip_training: bool = False

class PerSAM:
    def __init__(
        self,
        backbone_name: str,
        config: Optional[ModelConfig] = None,
        preload_models: bool = True
    ):
        self.config = config or ModelConfig()
        self.backbone_name = backbone_name
        
        if preload_models:
            self.processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.config.device)
        else:
            self.processor = None
            self.sam_model = None
        self.weights = None
        self.weights_np = None

    def _get_image_size(self) -> Tuple[int, int]:
        """Get required image size based on backbone.

        Returns:
            Tuple[int, int]: Image size (height, width).
        """
        if "vitb14" in self.backbone_name:
            return (896, 896)
        elif "vitb16" in self.backbone_name:
            return (1024, 1024)
        return (1024, 1024)  # default size

    def _train_mask_weights(
        self,
        ref_feat: torch.Tensor,
        target_feat: torch.Tensor,
        gt_mask: torch.Tensor,
        original_image: torch.Tensor,
        config: ModelConfig,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Train mask weights for personalized segment anything model using reference features and mask.

        Args:
            ref_feat (torch.Tensor): Reference features.
            target_feat (torch.Tensor): Target features.
            gt_mask (torch.Tensor): Ground truth mask.
            original_image (torch.Tensor): Original image tensor.
            config (ModelConfig): Configuration for model parameters and training.

        Returns:
            Tuple[torch.Tensor, np.ndarray]: Trained mask weights and numpy array of weights.
        """
        # Step 1: Calculate target features and similarity
        ref_feat = ref_feat.permute(1, 2, 0)
        h, w, C = ref_feat.shape

        # Extract target features from reference
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
        sim = target_feat @ ref_feat
        
        # Process similarity map
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        
        # Get inputs for SAM
        inputs = self.processor(
            original_image,
            return_tensors="pt"
        ).to(config.device)
        
        sim = self.processor.post_process_masks(
            sim.unsqueeze(1), 
            original_sizes=inputs["original_sizes"].tolist(),
            reshaped_input_sizes=inputs["reshaped_input_sizes"].tolist(),
            binarize=False
        )
        sim = sim[0].squeeze()
        
        # Get point selection
        topk_xy, topk_label = self.point_selection(sim, topk=1)
        
        # Prepare SAM inputs with selected points
        inputs = self.processor(
            original_image,
            input_points=[topk_xy.tolist()],
            input_labels=[topk_label.tolist()],
            return_tensors="pt"
        ).to(config.device)
        
        # Get image embeddings
        image_embeddings = self.sam_model.get_image_embeddings(inputs.pixel_values)
        
        # Initialize mask weights
        mask_weights = Mask_Weights().to(config.device)
        mask_weights.train()
        
        # Setup optimizer and scheduler
        optimizer = AdamW(mask_weights.parameters(), lr=1e-3, eps=1e-4)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=config.num_epochs)
        
        # Prepare ground truth mask
        gt_mask = gt_mask.flatten().unsqueeze(0).to(config.device)
        
        # Training loop
        for epoch in range(config.num_epochs):
            # Get model predictions
            with torch.no_grad():
                outputs = self.sam_model(
                    input_points=inputs.input_points,
                    input_labels=inputs.input_labels,
                    image_embeddings=image_embeddings,
                    multimask_output=True,
                )
            
            # Process predictions
            logits_high = self.postprocess_masks(
                masks=outputs.pred_masks.squeeze(1),
                input_size=inputs.reshaped_input_sizes[0].tolist(),
                original_size=inputs.original_sizes[0].tolist()
            )
            logits_high = logits_high[0].flatten(1)
            
            # Calculate weighted sum of masks
            weights = torch.cat(
                (1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights),
                dim=0
            )
            logits_high = (logits_high * weights).sum(0).unsqueeze(0)
            
            # Calculate losses
            dice_loss = self.calculate_dice_loss(logits_high, gt_mask)
            focal_loss = self.calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if hasattr(config, 'log_interval') and epoch % config.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch: {epoch}/{config.num_epochs}, LR: {current_lr:.6f}, '
                      f'Dice Loss: {dice_loss.item():.4f}, '
                      f'Focal Loss: {focal_loss.item():.4f}')
        
        # Prepare final weights
        mask_weights.eval()
        self.weights = torch.cat(
            (1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights),
            dim=0
        )
        self.weights_np = weights.detach().cpu().numpy()
        print("Finished PerSAM Training")

    @torch.no_grad
    def inference(
        self,
        test_img: torch.Tensor,
        sim,
        image_embeddings: torch.Tensor,
    ) -> Image:
        """
        Perform inference using the trained perSAM model and mask weights.

        Args:
            test_img (torch.Tensor): Test image tensor.
            sim: Similarity map.
            image_embeddings (torch.Tensor): Image embeddings.

        Returns:
            Image: Final mask as an image.
        """
        if "dinov2" in self.backbone_name:
            topk_xy, topk_label = self.top_k_with_distance(sim)
        else:
            topk_xy, topk_label = self.point_selection(sim, topk=2)
            
        inputs = self.processor(
            test_img, input_points=[topk_xy.tolist()],
            input_labels=[topk_label.tolist()], return_tensors="pt"
        ).to(self.config.device)

        # First-step prediction
        outputs = self.sam_model(
            input_points=inputs.input_points,
            input_labels=inputs.input_labels,
            image_embeddings=image_embeddings.unsqueeze(0),
            multimask_output=True
        )

        logits = (outputs.pred_masks[0].squeeze(0).cpu().numpy() * self.weights_np[..., None])
        logit = logits.sum(0)
            
        # Weighted sum three-scale masks
        logits_high = self.postprocess_masks(masks=outputs.pred_masks.squeeze(1),
                                            input_size=inputs.reshaped_input_sizes[0].tolist(),
                                            original_size=inputs.original_sizes[0].tolist())
        
        logits_high = logits_high[0] * self.weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)

        # Cascaded Post-Refinement 1
        mask = (logit_high > 0).detach().cpu().numpy()
        
        y, x = np.nonzero(mask)
        if len(x) == 0 or len(y) == 0:
            return None
        input_box = [[x.min(), y.min(), x.max(), y.max()]]
        input_boxes = self.processor(
            test_img, input_boxes=[input_box], return_tensors="pt"
        ).input_boxes.to(self.config.device)
        outputs_1 = self.sam_model(
            input_points=inputs.input_points,
            input_labels=inputs.input_labels,
            input_boxes=input_boxes,
            input_masks=torch.tensor(logit[None, None, :, :], device=self.config.device), image_embeddings=image_embeddings.unsqueeze(0),
            multimask_output=True
        )

        # Cascaded Post-Refinement 2
        masks = self.processor.image_processor.post_process_masks(
            outputs_1.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0].squeeze().numpy()
        best_idx = torch.argmax(outputs_1.iou_scores).item()
        y, x = np.nonzero(masks[best_idx])
        if len(x) == 0 or len(y) == 0:
            return None

        input_box = [[x.min(), y.min(), x.max(), y.max()]]
        input_boxes = self.processor(
            test_img, input_boxes=[input_box], return_tensors="pt"
        ).input_boxes.to(self.config.device)

        final_outputs = self.sam_model(
            input_points=inputs.input_points,
            input_labels=inputs.input_labels,
            input_boxes=input_boxes,
            input_masks=outputs_1.pred_masks.squeeze(1)[:,best_idx: best_idx + 1, :, :],
            image_embeddings=image_embeddings.unsqueeze(0),
            multimask_output=True
        )

        masks = self.processor.image_processor.post_process_masks(
            final_outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0].squeeze().numpy()
        best_idx = torch.argmax(final_outputs.iou_scores).item()
        final_mask = masks[best_idx]
        return final_mask

    def point_selection(self, mask_sim, topk=1):
        """
        Select top-k points based on mask similarity.

        Args:
            mask_sim: Mask similarity map.
            topk (int): Number of top points to select.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected points and labels.
        """
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()
        
        return topk_xy, topk_label

    def top_k_with_distance(self, mask_sim, num_points=5, min_distance=10):
        """
        Select top-k points based on mask similarity and distance.

        Args:
            mask_sim: Mask similarity map.
            num_points (int): Number of points to select.
            min_distance (int): Minimum distance between points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected points and labels.
        """
        w, h = mask_sim.shape
        mask_sim_np = mask_sim.cpu().numpy()
        
        # Threshold the mask to get high activation areas
        threshold = mask_sim_np.max() - (mask_sim_np.max() - mask_sim_np.min()) * 0.1 
        binary_mask = mask_sim_np > threshold
        
        # Label connected components
        labeled_mask, num_features = measure.label(binary_mask, return_num=True)
        # Calculate areas of connected components
        areas = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
        
        if areas:
            # Calculate average area of high activation regions
            avg_area = np.mean(areas)
            # Set min_distance as the square root of the average area
            min_distance = max(int(np.sqrt(avg_area)), 1)
        else:
            # Fallback if no high activation areas are found
            min_distance = 50
        
        # Get top k*2 points (we'll filter some out based on distance)
        top_k = np.sum(binary_mask)  # 5000 #num_points * 2
        flat_indices = np.argpartition(mask_sim_np.ravel(), -top_k)[-top_k:]
        top_points = np.column_stack(np.unravel_index(flat_indices, mask_sim_np.shape))
        
        selected_points = [top_points[0]]  # Start with the highest confidence point
        
        for point in top_points[1:]:
            if len(selected_points) == num_points or mask_sim_np[point[0], point[1]] < threshold:
                break
            distances = cdist([point], selected_points)[0]
            if np.all(distances >= min_distance):
                selected_points.append(point)
        return np.array(selected_points)[:, ::-1], np.ones(len(selected_points))
        
    def calculate_dice_loss(self, inputs, targets, num_masks=1):
        """
        Compute the DICE loss, similar to generalized IOU for masks.

        Args:
            inputs (torch.Tensor): Predictions for each example.
            targets (torch.Tensor): Binary classification label for each element in inputs.
            num_masks (int): Number of masks.

        Returns:
            torch.Tensor: DICE loss.
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def calculate_sigmoid_focal_loss(self, inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection.

        Args:
            inputs (torch.Tensor): Predictions for each example.
            targets (torch.Tensor): Binary classification label for each element in inputs.
            num_masks (int): Number of masks.
            alpha (float): Weighting factor to balance positive vs negative examples.
            gamma (float): Exponent of the modulating factor to balance easy vs hard examples.

        Returns:
            torch.Tensor: Focal loss.
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
    
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
    
        return loss.mean(1).sum() / num_masks

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...], img_size=1024) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (torch.Tensor): Batched masks from the mask_decoder, in BxCxHxW format.
            input_size (Tuple[int, ...]): The size of the image input to the model, in (H, W) format.
            original_size (Tuple[int, ...]): The original size of the image before resizing for input to the model, in (H, W) format.
            img_size (int): Image size.

        Returns:
            torch.Tensor: Batched masks in BxCxHxW format, where (H, W) is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks