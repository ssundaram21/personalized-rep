import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeLoss(torch.nn.Module):
    def __init__(self, device: torch.device, margin: float, **kwargs):
        """
        Initializes the HingeLoss module.

        Args:
            device (torch.device): The device to run the loss computation on.
            margin (float): The margin value for the hinge loss.
        """
        super(HingeLoss, self).__init__()
        self.device = device
        self.margin = margin

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the hinge loss.

        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed hinge loss.
        """
        # Map [0, 1] -> {0, 1}
        y_rounded = torch.round(y) 
        # Map {0, 1} -> {-1, 1}
        y_transformed = -1 * (1 - 2 * y_rounded) 
        return torch.max(torch.zeros(x.shape).to(self.device), self.margin + (-1 * (x * y_transformed))).sum()

class MultiPositiveInfoNCELoss(nn.Module):
    def __init__(self, device: torch.device, temperature: float = 0.07, **kwargs):
        """
        Initializes the MultiPositiveInfoNCELoss module.

        Args:
            device (torch.device): The device to run the loss computation on.
            temperature (float): The temperature scaling factor.
        """
        super(MultiPositiveInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.device = device
        
    def forward(self, query: torch.Tensor, positive_keys: torch.Tensor, negative_keys: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-positive InfoNCE loss.

        Args:
            query (torch.Tensor): The query tensor.
            positive_keys (torch.Tensor): The positive keys tensor.
            negative_keys (torch.Tensor): The negative keys tensor.

        Returns:
            torch.Tensor: The computed multi-positive InfoNCE loss.
        """
        # Normalize 
        query = F.normalize(query, dim=1).to(self.device)
        positive_keys = F.normalize(positive_keys, dim=1).to(self.device)
        negative_keys = F.normalize(negative_keys, dim=1).to(self.device)

        # Compute similarities
        positive_logits = torch.matmul(query, positive_keys.transpose(0, 1)) / self.temperature
        negative_logits = torch.matmul(query, negative_keys.transpose(0, 1)) / self.temperature

        # Compute log-sum-exp for positives
        pos_log_sum_exp = torch.logsumexp(positive_logits, dim=1, keepdim=True)

        # Compute log-sum-exp for all (positives and negatives)
        all_logits = torch.cat([positive_logits, negative_logits], dim=1)
        all_log_sum_exp = torch.logsumexp(all_logits, dim=1, keepdim=True)

        # Compute loss
        loss = -pos_log_sum_exp + all_log_sum_exp

        return loss.mean()

class InfoNCELoss(nn.Module):
    def __init__(self, device: torch.device, temperature: float = 0.07, **kwargs):
        """
        Initializes the InfoNCELoss module.

        Args:
            device (torch.device): The device to run the loss computation on.
            temperature (float): The temperature scaling factor.
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, query: torch.Tensor, positive_keys: torch.Tensor, negative_keys: torch.Tensor) -> torch.Tensor:
        """
        Computes the InfoNCE loss.

        Args:
            query (torch.Tensor): The query tensor.
            positive_keys (torch.Tensor): The positive keys tensor.
            negative_keys (torch.Tensor): The negative keys tensor.

        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Normalize 
        query = F.normalize(query, dim=1).to(self.device)
        positive_keys = F.normalize(positive_keys, dim=1).to(self.device)
        negative_keys = F.normalize(negative_keys, dim=1).to(self.device)

        # Compute similarities for negative keys (this is common for all positive keys)
        negative_logits = torch.matmul(query, negative_keys.transpose(0, 1))

        total_loss = 0
        n_positives = positive_keys.shape[0]

        for i in range(n_positives):
            # Compute similarity for the current positive key
            positive_logit = torch.sum(query * positive_keys[i], dim=1, keepdim=True)

            # Concatenate positive and negative logits
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            logits /= self.temperature

            # Create labels (0 is the index of the positive key)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

            # Compute cross entropy loss for this positive key
            loss = F.cross_entropy(logits, labels)
            total_loss += loss

        # Average the loss over all positive keys
        average_loss = total_loss / n_positives

        return average_loss
        
def get_loss_fn(loss: str, **kwargs) -> nn.Module:
    """
    Returns the loss function based on the provided loss name.

    Args:
        loss (str): The name of the loss function to retrieve.
        **kwargs: Additional keyword arguments for the loss function.

    Returns:
        torch.nn.Module: The loss function corresponding to the provided name.

    Raises:
        ValueError: If the provided loss name is not valid.
    """
    losses = {
        "hinge": HingeLoss(**kwargs),
        "info_nce_fixed": InfoNCELoss(**kwargs),
        "info_nce_multi_pos": MultiPositiveInfoNCELoss(**kwargs)
    }
    if loss not in losses:
        raise ValueError(f"Invalid loss name '{loss}'. Available options are: {list(losses.keys())}")
    return losses[loss]
