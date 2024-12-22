from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
from statistics import mean

def compute_auc(gts, preds):
    """
    Compute the precision-recall area under the curve (PR-AUC) for the given ground truths and predictions.

    Args:
        gts (torch.Tensor): Ground truth labels.
        preds (list): Predicted scores.

    Returns:
        float: Average precision score.
    """
    return average_precision_score(gts.cpu().numpy(), preds)

def classification(
    train_embeds, 
    train_labels, 
    test_embeds, 
    test_labels, 
    test_paths, 
    positive_cls, 
    save_extra=True
):
    """
    Perform classification and compute AUC for each class.

    Args:
        train_embeds (numpy.ndarray): Training embeddings.
        train_labels (numpy.ndarray): Training labels.
        test_embeds (numpy.ndarray): Test embeddings.
        test_labels (numpy.ndarray): Test labels.
        test_paths (list): Paths to test samples.
        positive_cls (int): Positive class label.
        save_extra (bool): Flag to save extra information.
        
    Returns:
        dict: Dictionary containing average and per-class AUC, predictions, and paths.
    """
    # Convert numpy arrays to torch tensors
    train_embeds = torch.from_numpy(train_embeds)
    train_labels = torch.from_numpy(train_labels)
    
    # Filter training data for the positive class
    train_mask = torch.isin(train_labels, torch.Tensor([positive_cls]))
    train_embeds = train_embeds[train_mask]
    train_labels = train_labels[train_mask]
    
    test_embeds = torch.from_numpy(test_embeds)
    test_labels = torch.from_numpy(test_labels)
    
    # Initialize dictionaries to store per-class AUC and predictions
    per_class_auc = {cls.item(): 0 for cls in train_labels.unique()}
    per_class_preds = {cls.item(): None for cls in train_labels.unique()}
    
    # Iterate over each unique class in training labels
    for cls in tqdm(torch.unique(train_labels)):
        train_embed_cls = train_embeds[train_labels == cls]
        train_label_cls = train_labels[train_labels == cls]
        test_label_cls = (test_labels == cls).type(torch.int)
        
        preds = []
        for i in range(test_labels.shape[0]):
            # Compute cosine similarity between test and train embeddings
            sim = F.cosine_similarity(test_embeds[i, :], train_embed_cls, dim=-1)
            sim_score, sim_ind = torch.max(sim).cpu().item(), torch.argmax(sim)
            preds.append(sim_score)
        
        # Compute AUC for the current class
        per_class_auc[cls.cpu().item()] = compute_auc(test_label_cls, preds)
        per_class_preds[cls.cpu().item()] = preds
    
    # Compute average AUC across all classes
    avg_results = mean(list(per_class_auc.values()))
    per_class_paths = list(test_paths)
    print(avg_results)
    
    # Prepare results dictionary
    if not save_extra:
        all_results = {"avg": avg_results, "per_class": per_class_auc}
    else:
        all_results = {
            "avg_classification": avg_results,
            "per_class_classification": per_class_auc,
            "preds_classification": per_class_preds,
            "paths_classification": per_class_paths
        }
    
    return all_results