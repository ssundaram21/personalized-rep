import torch
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def ndcg_score(ranked_labels, retrieval_cls):
    """
    Compute the NDCG score over all retrieved images for a single query.
    
    Args:
    - ranked_labels (torch.Tensor): The labels ranked by similarity for this query.
    - retrieval_cls (int): The desired retrieval class.
    
    Returns:
    - float: The NDCG score.
    """
    relevance_scores = (ranked_labels == retrieval_cls).float()
    
    # DCG calculation
    dcg = torch.sum(relevance_scores / torch.log2(torch.arange(2, relevance_scores.size(0) + 2, dtype=torch.float32)))
    
    # IDCG calculation
    ideal_relevance_scores = torch.sort(relevance_scores, descending=True).values
    idcg = torch.sum(ideal_relevance_scores / torch.log2(torch.arange(2, ideal_relevance_scores.size(0) + 2, dtype=torch.float32)))
    
    # NDCG calculation
    ndcg = (dcg / idcg) if idcg > 0 else torch.tensor(0)
    return ndcg.item() 

def top_k_accuracy(ranked_labels, retrieval_cls, k):
    """
    Compute the top-k accuracy for a single query.
    
    Args:
    - ranked_labels (torch.Tensor): The labels ranked by similarity for this query.
    - retrieval_cls (int): The desired retrieval class.
    - k (int): The cutoff for top-k accuracy.
    
    Returns:
    - float: 1.0 if retrieval_cls is in the top-k, otherwise 0.0.
    """
    top_k_labels = ranked_labels[:k]
    return float(retrieval_cls in top_k_labels)

def mean_reciprocal_rank(ranked_labels, retrieval_cls):
    """
    Compute the MRR score for a single query.
    
    Args:
    - ranked_labels (torch.Tensor): The labels ranked by similarity for this query.
    - retrieval_cls (int): The desired retrieval class.
    
    Returns:
    - float: The MRR score.
    """
    rank_position = (ranked_labels == retrieval_cls).nonzero(as_tuple=True)[0][0].item() + 1
    return 1.0 / rank_position


def retrieval(
    train_embeds, 
    train_labels, 
    test_embeds, 
    test_labels, 
    test_paths, 
    retrieval_cls
):
    """
    Perform retrieval and compute evaluation metrics.

    Args:
    - train_embeds (numpy.ndarray): Embeddings of the training set.
    - train_labels (numpy.ndarray): Labels of the training set.
    - test_embeds (numpy.ndarray): Embeddings of the test set.
    - test_labels (numpy.ndarray): Labels of the test set.
    - test_paths (list): Paths of the test set images.
    - retrieval_cls (int): The desired retrieval class.

    Returns:
    - dict: A dictionary containing average and individual retrieval metrics.
    """
    retrieval_embeds = torch.from_numpy(train_embeds)
    retrieval_labels = torch.from_numpy(train_labels)

    query_embeds = torch.from_numpy(test_embeds)
    query_labels = torch.from_numpy(test_labels)
    query_mask = torch.isin(query_labels, torch.Tensor([retrieval_cls]))
    query_embeds = query_embeds[query_mask]
    query_labels = query_labels[query_mask]
    query_paths = test_paths[query_mask]
            
    ndcg_scores = []
    top_k_accuracies = []
    mrr_scores = []
    for i in range(query_labels.shape[0]):
        # Get a similarity matrix with size (test_embed_stack.shape[0], train_embed_stack.shape[0])
        sim = F.cosine_similarity(query_embeds[i, :], retrieval_embeds, dim=-1)
        ranked_indices = torch.argsort(sim, descending=True)
        ranked_retrieval_labels = retrieval_labels[ranked_indices]
        
        # Calculate NDCG, top-k accuracy, and MRR for this query
        ndcg_scores.append(ndcg_score(ranked_retrieval_labels, retrieval_cls))
        top_k_accuracies.append(top_k_accuracy(ranked_retrieval_labels, retrieval_cls, 10))
        mrr_scores.append(mean_reciprocal_rank(ranked_retrieval_labels, retrieval_cls))

    average_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    average_top_k_accuracy = sum(top_k_accuracies) / len(top_k_accuracies)
    average_mrr = sum(mrr_scores) / len(mrr_scores)

    avg_results = {'NDCG': average_ndcg, 'MRR': average_mrr, 'top-10 acc': average_top_k_accuracy}
    all_results = {"avg_retrieval": avg_results, "mrrs_retrieval": mrr_scores, "ndcgs_retrieval": ndcg_scores, "topks_retrieval": top_k_accuracies}
    print(avg_results)
    return all_results