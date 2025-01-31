
# Import libraries
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Define ROC calculation function
def calculate_roc(predictions, ground_truth, num_thresholds=1000):
    """
    Manually calculate the ROC curve for large-scale 3D segmentation data.

    Args:
        predictions (torch.Tensor): Predicted probabilities, shape (N, X, Y, Z).
        ground_truth (torch.Tensor): Ground truth binary labels, shape (N, X, Y, Z).
        num_thresholds (int): Number of thresholds to compute the ROC curve.

    Returns:
        fpr_list (list): False Positive Rates for each threshold.
        tpr_list (list): True Positive Rates for each threshold.
        auc_score (float): Approximation of the AUC.
    """
    # Flatten the tensors
    preds_flat = predictions.flatten().cpu().numpy()
    truth_flat = ground_truth.flatten().cpu().numpy()

    # Define thresholds
    thresholds = np.linspace(0, 1, num_thresholds)

    # Initialize lists for TPR and FPR
    tpr_list = []
    fpr_list = []

    # Iterate over thresholds
    for threshold in thresholds:

        # Binarize predictions
        preds_binary = preds_flat > threshold

        # Calculate confusion matrix components
        tp = np.sum((preds_binary == 1) & (truth_flat == 1))  # True Positives
        fp = np.sum((preds_binary == 1) & (truth_flat == 0))  # False Positives
        tn = np.sum((preds_binary == 0) & (truth_flat == 0))  # True Negatives
        fn = np.sum((preds_binary == 0) & (truth_flat == 1))  # False Negatives

        # Compute TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Fall-out

        # Store results
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Approximate AUC using the trapezoidal rule
    auc_score = np.trapz(tpr_list, fpr_list)

    # Return results
    return fpr_list, tpr_list, auc_score

# Define function to get test metrics
def get_test_metrics(logits, ground_truth):

    # Update confusion matrix
    pred = logits.argmax(dim=1)
    TP = ((pred == 1) & (ground_truth == 1)).sum().item()
    FP = ((pred == 1) & (ground_truth == 0)).sum().item()
    TN = ((pred == 0) & (ground_truth == 0)).sum().item()
    FN = ((pred == 0) & (ground_truth == 1)).sum().item()

    # Return results
    return TP, FP, TN, FN

# Define function to test model
def test_model(model, dataset_test):

    # Get constants
    device = next(model.parameters()).device

    # Set model to evaluation mode
    model.eval()

    # Set up data loader for test set
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Initialize test metrics
    TP, FP, TN, FN = 0, 0, 0, 0
    fpr, tpr, auc = 0, 0, 0

    # Calculate metrics
    print('Calculating metrics')
    for batch_idx, (scan, mask) in enumerate(loader_test):

        # Status update
        if batch_idx % 10 == 0:
            print(f'-- Batch {batch_idx}/{len(loader_test)}')

        # Send to device
        scan = scan.to(device)
        mask = mask.to(device)

        # Forward pass
        with torch.no_grad():
            logits = model(scan)
            pred = logits.softmax(dim=1)[:, 1]

        # Update confusion matrix
        TP_, FP_, TN_, FN_ = get_test_metrics(logits, mask)
        TP += TP_
        FP += FP_
        TN += TN_
        FN += FN_

        # Calculate ROC curve
        fpr_, tpr_, auc_ = calculate_roc(pred, mask)
        fpr += np.array(fpr_) / len(loader_test)
        tpr += np.array(tpr_) / len(loader_test)
        auc += auc_ / len(loader_test)

    # Calculate metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Package results
    test_metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc,
    }

    # Return
    return test_metrics

    

