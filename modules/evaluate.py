import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def square_error(Z, preZ):
    return torch.sqrt((Z-preZ)**2).sum()

def cosine_similarity(Z, preZ):
    numerator= torch.sum(Z*preZ)
    denominator= torch.sqrt(torch.sum(Z**2))*torch.sqrt(torch.sum(preZ**2))
    return numerator/denominator

def spectral_analysis(Z, preZ):
    try:
        preEig, _ = torch.linalg.eig(preZ)
        Eig, _ = torch.linalg.eig(Z)
        return cosine_similarity(Eig.real, preEig.real)
    except:
        return 0

def recall(Z, preZ):
    numerator= torch.sum(Z*preZ)
    denominator= torch.sum(torch.abs(Z.float()))
    return numerator/denominator

def precision(Z, preZ):
    numerator= torch.sum(Z*preZ)
    denominator= torch.sum(torch.abs(preZ.float()))
    return numerator/denominator
    
def jaccard_index(Z, preZ):
    numerator= torch.sum(Z*preZ)
    denominator= torch.sum((Z+preZ)-Z*preZ)
    return numerator/denominator

def normalized_hamming_distance(Z, preZ):
    """Calculates the Normalized Hamming Distance between two matrices."""

    assert Z.shape == preZ.shape, "Matrices must have the same shape."
    num_nodes = Z.shape[0]
    max_distance = num_nodes * (num_nodes - 1)   # Maximum possible distance for a full matrix

    hamming_distance = torch.sum(torch.logical_xor(Z, preZ)) #01, 10, (11, 00)
    normalized_distance = torch.sqrt(hamming_distance /max_distance)

    return normalized_distance

def draw_auc_roc(A, preA):
    # Flatten matrices into 1D arrays
    y_true = A.numpy().flatten()
    y_pred = preA.numpy().flatten()

    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    print(fpr)

    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
