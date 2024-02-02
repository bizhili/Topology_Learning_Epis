import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


def continious_to_sparcity(my_array, top= 400):
    # Flatten the array to a 1D array
    flat_array = my_array.flatten()

    # Sort the flattened array in descending order
    sorted_indices = np.argsort(flat_array)[::-1]

    # Set the top 400 elements to 1 and the rest to 0
    flat_array[sorted_indices[:top]] = 1
    flat_array[sorted_indices[top:]] = 0

    # Reshape the modified 1D array back to the original shape
    result_array = flat_array.reshape(my_array.shape)
    return result_array

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

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

    numerator= torch.sum(Z*preZ)
    denominator= torch.sum(torch.abs(Z.float()))
    return numerator/denominator

def precision(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

    numerator= torch.sum(Z*preZ)
    denominator= torch.sum(torch.abs(preZ.float()))
    return numerator/denominator

def selectivity(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

    numerator= torch.sum((1-Z)*(1-preZ))
    denominator= torch.sum(torch.abs((1-Z).float()))
    return numerator/denominator

def negative_predictive(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix


    numerator= torch.sum((1-Z)*(1-preZ))
    denominator= torch.sum(torch.abs((1-preZ).float()))
    return numerator/denominator

def accuracy(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

    numerator= torch.sum(1-torch.logical_xor(Z, preZ).float())
    denominator= Z.shape[0]*Z.shape[0]
    return numerator/denominator

def jaccard_index(Z, preZ):

    links= int(torch.sum(Z>1e-6))
    IMatrix= torch.eye(Z.shape[0], device= "cpu")
    Z= torch.tensor(continious_to_sparcity(Z.numpy(), links))+IMatrix
    preZ= torch.tensor(continious_to_sparcity(preZ.numpy(), links))+IMatrix

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

def draw_auc_roc(As, preAs, legends= []):
    plt.figure()
    for i, A in enumerate(As):
        # Flatten matrices into 1D arrays
        y_true = A.numpy().flatten()
        y_pred = preAs[i].numpy().flatten()

        # Calculate false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{legends[i]}(area=%0.3f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label= "Radom pred")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def draw_prc(As, preAs, legends= []):
    plt.figure()
    for i, A in enumerate(As):
        # Flatten matrices into 1D arrays
        y_true = A.numpy().flatten()
        y_pred = preAs[i].numpy().flatten()

        # Calculate false positive rate, true positive rate, and thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        # Calculate AUC
        #roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.plot(recall, precision, label=f'{legends[i]}')

    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")
    plt.show()