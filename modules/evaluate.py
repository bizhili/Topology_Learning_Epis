import torch

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




