import numpy as np
import random
import torch
def create_transfer_net(Aw, G, paras, mf= 10, device= "cpu"):
    edgeList= list(G.edges())
    AwSub= Aw.cpu().numpy().copy()
    Awf= torch.tensor(AwSub, device= device)
    AwSubM= int((AwSub>0).sum()/2)
    seleLinks = random.sample(range(AwSubM), mf)
    seleLinksT= []
    routingNodes= []
    print([edgeList[i] for i in seleLinks])

    for linkID in seleLinks:
        edge= edgeList[linkID]
        seleLinksT.append(edge)
        i= edge[0]
        j= edge[1]
        neighborI= (Aw[i, :]>1e-5)*1
        neighborJ= (Aw[j, :]>1e-5)*1
        sharedNodeID= torch.nonzero(neighborI*neighborJ)[:, 0]
        sharedNodeID= list(set(sharedNodeID.tolist())-set(seleLinks))
        if len(sharedNodeID):
            weightOfChoice= Aw[torch.tensor(sharedNodeID, dtype= torch.int32), :].sum(dim= 1)
            probability= weightOfChoice/weightOfChoice.sum()
            rN= np.random.choice(sharedNodeID, 1, p= probability.cpu().detach().numpy()) # select node
            routingNodes.append(rN[0])
            Awf[i, j]= 0
            Awf[j, i]= 0
            Awf[j, rN]+= paras.identicalf
            Awf[rN, j]+= paras.identicalf
            Awf[i, rN]+= paras.identicalf
            Awf[rN, i]+= paras.identicalf

    return Awf, [edgeList[i][0] for i in seleLinks]+[edgeList[i][1] for i in seleLinks], seleLinksT, routingNodes


def row_normalize(mat, eps=1e-12):
    """Row-normalize a matrix to make each row a probability distribution."""
    row_sums = mat.sum(dim=1, keepdim=True).clamp_min(eps)
    return mat / row_sums

def relative_entropy(A, Af, eps=1e-12):
    """
    Compute row-wise KL divergence KL(A || Af).
    A, Af: [n, n] adjacency matrices (non-negative).
    Returns scalar loss.
    """
    # Row-normalize both to probabilities
    P = row_normalize(A, eps)
    Q = row_normalize(Af, eps)

    # Add epsilon to avoid log(0)
    P_safe = P.clamp_min(eps)
    Q_safe = Q.clamp_min(eps)

    # KL divergence: sum_i sum_j P_ij (log P_ij - log Q_ij)
    kl = (P_safe * (P_safe.log() - Q_safe.log())).sum()
    return kl