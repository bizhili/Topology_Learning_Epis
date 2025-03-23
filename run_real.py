# %load_ext autoreload
# %autoreload 2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import argparse
import modules.A_mat as A_mat
import modules.pramameters as pramameters
import modules.plotGraph as plotGraph
import modules.utils as utils
import modules.nn as mynn
import modules.evaluate as evaluate
import pandas as pd


printFlag= 1
plotFlag= 1

#Changeable parameters 
parser = argparse.ArgumentParser(description='Topology fitting parameters')
pramameters.add_arguments(parser)
paras= pramameters.read_arguments(parser)
device= paras.device if torch.cuda.is_available() else "cpu"
#preset
paras.modelLoad= "AA"
# paras.epoches= 20000
# paras.epi= "ALL_"# Covid, H1N1, Sars, ALL


paras.strains= 1
if paras.epi== "ALL_":
    paras.strains = 3

# Read the CSV file into a NumPy array
epiOData = np.genfromtxt(f'testData/test_data.csv', delimiter=',', skip_header=0)
popData = np.genfromtxt(f'testData/test_pop.csv', delimiter=',', skip_header=0)
popIdxData = np.genfromtxt(f'testData/test_pop_inx.csv', delimiter=',', skip_header=0)
popIdxData= [int(i) for i in popIdxData]
pop_file_path = 'testData/Country_Population_final.csv'
data = pd.read_csv(pop_file_path)
popName= data["Country"].tolist()
popIdxAll= data["idx"]
popCropArea= data["CropArea"].to_numpy()
popCity= data["CityPop"].to_numpy()
popAll= data["CityPop"].to_numpy()
curPopName= [popName[int(i)] for i in popIdxData]
epiOData= np.array_split(epiOData, 3, axis=0)
epiOData= np.stack(epiOData, axis=1)
if paras.epi== "Sars_":
    epiOData= epiOData[:, 0:1, :]
elif paras.epi== "H1N1_":
    epiOData= epiOData[:, 1:2, :]
elif paras.epi== "Covid_":
    epiOData= epiOData[:, 2:3, :]
popData= popAll[popIdxData]#/(popCropArea[popIdxData]/1e4+1e-9)
epiData= epiOData.copy()/popData[None, :]#[np.argmax(popData)]
epiData[1:, ...]= epiData[1:, ...]- epiData[:-1, ...]
epiData[epiData<0]= 0
epiData= epiData.T

mask= np.ones(epiData.shape)
mask= mask*popData[:, None, None]
divide= torch.Tensor(epiData).to(device)
maskTensor= torch.Tensor(mask).to(device= device)
paras.n= divide.shape[0]
P= torch.Tensor(popData).to(device)


# paras.seed= 1018#1, 10, 20, 30, 40, 50

torch.manual_seed(paras.seed)
timeHorizon= divide.shape[2]-1
myMatch= mynn.matchingA(timeHorizon+1, paras.strains, paras.n, channel= 5,  device= device)
myEpi= mynn.EpisA(input_dim= timeHorizon+1, num_heads= paras.strains, n= paras.n, device= device)


optimizer1 = torch.optim.Adam(myMatch.parameters(),lr= 3e-4)#myMatch.parameters()
optimizer2 = torch.optim.Adam({myEpi.taus},lr= 3e-4)
optimizer3 = torch.optim.Adam({myEpi.R0dTaus},lr= 3e-4)
myloss= torch.nn.MSELoss(reduction='sum')
losses= []
predError= []
maskDivide= maskTensor*divide
ratio= 1/(torch.norm(maskDivide.norm(dim= 2, keepdim= True), dim= 0, keepdim = True))


Adj = np.genfromtxt('empirical_data/Flights_adj.csv', delimiter=',', skip_header= 0)
np.fill_diagonal(Adj, 0)
popIdxDataNp= np.array(popIdxData, dtype= "int")
AdjCur= Adj[ popIdxDataNp,:]
AdjCur= AdjCur[ :,popIdxDataNp]
AdjCurTen= torch.tensor(AdjCur, dtype= torch.float32).to(device)



evaluateMeth= [evaluate.spectral_similarity, evaluate.pearson_correlation, evaluate.jaccard_similarity,evaluate.ROC_AUC, evaluate.PR_AUC]
for j in range(paras.epoches):
    optimizer1.zero_grad()  
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    inferZmat= myMatch(divide, paras.modelLoad)
    predSignal, signal, PreZ= myEpi(divide, inferZmat)
    loss= myloss((predSignal*maskTensor*ratio)[:, :, 0:-1],(signal*maskTensor*ratio)[:, :, 1:])/100 + torch.var(myEpi.taus, dim= 0).sum()\
            + torch.var(myEpi.R0dTaus, dim= 0).sum()
    if torch.isnan(loss).any():
        utils.log_print(printFlag, f"meet nan value at {j}")#
        break
    errorTmp= myloss(predSignal[:, :, 0:-1] ,signal[:, :, 1:])
    losses.append(loss.item())
    predError.append(errorTmp.item())
    loss.backward(retain_graph=True)
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()


inferZmatNp= inferZmat.cpu().detach().numpy().squeeze()


IMatrix= torch.eye(paras.n, device= device)
PreA= A_mat.reverse_A_mat(inferZmat, P)
PreA= PreA*P
PreA= PreA/torch.norm(PreA)

utils.log_print(printFlag,torch.var(myEpi.taus, dim= 0))

fileName= f"{paras.modelLoad}/{paras.modelLoad}_{paras.epi}_{paras.strains}_{paras.n}_{paras.seed}" #SARS_H1N1_COVID
np.savez("results/"+fileName+".npz", A= AdjCurTen.cpu().detach(), Apre= PreA.cpu().detach(),
         loss= losses, tausP= myEpi.taus.cpu().detach(), 
         r0sP= (myEpi.R0dTaus*myEpi.taus).cpu().detach(), signal= signal.cpu().detach(), predSignal= predSignal.cpu().detach())