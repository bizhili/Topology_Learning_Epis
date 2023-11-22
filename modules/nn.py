import torch

#neural network to compute similarity of two metapopulation nodes
class matchingA(torch.nn.Module):
    def __init__(self, timeDim, strainDim, n, midLayer= 50, device= "cpu"):
        super(matchingA, self).__init__()
        self.Wu= torch.nn.Linear(timeDim, midLayer, device= device)
        self.Wv= torch.nn.Linear(timeDim, midLayer, device= device)

        #self.midU= torch.nn.Linear(midLayer, midLayer, device= device)
        #self.midV= torch.nn.Linear(midLayer, midLayer, device= device)

        self.Wnorm= torch.nn.Linear(strainDim, 1, bias= False, device= device)
        self.myEye= torch.eye(n, dtype= torch.float32, device= device)
        self.myMask= torch.ones(n, dtype= torch.float32, device= device)- self.myEye
        self.myRelu= torch.nn.Sigmoid()
    def forward(self, x): #[50, 1, 36]
        transU= (self.Wu(x))[:, None, ...]#[50, 1, 1, 36]
        transV= (self.Wv(x))[None, ...]   #[1, 50, 1, 36]
        Atemp= (transU*transV).sum(dim=-1)
        Anorm= self.Wnorm(Atemp)
        Ainfer= self.myRelu(Anorm)*self.myMask[..., None]
        return Ainfer.squeeze()
    
#neural network to compute the SIR spidemic gradient
class EpisA(torch.nn.Module):
    def __init__(self, input_dim= 20, num_heads= 1, n= 50, device= "cpu"):
        super(EpisA, self).__init__()
        self.device= device
        self.num_heads= num_heads
        self.n= n
        self.taus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*6.2
        self.taus= torch.nn.Parameter(self.taus)
        self.R0dTaus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*1.35
        self.R0dTaus= torch.nn.Parameter(self.R0dTaus)
        self.mat, self.mask= self.create_temporal_mat(input_dim)
        self.myRelu= torch.nn.ReLU()
        self.mySig= torch.nn.Sigmoid()
        self.mySoft= torch.nn.Softmax(dim=2)
        self.myEye= torch.eye(n, dtype= torch.float32, device= device)

    def alpha(self, i, R0, tau):
        return 1-torch.exp(-(R0/tau)*i)
    
    def create_temporal_mat(self, lang):
        mat= torch.zeros((lang, lang), dtype= torch.float32, device= self.device)
        mask= torch.zeros((lang, lang), dtype= torch.float32, device= self.device)
        for i in range(lang):
            for j in range(i+1):
                mat[i, j]= i- j
                mask[i, j]= 1
        return mat[None, None, ...], mask[None, None, ...]

    def forward(self, x, Amat= None): # shape: (1, 2, 20), dim of nodes, dim of heads, dim of signal
        # divide= self.mySoft(self.output)*x[:, :, -1:]#(1, 2, 20), dim of nodes, dim of heads, dim of signal
        # divide= divide.transpose(1, 2)
        tempAmat= Amat.T+ self.myEye
        #noise= x[:, 0, :] #\noise delta S
        signal= self.myRelu(x) #\delta S
        Ss= 1- torch.cumsum(signal, dim= -1) #easiy negative
        IsMat= torch.exp(self.mat*torch.log(1-1/self.taus[... , None, None]))*self.mask
        Is= torch.matmul(IsMat, signal[..., None]).squeeze(dim=-1)
        alpha= (1-torch.exp(-self.R0dTaus[... , None]*Is))
        temp= tempAmat[..., None, None]*alpha[:, None, ...]
        Alpha= temp.sum(dim= 0)
        predSignal= Alpha*Ss
        #signalPredict= self.alpha(Is[0: -1], R0, tau)*Ss[0:-1] 
        return predSignal, signal, tempAmat.T

#neural network to compute similarity of two metapopulation nodes
class matchingB(torch.nn.Module):
    def __init__(self, timeDim, strainDim, n, midLayer= 50, device= "cpu"):
        super(matchingB, self).__init__()

        self.Amat= torch.randn((n, n), dtype= torch.float32, device=device)-4
        self.AmatMask= 1- torch.eye(n, dtype= torch.float32, device=device)
        self.Amat= torch.nn.Parameter(self.Amat)
        self.mySig= torch.nn.Sigmoid()

    def forward(self, x): #[50, 1, 36]

        return self.mySig(self.Amat)*self.AmatMask

class EpisB(torch.nn.Module):
    def __init__(self, input_dim= 20, num_heads= 1, n= 50, device= "cpu"):#
        super(EpisB, self).__init__()
        self.device= device
        self.num_heads= num_heads
        self.n= n
        self.taus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*2
        self.taus= torch.nn.Parameter(self.taus)
        self.R0dTaus= torch.ones((n, num_heads), dtype= torch.float32, device=device)*1
        self.R0dTaus= torch.nn.Parameter(self.R0dTaus)
        self.myRelu= torch.nn.ReLU()
        self.mySig= torch.nn.Sigmoid()
        self.mySoft= torch.nn.Softmax(dim=2)
        self.Amateye= torch.eye(n, dtype= torch.float32, device=device)

    def alpha(self, i, R0, tau):
        return 1-torch.exp(-(R0/tau)*i)

    def core_function(self, S, I, tempAmat):#x: (50, 2, 1), dim of nodes, dim of heads, (S, I)
        alpha= (1-torch.exp(-self.R0dTaus[... , None]*I))
        temp= tempAmat[..., None, None]*alpha[:, None, ...]
        Alpha= temp.sum(dim= 0)
        dS= Alpha*S
        S= S- dS
        I= I-I/self.taus[..., None]+dS
        return S, I, dS


    def forward(self, x, Amat): # shape: (50, 2, 20), dim of nodes, dim of heads, dim of signal
        tempAmat= self.mySig(Amat.T)/10+ self.Amateye
        signal= self.myRelu(x) #\delta S
        timeHorizonT= signal.shape[2]
        IT= signal[:, :, 0:1]
        ST= torch.ones_like(IT, device= self.device)- IT
        predSignal= []
        for _ in range(timeHorizonT):
            ST, IT, dS= self.core_function(ST, IT, tempAmat)
            predSignal.append(dS.clone())

        predSignal= torch.stack(predSignal, dim= -1).squeeze(dim= -2)

        return predSignal, signal, tempAmat.T

