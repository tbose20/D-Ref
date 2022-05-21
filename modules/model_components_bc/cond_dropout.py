import torch
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig
import json
from torch.autograd import Variable

with open('modules/config.txt', 'r') as f:
    args = json.load(f)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Cond_dropout(nn.Module):
    def __init__(self):
        
        super(Cond_dropout, self).__init__()
        #if p < 0 or p > 1:
        #     raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        #self.p = p

        
    def forward(self, X, IND_drop):

        if self.training:
                #print("filter_ids_FP is cuda: ",filter_ids_FP)
                #filter_ids_FP = [X[0,1],X[0,2]]
                #print("filter_ids_FP is cuda: ",filter_ids_FP)

                #print("X is cuda: ",X.is_cuda)

           #     IND_drop = Variable(torch.ones(X.shape)).to(device)
           #     for i in range(len(filter_ids_FP)):
           #        ind_fill = (X == filter_ids_FP[i]).nonzero()
           #      #  print("filter_probs_FP: ",filter_probs_FP[i])
           #      #  print("ind_fill shape: ",ind_fill.shape[1],ind_fill)
           #        ones_fill = Variable(torch.ones(ind_fill.shape[1])).to(device)
           #      #  print("ones_fill: ",ones_fill)
           #        value = ones_fill*filter_probs_FP[i]
           #      #  print("value to fill: ",value)
           #        IND_drop.index_put_(tuple(ind_fill.t()), value)
           #     
           #     for i in range(len(filter_ids_FP_TP)):
           #        ind_fill = (X == filter_ids_FP_TP[i]).nonzero()
           #        ones_fill = Variable(torch.ones(ind_fill.shape[1])).to(device)
           #        value = ones_fill*filter_probs_FP_TP[i]
           #        IND_drop.index_put_(tuple(ind_fill.t()), value)
                
                #print("IND_drop: ",IND_drop)
                #print("X shape: ",X.shape)
                IND_drop = torch.unsqueeze(IND_drop,2) #adding a third dimension matching the shape with X
                #print("IND_drop shape: ",IND_drop.shape)
                bern = torch.bernoulli(1-IND_drop)
                #print("bern: ",bern)
                #binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
                #return X * binomial.sample(X.size()) * (1.0/(1-self.p))
                return X*bern  #TODO scaling or not?
        return X

