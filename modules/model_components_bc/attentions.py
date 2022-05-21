import torch
import torch.nn as nn
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
class TanhAttention(nn.Module) :
    def __init__(self, hidden_dim) :
        super(TanhAttention, self).__init__()
        
        self.attn1 =  nn.Linear(hidden_dim*2, 
                                hidden_dim*2 // 2)
        stdv = 1. / math.sqrt(hidden_dim*2)
        self.attn1.weight.data.uniform_(-stdv, stdv)
        self.attn1.bias.data.fill_(0)

        self.attn2 = nn.Linear(hidden_dim*2 // 2, 1, 
                        bias = False)
        stdv = 1. / math.sqrt(hidden_dim * 2 // 2) #*2 after hidden_dim
        self.attn2.weight.data.uniform_(-stdv, stdv)
        
  
    def forward(self, hidden, mask) :
          
     
        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        
        attn2.masked_fill_(mask.bool(), -float('inf'))

        weights = torch.softmax(attn2, dim = -1)
        
        return weights


class DotAttention(nn.Module):
    
    def __init__(self, hidden_dim):
        super(DotAttention, self).__init__()
        
        self.attn1 = nn.Linear(hidden_dim * 2, 1, bias = False)
        stdv = 1. / math.sqrt(hidden_dim*2)
        self.attn1.weight.data.uniform_(-stdv,stdv)

          
    def forward(self, hidden, mask):

        attn1 = (self.attn1(hidden) / (hidden.shape[-1]) ** 0.5).squeeze(-1)
      #  print("attn1 size: ",attn1.size())     
      #  print("mask size: ",mask.size())
        if not attn1.is_cuda:
           attn1 =  attn1.to(device)
       # if not mask.is_cuda:
#        print("device: ",device)
        mask = mask.to(device)
#        print("mask.is_cuda: ",mask.is_cuda)
#        print("attn1: ",attn1)
#        print("mask: ",mask)
        attn1.masked_fill_(mask.bool(), -float('inf'))
    
        weights = torch.softmax(attn1, dim = -1)
        
        return weights



