import os
import sys  
import time 
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn 
from .utils import CFG



class PGONet(torch.nn.Module) :
    
    def __init__(self, input_dim:int = CFG.conv_input_dim, num_classes:int = CFG.num_labels, gru_out:int=512, hidden_dim:int=516, num_heads:int = 8, bidirectional:bool=True):
        super(PGONet, self).__init__()
        
        self.ReLu = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(0.15)
        #self.H_BN = torch.nn.BatchNorm1d(1024)
        self.M = torch.nn.LayerNorm(875)
        self.LayerN = torch.nn.LayerNorm(input_dim, eps=1e-6)
        
        # Input Layers
        self.input_linear1 = torch.nn.Linear(input_dim, 1000)
        
        # Hidden Layers
        self.h_linear1 = torch.nn.Linear(1000, 875)
        self.h_linear2 = torch.nn.Linear(875, 756)
        self.h_linear3 = torch.nn.Linear(756, hidden_dim)
        
        
        # Hindden -->> GRU Blocks
        self.H_to_G = torch.nn.Linear(hidden_dim, 816)
        self.GRU = torch.nn.GRU(input_size=816,
                                hidden_size=gru_out,
                                num_layers=3,
                                bidirectional=bidirectional)
        
        
        if bidirectional:
            self.linear4 = torch.nn.Linear(gru_out*2, gru_out*2)
        else :
            self.linear4 = torch.nn.Linear(gru_out, gru_out)
            
            
        # Attention 
        self.Attention = torch.nn.Sequential(
                                  nn.Linear(gru_out*2 if bidirectional else gru_out,
                                            64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, num_heads)
                                 )
        
        
        self.FC1 = torch.nn.Linear(num_heads*gru_out*2 if bidirectional else num_heads*gru_out,
                                   num_heads*gru_out*2 if bidirectional else num_heads*gru_out)
        
        self.FC2 = torch.nn.Linear(num_heads*gru_out*2 if bidirectional else num_heads*gru_out,
                                   num_classes)
        
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    
    
    def forward(self, x, mask):
        
        # Input Blocks
        a_1 = self.LayerN(x)
        a_1 = self.input_linear1(a_1)
        a_1 = self.ReLu(a_1)
        
        
        # Hidden Blocks
        h_1 = self.h_linear1(a_1)
        h_1 = self.ReLu(h_1)
        h_1 = self.M(h_1)
        h_1 = self.h_linear2(h_1)
        h_1 = self.ReLu(h_1)
        h_1 = self.dropout(h_1)
        h_1 = self.h_linear3(h_1)
        
        GRU_Input = self.H_to_G(h_1)
        h_1,_ = self.GRU(GRU_Input)
        h_1 = self.ReLu(h_1)
        h_1 = self.dropout(h_1)
        h_1 = self.linear4(h_1)
        
        
        # Attention Blocks
        atten = self.Attention(h_1)
        atten = atten.masked_fill(mask[:, :, None] == 0, -1e9)
        atten = F.softmax(atten, dim=1)
        atten = atten.transpose(1,2)   # [B, num_heads, L]
        
        h_1 = atten@h_1    # [B, num_heads, hidden_dim]
        h_1 = torch.flatten(h_1, start_dim=1) # [B, num_heads*hidden_dim]
        
        out = self.FC1(h_1)
        out = self.ReLu(out)
        out = self.dropout(out)
        out = self.FC2(out)
        
        return out