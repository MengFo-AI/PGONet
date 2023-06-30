import os 
import sys 
import gc
import time
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from .utils import CFG
from torch.utils.data import Dataset 




class ProteinDataset(Dataset) :
    def __init__(self, cfg=CFG) :
        super(ProteinDataset, self).__init__()
        
        
        print('Loading data.........')
        embeds = np.load(cfg.train_embeds_path)
        ids = np.load(cfg.train_ids_path)
        print('Loaded data !')
        embeds_list = []
        for I in range(embeds.shape[0]) :
            embeds_list.append(embeds[I, :])
            
        self.df = pd.DataFrame(data={"EntryID": ids, "embed" : embeds_list})
        self.df = self.df.merge(cfg.df_labels, on="EntryID")
        
        gc.collect()
        del embeds_list
        
        
    def __len__(self) :
        return len(self.df)
        
        
    def __getitem__(self, index) :
        
        X = torch.tensor(self.df.iloc[index]['embed'], dtype = torch.float32)
        Y = torch.tensor(self.df.iloc[index]["labels_vect"], dtype = torch.float32)
        
        return X, Y
    
    
    def padding(self, batch, cfg=CFG):
        #print(batch[0])
        maxlen = max([protein_feat.shape[0] for protein_feat,_ in batch])
        batch_protein_feat = []
        batch_protein_mask = []
        batch_protein_label = [label for _,label in batch]
        
        for protein_feat, _ in batch:
            padded_protein_feat = torch.zeros(maxlen, cfg.conv_input_dim)
            padded_protein_feat[:protein_feat.shape[0]] = protein_feat
            batch_protein_feat.append(padded_protein_feat)

            protein_mask = torch.zeros(maxlen)
            protein_mask[:protein_feat.shape[0]] = 1
            batch_protein_mask.append(protein_mask)

        return {'feat':torch.stack(batch_protein_feat), 
                'mask':torch.stack(batch_protein_mask), 
                'label':torch.stack(batch_protein_label)}
        
        
    def collect_fn(self, batch):
        out = self.padding(batch)
        
        return out
        
    