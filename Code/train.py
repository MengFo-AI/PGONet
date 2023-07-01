import os 
import gc
import torch
import numpy as np 
import pandas as pd 

import time 
from tqdm.auto import tqdm
from model import PGONet
from utils import CFG, val_epoch, train_one_epoch
from dataset import ProteinDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader












model = PGONet(input_dim=CFG.conv_input_dim, num_classes=CFG.num_labels)
print('=============== Use the LinearModel ===============')
    
# About the dataset
train_dataset = ProteinDataset(cfg=CFG)


# Use to training (random split the dataset(train) )
train_set, val_set = random_split(train_dataset,
                                  lengths = [int(len(train_dataset)*CFG.train_size), 
                                             len(train_dataset)-int(len(train_dataset)*CFG.train_size)
                                            ])

# Denfine all DataLoader
train_loader = DataLoader(train_set, #train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers,
                          collate_fn=train_dataset.collect_fn)


val_loader = DataLoader(val_set, 
                        batch_size=192,#CFG.batch_size, 
                        shuffle=True,
                        num_workers=CFG.num_workers,
                        collate_fn=train_dataset.collect_fn)




# Start training........
loss_history = [] # all loss list
best_score = 0.0 # upgrade the best-score


for epoch in tqdm(range(CFG.n_epochs)) :
    # Start training the model......
    
    
    model, score_train, scheduler = train_one_epoch(model=model,
                                                    loader=train_loader,
                                                    epoch=epoch,
                                                    cfg=CFG)
    
    
    
    # Start valid the model....... 
    val_loss_history, val_f1score_history, avg_loss, avg_score = val_epoch(model=model,
                                                                           loader=val_loader,
                                                                           cfg=CFG)
    scheduler.step(avg_loss)
    
    
    
    #if score_train > best_score :
    if avg_score > best_score :
        
        best_score =  avg_score #score_train
        torch.save(model.state_dict(), "PTFModel.bin")
        print("Saving Model ...")
        
    torch.cuda.empty_cache()
    gc.collect()
    
