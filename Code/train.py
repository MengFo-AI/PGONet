import os 
import gc
import torch
import numpy as np 
import pandas as pd 

import time 
import argparse
from tqdm.auto import tqdm
from model import PGONet
from utils import CFG, val_epoch, train_one_epoch
from dataset import ProteinDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader




def trainer(CFG=CFG) :

    model = PGONet(input_dim=CFG.conv_input_dim, num_classes=CFG.num_labels)
    print('=============== Use the PGONet ===============')
    
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
    



if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument("train_size", type = float, default=0.9, help="train / val Dataset random split rate")
    parser.add_argument("wandb", type = bool, default=False, help="Wandb api (default is Flase, if you need please keep it on.)")
    parser.add_argument("weight_decay", type = float, default=0.01, help="weight decay (float)")
    parser.add_argument("num_workers", type = int, default=2)
    parser.add_argument("max_grad_norm", type = float, default=100.0, help = "Grad clip")
    parser.add_argument("n_epochs", type = int, default=100, help = "Model traing epochs")
    parser.add_argument("batch_size", type = int, default=128, help = "Batch size for Protein feature dataset")
    parser.add_argument("lr", type = float, default=1e-5, help = "Training learning rate (default : 1e-5)")
    parser.add_argument("conv_input_dim", type = int, default= 512, help="Model input data dim") 

    CFG.batch_size = parser.batch_size
    CFG.lr = parser.lr 
    CFG.max_grad_norm = parser.max_grad_norm
    CFG.n_epochs = parser.n_epochs
    CFG.num_workers = parser.num_workers
    CFG.wandb = parser.wandb
    CFG.train_size = parser.train_size
    CFG.weight_decay = parser.weight_decay
    CFG.conv_input_dim = parser.conv_input_dim