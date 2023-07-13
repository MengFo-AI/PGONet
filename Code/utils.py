import os 
import sys 
import gc 
import time
import torch 
import pandas as pd 
import numpy as np 


from tqdm.auto import tqdm
from metrics import *
from torchmetrics.classification import MultilabelF1Score



# Device

if torch.cuda.is_available() :
    device = torch.device('cuda')
    os.system('nvidia-smi')
elif torch.backends.mps.is_available() :
    device = torch.device('mps')

else :
    device = torch.device('cpu')




def seed_everything(seed = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(seed=42)

# Config

class CFG:
    train_ids_path:str = '/kaggle/input/t5embeds/train_ids.npy'
    train_embeds_path:str = '/kaggle/input/cafa5-protein-feature/train_embeds.npy'
    df_labels_path:str = '/kaggle/input/cafa5-protein-feature/train_targets(T5)_top500.pkl'
    
    df_labels = pd.read_pickle(df_labels_path)
    print(f'[+] Verify -------->>>>>> {df_labels.labels_vect[0].shape}')
    
    num_labels:int = int(df_labels['labels_vect'][0].shape[0]) 
    print(f'Num of the labels: {num_labels}')
    
    
    train_size:float=0.9#0.9
    apex:bool = False # use amp maybe grad infinity
    wandb:bool = False
        
    metrics = 'f1' # m = matthews  / f1(torch metrics的) / f1_torch(自己写的)
    model_version = 'v3_pca_attention'
    conv_input_dim:int = 512 #512 #1024(T5原始数据) # train data X[0] shape 
        
    weight_decay:float = 0.01
    num_workers:int = 2
    max_grad_norm:float = 100.0 #5.0 #3.0 # 100

    n_epochs:int = 100#04#5 #15 #30 #20
    batch_size:int = 128#128 #256 #32
    lr:float = 1e-5 #0.001 #8e-3 #lr=8e-4
    
    device = device





# Only at the kaggle 
if CFG.wandb:
    
    import wandb

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_api")
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(project='CAFA5-[Protein]-[Main]-[MLP Model]', 
                     name=f'PGONET Linear AMP:{str(CFG.apex)}',
                     config=class2dict(CFG),
                     group=CFG.model_version,
                     job_type="train",
                     anonymous=anony)







def val_epoch(model, loader, cfg=CFG) :
    f1_score = MultilabelF1Score(cfg.num_labels).to(cfg.device)
    criterion = torch.nn.CrossEntropyLoss()
    
    ## VALIDATION PHASE : 
    losses, val_loss_history = [], [] 
    scores, val_f1score_history = [], []
    
    for b in loader:
        embed = b['feat'].to(cfg.device)
        targets = b['label'].to(cfg.device)
        mask = b['mask'].to(cfg.device)
        
        preds = model(embed, mask)
        loss= criterion(preds, targets)
        if cfg.metrics == 'm' :
            score=Matthews(targets, preds)
        elif cfg.metrics == 'f1_torch' :
            score=f1_torch(targets, preds)
        else :
            score=f1_score(preds, targets)
            
        losses.append(loss.item())
        scores.append(score.item())
        
    avg_loss = np.mean(losses)
    avg_score = np.mean(scores)
    print("Running Average VAL Loss : ", avg_loss)
    print("Running Average VAL F1-Score : ", avg_score)
    val_loss_history.append(avg_loss)
    val_f1score_history.append(avg_score)
    
    if cfg.wandb:
        wandb.log({f"Avg Loss":avg_loss,
                   f"Avg VAL F1-Score":avg_score})
                
    return val_loss_history, val_f1score_history, avg_loss, avg_score





def train_one_epoch(model, loader, epoch, cfg=CFG):
    
    model.to(cfg.device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()#BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    
    f1_score = MultilabelF1Score(cfg.num_labels).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.5,
                                                           patience=4)#1)
    
    #################
    loss_sum:float = 0.0
    count_loss:float = 0.0
    #################
    
    for m in tqdm(loader):
        x = m['feat'].to(cfg.device).float()
        mask = m['mask'].to(cfg.device).float()
        y = m['label'].to(cfg.device).float()
        batch_size = y.shape[0]
        
        optimizer.zero_grad()
        y_pred = model(x, mask)
        loss = criterion(y_pred, y)
                
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.max_grad_norm)

        optimizer.step()
        
        #################
        count_loss += batch_size
        loss_sum += loss.item() * batch_size
        avg_loss = loss_sum / count_loss
        
        if cfg.metrics == 'm' :
            score_=Matthews(y, y_pred)
        elif cfg.metrics == 'f1_torch' :
            score_=f1_torch(y, y_pred)
        else :
            score_ = f1_score(y_pred, y)
        
        # Wandb
        if cfg.wandb:
            wandb.log({f"Train Loss":avg_loss,
                       f"Train F1-Score":score_,
                       f"Train Grad":grad_norm})
        
        
            
    print(f"Epoch[{epoch}] [+] Train Loss: {(loss_sum/len(loader)):.04f}| Avg Loss: {(loss_sum/count_loss):.04f}, Score ({cfg.metrics}): {score_}, Grad: {(grad_norm):.04f}")
    
    return model, score_, scheduler

