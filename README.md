# PGONet
PGONet: protein function prediction from sequence

# Use
- 修改`utils.py`中的CFG
```python
# change cfg class

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
```

- Run
```
>>> python train.py
```
