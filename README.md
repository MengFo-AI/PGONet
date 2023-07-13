# PGONet
PGONet: protein function prediction from sequence

# Use
- about training argument `train.py`
```python
parser.add_argument("train_size", type = float, default=0.9, help="train / val Dataset random split rate")
    parser.add_argument("wandb", type = bool, default=False, help="Wandb api (default is Flase, if you need please keep it on.)")
    parser.add_argument("weight_decay", type = float, default=0.01, help="weight decay (float)")
    parser.add_argument("num_workers", type = int, default=2)
    parser.add_argument("max_grad_norm", type = float, default=100.0, help = "Grad clip")
    parser.add_argument("n_epochs", type = int, default=100, help = "Model traing epochs")
    parser.add_argument("batch_size", type = int, default=128, help = "Batch size for Protein feature dataset")
    parser.add_argument("lr", type = float, default=1e-5, help = "Training learning rate (default : 1e-5)")
    parser.add_argument("conv_input_dim", type = int, default= 512, help="Model input data dim") 
```

- Run
```
>>> python train.py
```

# Reference

- [Fast and accurate protein function prediction from sequence through pretrained language model and homology-based label diffusion](https://github.com/biomed-AI/SPROF-GO/)

