# -*- coding: utf-8 -*-
"""
Training Entrance for LightGCN.
Original code inspired by: https://github.com/gusye1234/LightGCN-PyTorch
Modified by: chenz0zz0

Key Optimizations:
1. Early Stopping: Introduced a patience-based early stopping mechanism monitored 
   via Recall@20 to prevent overfitting.
2. Best Weights Auto-Saving: Automatically saves the model state when a new best 
   Recall metric is achieved.
"""

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best_recall = 0.0
patience = 10 
stop_count = 0

if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=1, w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        
        if (epoch + 1) % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            
            cur_recall = results['recall'][0]
            if cur_recall > best_recall:
                best_recall = cur_recall
                stop_count = 0
                torch.save(Recmodel.state_dict(), weight_file)
                world.cprint(f"!!! Best Model Saved (Recall@20: {best_recall:.4f})")
            else:
                stop_count += 1
                print(f"Patience counter: {stop_count}/{patience}")
            
            if stop_count >= patience:
                world.cprint(f"Early stopping at epoch {epoch+1}. Best Recall@20: {best_recall:.4f}")
                break
finally:
    if world.tensorboard:
        w.close()