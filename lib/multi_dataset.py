import argparse
import os
import sys
import time
sys.path.append('.')

import numpy as np
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lib.dataset_online import SelfDatasetsOnline
import random


class MULTI_DataLoader(object):
    def __init__(self,main_loader,aux_loaders,ratios):
        """
            ratios size = aux_loader size + 1
            ratios : [5,1,2] present main_loader : aux_loader[0] : aux_loader[1] = 5:1:2
        """
        self.main_loader = main_loader
        self.aux_loaders = aux_loaders
        self.ratios = ratios / np.sum(ratios)
        self.main_iter = iter(main_loader)
        self.aux_iters = [iter(aux_loader) for aux_loader in aux_loaders]
        self.max_iter_size = np.floor(len(self.main_loader.dataset) / self.main_loader.batch_size).astype(np.int32) if self.main_loader.drop_last else np.ceil(len( self.main_loader.dataset) / self.main_loader.batch_size).astype(np.int32)
        self.current_iter = 0

    def init_iter(self):
        self.main_iter = iter(self.main_loader)
        self.current_iter = 0
        # self.aux_iters = [iter(aux_loader) for aux_loader in aux_loaders]
        
    def get_iter_flag(self):
        return self.current_iter < self.max_iter_size

    def get_iter_num(self):
        return self.current_iter
    
    def change_ratios(self,ratios):
        self.ratios = ratios / np.sum(ratios)

    def get_iter(self):
        
        if not self.get_iter_flag():
            print("Main task dataset has finished one epoch")
            return None
        
        random_seed = random.random()
        loader_output = None
        if random_seed <= self.ratios[0]:
        
            try:
                loader_output = next(self.main_iter)
                self.current_iter += 1
            except  StopIteration:
                raise("Current iter size is overflow .")

        else :
            random_seed = random_seed - self.ratios[0]
            for i in range(len(self.aux_iters)):
                if random_seed <= self.ratios[i]:
                    try:
                        loader_output = next(self.aux_iters[i])
                    except StopIteration:
                        self.aux_iters[i] = iter(self.aux_loaders[i])
                        loader_output = next(self.aux_iters[i])
                    break
                random_seed = random_seed -self.ratios[i]

        return loader_output


if __name__ == "__main__":

    with open(os.path.join('experiments','WFLW.json')) as f:
            cfg = json.load(f)
    main_dataset = SelfDatasetsOnline(cfg,is_train=False,heatmap_sigma=1.5,heatmap_size=64)
    main_loader = DataLoader(
            dataset=main_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True
    )

    with open(os.path.join('experiments','300W.json')) as f:
            cfg = json.load(f)
    aux300W_dataset = SelfDatasetsOnline(cfg,is_train=True,heatmap_sigma=1.5,heatmap_size=64)
    aux300W_loader = DataLoader(
            dataset=aux300W_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=False
    )

    multi_dataloader = MULTI_DataLoader(main_loader,[aux300W_loader],[6,1])
    
    main_num = 0
    aux_num = 0
    while 1 :
        multi_dataloader.init_iter()
        while multi_dataloader.get_iter_flag():

            imgs , landmarks ,heatmaps = multi_dataloader.get_iter()
            if landmarks.shape[1] == 98 :
                main_num = main_num + imgs.shape[0]
            else :
                aux_num = aux_num + imgs.shape[0]
   
        print(main_num,aux_num)