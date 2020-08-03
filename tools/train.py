import argparse
import os
import sys
import time
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lib.utils import *
from lib.dataset_online import SelfDatasetsOnline
from models.pfld_avg import PFLD_AVG
from models.pfld_half import PFLD_HALF
from losses.wing_loss import WingLoss
from losses.adaptive_wing_loss import AdaptiveWingLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Training infos')
    
    parser.add_argument('--data_type',type=str,default="WFLW")
    parser.add_argument('--dataset_exp',type=int,default=1)
    parser.add_argument('--model_type',type=str,default="PFLD_AVG")
    parser.add_argument('--resume_checkpoints',type=str,default="")
    parser.add_argument('--pretrained',type=str,default="")
    parser.add_argument('--lr_reduce_patience',type=int,default=0,help="use ReduceLROnPlateau")
    parser.add_argument('--model_dir',type=str,default="pfld_native",help="model save in checkpoint/data_type/model_dir/*_checkpoint.pth")
    parser.add_argument('--load_epoch',type=int,default=0,help="wheather load epoch and lr when restore checkpoint. 0 is no, others is yes")
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--gpus',type=str,default='0,1,2,3',help="Set gpus environ_devices")
    parser.add_argument('--batch_size',type=int,default=320,help="batch size on one gpu")
    parser.add_argument('--loss_type',type=str,default='L2')
    parser.add_argument('--workers',type=int,default=8)
    # Heatmap hyper-parameter
    parser.add_argument('--image_size',type=int,default=112)
    return parser.parse_args()

def main(args):

    cfg,logger,best_nme,model_save_dir,last_epoch,end_epoch = Init(args)

    os.environ['CUDA_VISIBLE_DEVICES'] =  args.gpus #",".join(str(gpu) for gpu in cfg["GPUS"])
    gpu_nums = len(args.gpus.split(',')) # len(cfg["GPUS"])
    model = None
    if args.model_type in ["PFLD_AVG","pfld_avg"]:
        model = PFLD_AVG(cfg["NUM_JOINTS"]*2)
    elif args.model_type in ["PFLD_HALF","pfld_half"]:
        model = PFLD_HALF(cfg["NUM_JOINTS"]*2)
    elif args.model_type in ["PFLD_FORTH","pfld_forth","pfld_0.25"]:
        model = PFLD_HALF(cfg["NUM_JOINTS"]*2,4)
    else:
        raise("Not support {} mode now.".format(args.model_type))
    model.init_weights(pretrained=args.pretrained)
    
    if args.loss_type in ["L2","l2","MSE","mse"]:
        criterion = nn.MSELoss(size_average=True).cuda()
    elif args.loss_type in ["L1","l1"]:
        criterion = nn.L1Loss(size_average=True).cuda()
    elif args.loss_type in ["Smooth_L1","SMOOTH_L1"]:
        criterion = nn.SmoothL1Loss(size_average=True).cuda()
    elif args.loss_type in ["WING","wing"]:
        criterion = WingLoss().cuda()
    elif args.loss_type in ["AWING","ADAPTIVE_WING"]:
        criterion = AdaptiveWingLoss().cuda()
    else:
        raise("Not support {} loss now.".format(args.loss_type))

    model = nn.DataParallel(model,range(gpu_nums)).cuda()
    devices = torch.device("cuda:0")
    model.to(devices)

    optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            weight_decay=1e-6,
        )
    
    if os.path.exists(args.resume_checkpoints) or os.path.islink(args.resume_checkpoints):
        checkpoint = torch.load(args.resume_checkpoints)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Restore epoch {} from {}".format(checkpoint['epoch'] , args.resume_checkpoints))
        best_nme = checkpoint['best_nme']
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg["LR_STEP"],
        0.1, last_epoch-1
    )

    if args.lr_reduce_patience:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,mode='min',factor=0.1,patience=args.lr_reduce_patience,threshold=1e-4
        )
        
    last_epoch = args.load_epoch
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    
    train_dataset = SelfDatasetsOnline(cfg,is_train=True,transforms=transform,dataset_exp=args.dataset_exp)
    test_dataset = SelfDatasetsOnline(cfg,is_train=False,transforms=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size*gpu_nums,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=(args.batch_size//2)*gpu_nums,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    for epoch in range (last_epoch,end_epoch):
       
        logger.info("Use {} train epoch {}".format(lr_repr(optimizer),epoch))
        
        # train direct coordinate regression
        start_time = time.time()
        loss,ocular_nme,pupil_nme = train(model,train_loader,criterion,optimizer,epoch)
        logger.info("{}'epoch train time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(epoch,time.time()-start_time,loss,ocular_nme*100.0,pupil_nme*100.0))
        
        # validation
        start_test_time = time.time()
        loss,ocular_nme,pupil_nme = test(model,test_loader,criterion)
        logger.info("{}'epoch test  time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(epoch,time.time()-start_test_time,loss,ocular_nme*100.0,pupil_nme*100.0))
        
        # save 
        if ocular_nme < best_nme['nme'] :
            best_nme = {'epoch':epoch,'loss':loss,'nme':ocular_nme,'pupil_nme':pupil_nme}
            logger.info('epoch {} reach better, save {}_checkpoint.pth'.format(epoch,epoch))    
            save_checkpoint(model.state_dict(),best_nme,optimizer.state_dict(),model_save_dir)

        # lr.step
        if args.lr_reduce_patience:
            # reduceLROnPlateau
            lr_scheduler.step(loss)
        else:
            # MultiStepLR or StepLR
            lr_scheduler.step()
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
