"""
    control main task by ratio not models
"""

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
from lib.multi_dataset import MULTI_DataLoader
from models.multi_flan import MULTI_FLAN
from models.multi_flan2 import MULTI_FLAN2

same_index = {
    '98':[33,46,60,64,68,72,54,76,82,16],
    '68':[17,26,36,39,42,45,30,48,54,8],
    '29':[0,1,8,10,11,9,20,22,23,28],
    '19':[0,5,6,8,9,11,13,15,17,18]
}

def parse_args():
    parser = argparse.ArgumentParser(description='Training infos')
    parser.add_argument('--main_data',type=str,default="COFW")
    parser.add_argument('--aux_datas',type=str,default="WFLW")
    parser.add_argument('--aux_ratios',type=str,default="3,1")
    parser.add_argument('--ratios_decay',type=float,default=1.0)
    parser.add_argument('--loss_alpha',type=float,default=0.5)
    parser.add_argument('--dataset_exp',type=int,default=10)

    parser.add_argument('--model_type',type=str,default="MULTI_FLAN2")
    parser.add_argument('--resume_checkpoints',type=str,default="")
    parser.add_argument('--model_dir',type=str,default="Mix")
    
    parser.add_argument('--lr_reduce_patience',type=int,default=0)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--gpus',type=str,default="0")
    parser.add_argument('--batch_size',type=int,default=48)
    parser.add_argument('--loss_type',type=str,default="L1")

    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--image_size',type=int,default=112)
    
    parser.add_argument('--show_others',default=False,action='store_true')
    parser.add_argument('--mix_loss',default=False,action='store_true')
    parser.add_argument('--auxdata_aug',default=False,action='store_true')

    return parser.parse_args()

def Init_mix(args):
    args.aux_datas = args.aux_datas.split(',')
    args.aux_ratios = np.asarray(args.aux_ratios.split(','),dtype=np.float32)

    with open(os.path.join('experiments','{}.json'.format(args.main_data))) as f:
        main_cfg = json.load(f)

    aug_cfgs = {}

    for aux_data in args.aux_datas:
        with open(os.path.join('experiments','{}.json'.format(aux_data))) as f:
            aux_cfg = json.load(f)
            aug_cfgs[str(aux_cfg["NUM_JOINTS"])] = aux_cfg

    return args, main_cfg , aug_cfgs

def test_models(models,test_loader,criterion,model_index):
    losses = AverageMeter()
    losses.reset()
    models.eval()

    interocular_sum_nme = 0
    inter_pupil_sum_nme = 0

    with torch.no_grad():
        for i,(imgs,landmarks) in enumerate(test_loader):
            imgs = imgs.cuda(non_blocking=True)
            preds_array = models(imgs)
            preds = preds_array[model_index].cuda()
            landmarks = landmarks.cuda(non_blocking=True)
            loss = criterion(preds,landmarks)

            interocular_batch_nme , inter_pupil_batch_nme = compute_nme(preds.cpu(),landmarks.cpu())
            interocular_sum_nme += np.sum(interocular_batch_nme)
            inter_pupil_sum_nme += np.sum(inter_pupil_batch_nme)
            
            losses.update(loss.item(),landmarks.size(0))

    interocular_nme = interocular_sum_nme / len(test_loader.dataset)# test_loader.sampler.num_samples
    inter_pupil_nme = inter_pupil_sum_nme / len(test_loader.dataset)# test_loader.sampler.num_samples

    return losses.avg ,interocular_nme , inter_pupil_nme


def main(args):

    args, main_cfg, cfgs = Init_mix(args)
    logger = create_logger("Mix_{}_{}_{}".format(args.main_data,args.model_type,args.model_dir))
    logger.info(args)
    model_save_dir = os.path.join('checkpoint','Mix',args.main_data,args.model_dir)
    check_mkdir(model_save_dir)
    
    os.environ['CUDA_VISIBLE_DEVICES'] =  args.gpus #",".join(str(gpu) for gpu in cfg["GPUS"])
    gpu_nums = len(args.gpus.split(',')) # len(cfg["GPUS"])

    landmark_nums = [main_cfg["NUM_JOINTS"]*2]
    for key in cfgs.keys():
        landmark_nums.append(cfgs[key]["NUM_JOINTS"]*2)

    # define models
    if args.model_type in ["multi_flan","MULTI_FLAN"]:
        models = MULTI_FLAN(landmark_nums) #models[0] is main_

    elif args.model_type in ["multi_flan2","MULTI_FLAN2"]:
        models = MULTI_FLAN2(landmark_nums) #models[0] is main_

    elif args.model_type in ["multi_flan2_0.5","MULTI_FLAN2_HALF"]:
        models = MULTI_FLAN2(landmark_nums,2) #models[0] is main_

    elif args.model_type in ["multi_flan2_0.25","MULTI_FLAN2_FORTH","MULTI_FLAN2_QUARTER"]:
        models = MULTI_FLAN2(landmark_nums,4) #models[0] is main_
        
    else:
        print("ERROR :Not support {} network".format(args.model_type))
        exit()
    
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
    
    models = nn.DataParallel(models,range(gpu_nums)).cuda()
    devices = torch.device("cuda:0")
    models.to(devices)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, models.parameters()),
        lr = args.lr,
        weight_decay=1e-6
    )
    
    if args.lr_reduce_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,mode='min',factor=0.1,patience=args.lr_reduce_patience,threshold=1e-4
        )

    else :
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, main_cfg["LR_STEP"],
                0.1, -1
        )

    # best_loss = 10 # define a big number
    best_loss = float("inf")
    if os.path.exists(args.resume_checkpoints):
        checkpoint = torch.load(args.resume_checkpoints)    
        models.load_state_dict(checkpoint['state_dict'])        
        best_nme = checkpoint['best_nme']
        best_loss = best_nme['loss']

        
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    
    main_train_dataset = SelfDatasetsOnline(main_cfg,is_train=True,transforms=transform,dataset_exp=args.dataset_exp,image_size=args.image_size)
    main_test_dataset = SelfDatasetsOnline(main_cfg,is_train=False,transforms=transform,image_size=args.image_size)

    aux_dataset = {}
    for key in cfgs.keys():
        if(cfgs[key] != main_cfg):
            aux_dataset[key] = {}
            aux_dataset[key]["train"] = SelfDatasetsOnline(cfgs[key],is_train=True,transforms=transform,dataset_exp=1,image_size=args.image_size,data_aug=args.auxdata_aug)
            if args.show_others:
                aux_dataset[key]["test"] = SelfDatasetsOnline(cfgs[key],is_train=False,transforms=transform,image_size=args.image_size)

    main_train_loader = DataLoader(
        dataset = main_train_dataset,
        batch_size= args.batch_size * gpu_nums,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    main_test_loader = DataLoader(
        dataset = main_test_dataset,
        batch_size = args.batch_size * gpu_nums,
        shuffle= False,
        num_workers= args.workers,
        pin_memory= True
    )

    aux_loader = {"train":[],"test":{}} # train is diff from test
    for key in aux_dataset.keys():
        aux_loader['train'].append(DataLoader(
            dataset = aux_dataset[key]['train'],
            batch_size = args.batch_size * gpu_nums,
            shuffle = True,
            num_workers=args.workers,
            pin_memory= True
        ))
        if args.show_others:
            aux_loader['test'][key] = DataLoader(
                dataset = aux_dataset[key]['test'],
                batch_size = args.batch_size * gpu_nums,
                shuffle = False,
                num_workers=args.workers,
                pin_memory= False
            )

    mix_train_dataloader = MULTI_DataLoader(main_train_loader,aux_loader['train'],args.aux_ratios)
    ratio_speed_array = [1] + [args.ratios_decay] * (args.aux_ratios.size -1) # each epoch ratio will reduce

    for epoch in range(0,main_cfg['END_EPOCH']):
        logger.info("Use {} train epoch {}".format(lr_repr(optimizer),epoch))
        
        # train
        mix_train_dataloader.init_iter()
        train_loss = AverageMeter()
        train_loss.reset()

        models.train()

        # one epoch training
        start_time = time.time()
        while mix_train_dataloader.get_iter_flag():
            
            imgs,landmarks = mix_train_dataloader.get_iter() # landmark is normlised
            imgs = imgs.cuda(non_blocking=True)
            landmarks = landmarks.cuda(non_blocking=True)
            current_landmark_num = landmarks.size(1)
            current_index = landmark_nums.index(current_landmark_num)

            pre_landmarks_array = models(imgs)
            
            preds_landmarks = pre_landmarks_array[current_index].cuda()

            loss = criterion(preds_landmarks,landmarks)
            
            if args.mix_loss:
                preds_landmarks = preds_landmarks.view(preds_landmarks.size(0),-1,2)
                landmarks = landmarks.view(landmarks.size(0),-1,2)
                main_indexs = same_index[str(main_cfg["NUM_JOINTS"])]

                if current_landmark_num // 2 == main_cfg["NUM_JOINTS"]:
                    
                    for key in cfgs.keys():
                        temp_landmark_index = landmark_nums.index(cfgs[key]["NUM_JOINTS"]*2)
                        temp_head_landmarks = pre_landmarks_array[temp_landmark_index].cuda().view(preds_landmarks.size(0),-1,2)
                        temp_indexs = same_index[str(cfgs[key]["NUM_JOINTS"])]
                        loss = loss + args.loss_alpha * criterion(preds_landmarks[:,main_indexs],temp_head_landmarks[:,temp_indexs])

                else:
                    aux_indexs = same_index[str(current_landmark_num // 2)]     
                    main_landmark_index = landmark_nums.index(main_cfg["NUM_JOINTS"]*2)
                    main_head_landmarks = pre_landmarks_array[main_landmark_index].cuda().view(preds_landmarks.size(0),-1,2)
                    loss = loss + args.loss_alpha * criterion(main_head_landmarks[:,main_indexs],preds_landmarks[:,aux_indexs])
                

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(),landmarks.size(0))

        logger.info("{}'epoch train time :{:<6.2f}s loss :{:.8f} ".format(epoch,time.time()-start_time,train_loss.avg))
        
        # adjust dataset ratios
        args.aux_ratios = args.aux_ratios * ratio_speed_array
        mix_train_dataloader.change_ratios(args.aux_ratios)
        print("Change Traing data ratios : {}".format(args.aux_ratios))
        
        # test 
        ## test main task
        start_test_time = time.time()
        loss,ocular_nme,pupil_nme = test_models(models,main_test_loader,criterion,0)

        # save best model
        if loss < best_loss:
            best_loss = loss
            best_nme = {'epoch':epoch,'loss':loss,'nme':ocular_nme,'pupil_nme':pupil_nme}
            logger.info('epoch {} reach better, save {}_checkpoint.pth'.format(epoch,epoch))    
            file_path = os.path.join(model_save_dir,"{}_checkpoint.pth".format(epoch))
            best_path = os.path.join(model_save_dir,"best.pth")
            torch.save({
                "best_nme":best_nme,
                "state_dict":models.state_dict(),
                "epoch":epoch
            },file_path)
            if os.path.islink(best_path):
                os.remove(best_path)
            # symlink is create a relative path file : a is exist file and relative path,b is link and absolute path
            os.symlink(os.path.join("./","{}_checkpoint.pth".format(epoch)),best_path)
        
        logger.info("{}'epoch {} test time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(epoch,main_cfg["NUM_JOINTS"],time.time()-start_test_time,loss,ocular_nme*100.0,pupil_nme*100.0))

        # lr.step
        if args.lr_reduce_patience:
            scheduler.step(loss)
        else :
            scheduler.step()

        if args.show_others:
            ## test aux task
            for key in aux_loader['test'].keys():
                start_test_time = time.time()
                current_landmark_num = cfgs[key]["NUM_JOINTS"]
                current_index = landmark_nums.index(current_landmark_num//2)
                loss,ocular_nme,pupil_nme = test_models(models,aux_loader['test'][key],criterion,current_index) 
                logger.info("{}'epoch {} test time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(epoch,key,time.time()-start_test_time,loss,ocular_nme*100.0,pupil_nme*100.0))



if __name__ == "__main__":
    args = parse_args()
    main(args)
