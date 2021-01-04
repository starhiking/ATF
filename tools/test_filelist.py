import argparse
import os
import sys
import time
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lib.utils import *
from lib.dataset_online import SelfDatasetsOnline
from models.pfld_avg import PFLD_AVG
from models.pfld_half import PFLD_HALF
from models.multi_flan import MULTI_FLAN
from models.multi_flan2 import MULTI_FLAN2

from losses.wing_loss import WingLoss
from losses.adaptive_wing_loss import AdaptiveWingLoss

test_files = {
    "WFLW":None,
    "300W":None,
    "AFLW":None,
    "COFW":None
}

test_files["WFLW"] = [ 
    "native_dataset/WFLW/annotations/face_landmarks_wflw_test_blur_with_box.csv",
    "native_dataset/WFLW/annotations/face_landmarks_wflw_test_expression_with_box.csv",
    "native_dataset/WFLW/annotations/face_landmarks_wflw_test_illumination_with_box.csv",
    "native_dataset/WFLW/annotations/face_landmarks_wflw_test_largepose_with_box.csv",
    "native_dataset/WFLW/annotations/face_landmarks_wflw_test_makeup_with_box.csv",
    "native_dataset/WFLW/annotations/face_landmarks_wflw_test_occlusion_with_box.csv",
    "native_dataset/WFLW/annotations/face_landmarks_wflw_test_with_box.csv"
    ]

test_files["300W"] = [
    "native_dataset/300W/annotations/face_landmarks_300w_valid_challenge_with_box.csv",
    "native_dataset/300W/annotations/face_landmarks_300w_valid_common_with_box.csv",
    "native_dataset/300W/annotations/face_landmarks_300w_valid_with_box.csv",
    "native_dataset/300W/annotations/face_landmarks_300w_test_with_box.csv"
    ]

test_files["AFLW"] = [
    "native_dataset/AFLW/annotations/face_landmarks_aflw_test_frontal_with_box.csv",
    "native_dataset/AFLW/annotations/face_landmarks_aflw_test_with_box.csv"
]

test_files["COFW"] = [
    "native_dataset/COFW/annotations/face_landmarks_cofw_test_with_box.csv"
]

def parse_args():
    parser = argparse.ArgumentParser(description='Training infos')
    
    parser.add_argument('--model_type',type=str,default="MULTI_FLAN2")
    parser.add_argument('--resume_checkpoints',type=str,default="checkpoint/300W/PFLD_AVG_base_COFW_1e-2/best.pth")
    parser.add_argument('--data_type',type=str,default="300W")
    parser.add_argument('--aux_datas',type=str,default="COFW")
    parser.add_argument('--gpus',type=str,default='0',help="Set gpus environ_devices")
    parser.add_argument('--batch_size',type=int,default=50, help="batch size on one gpu")
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--show_others',default=False,action="store_true")

    parser.add_argument('--image_size',type=int,default=112)

    #unused
    parser.add_argument('--lr',type=float,default=1)
    parser.add_argument('--model_dir',type=str,default="")

    
    return parser.parse_args()

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
            preds = preds_array.cuda()
            landmarks = landmarks.cuda(non_blocking=True)
            loss = criterion(preds,landmarks)

            interocular_batch_nme , inter_pupil_batch_nme = compute_nme(preds.cpu(),landmarks.cpu())
            interocular_sum_nme += np.sum(interocular_batch_nme)
            inter_pupil_sum_nme += np.sum(inter_pupil_batch_nme)
            
            losses.update(loss.item(),landmarks.size(0))

    interocular_nme = interocular_sum_nme / len(test_loader.dataset)# test_loader.sampler.num_samples
    inter_pupil_nme = inter_pupil_sum_nme / len(test_loader.dataset)# test_loader.sampler.num_samples

    return losses.avg ,interocular_nme , inter_pupil_nme

def get_model(cfg):
    model = None
    if args.model_type in ["PFLD","pfld"]:
        model = PFLD(cfg["NUM_JOINTS"]*2)
    elif args.model_type in ["PFLD_AVG","pfld_avg"]:
        model = PFLD_AVG(cfg["NUM_JOINTS"]*2)
    elif args.model_type in ["PFLD_HALF","pfld_half"]:
        model = PFLD_HALF(cfg["NUM_JOINTS"]*2)
    else:
        raise("Not support {} mode now.".format(args.model_type))
    return model

def load_model(model, gpu_nums, args):
    model = nn.DataParallel(model, range(gpu_nums)).cuda()
    devices = torch.device("cuda:0")
    model.to(devices)

    if os.path.isfile(args.resume_checkpoints) or os.path.islink(args.resume_checkpoints):
        pretrained_dict = torch.load(args.resume_checkpoints)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']

        model.load_state_dict(pretrained_dict)
    return model

def main(args):
    cfg,logger,best_nme,model_save_dir,last_epoch,end_epoch = Init(args)
    os.environ['CUDA_VISIBLE_DEVICES'] =  args.gpus 
    gpu_nums = len(args.gpus.split(','))

    model = get_model(cfg)
    model = load_model(model, gpu_nums, args)
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

    criterion = nn.L1Loss(size_average=True).cuda()

    for test_csv in test_files[args.data_type]:
        cfg['Test_csv'] = test_csv
        logger.info("Test {}".format(test_csv))

        test_dataset = SelfDatasetsOnline(cfg, is_train=False, dataset_exp=1, transforms=transform)
        test_loader = DataLoader(
            dataset = test_dataset,
            batch_size = (cfg["BATCHSIZE_PERGPU"])*gpu_nums,
            shuffle = True,
            num_workers = cfg["WORKERS"],
            pin_memory = False
        )

        start_test_time = time.time()
        loss, ocular_nme, pupil_nme = test_models(model,test_loader,criterion,0)
        logger.info("test  time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%".format(time.time()-start_test_time,loss,ocular_nme*100.0,pupil_nme*100.0))


if __name__ == "__main__":
    args = parse_args()
    main(args)
