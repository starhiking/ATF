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
# from models.multi_flan3 import MULTI_FLAN3

test_files = {"WFLW": None, "300W": None, "AFLW": None, "COFW": None}

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

same_index = {
    '98': [33, 46, 60, 64, 68, 72, 54, 76, 82, 16],
    '68': [17, 26, 36, 39, 42, 45, 30, 48, 54, 8],
    '29': [0, 1, 8, 10, 11, 9, 20, 22, 23, 28],
    '19': [0, 5, 6, 8, 9, 11, 13, 15, 17, 18]
}


def parse_args():
    parser = argparse.ArgumentParser(description='Training infos')

    parser.add_argument('--model_type', type=str, default="MULTI_FLAN2")
    parser.add_argument('--resume_checkpoints', type=str, default="")
    parser.add_argument('--main_data', type=str, default="COFW")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--show_others', default=False, action='store_true')
    parser.add_argument('--aux_datas', type=str, default="WFLW")

    parser.add_argument('--aux_ratios', type=str, default="3,1")
    parser.add_argument('--ratios_decay', type=float, default=1.0)
    parser.add_argument('--loss_alpha', type=float, default=0.5)
    parser.add_argument('--dataset_exp', type=int, default=10)
    parser.add_argument('--model_dir', type=str, default="Mix")
    parser.add_argument('--lr_reduce_patience', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss_type', type=str, default="L1")
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--mix_loss', default=False, action='store_true')
    parser.add_argument('--auxdata_aug', default=False, action='store_true')

    return parser.parse_args()


def Init_mix(args):
    args.aux_datas = args.aux_datas.split(',')
    args.aux_ratios = np.asarray(args.aux_ratios.split(','), dtype=np.float32)

    with open(os.path.join('experiments',
                           '{}.json'.format(args.main_data))) as f:
        main_cfg = json.load(f)

    aug_cfgs = {}

    for aux_data in args.aux_datas:
        with open(os.path.join('experiments',
                               '{}.json'.format(aux_data))) as f:
            aux_cfg = json.load(f)
            aug_cfgs[str(aux_cfg["NUM_JOINTS"])] = aux_cfg

    return args, main_cfg, aug_cfgs


def test_models(models, test_loader, criterion, model_index):
    losses = AverageMeter()
    losses.reset()
    models.eval()

    interocular_sum_nme = 0
    inter_pupil_sum_nme = 0

    with torch.no_grad():
        for i, (imgs, landmarks) in enumerate(test_loader):
            imgs = imgs.cuda(non_blocking=True)
            preds_array = models(imgs)
            preds = preds_array[model_index].cuda()
            landmarks = landmarks.cuda(non_blocking=True)
            loss = criterion(preds, landmarks)

            interocular_batch_nme, inter_pupil_batch_nme = compute_nme(
                preds.cpu(), landmarks.cpu())
            interocular_sum_nme += np.sum(interocular_batch_nme)
            inter_pupil_sum_nme += np.sum(inter_pupil_batch_nme)

            losses.update(loss.item(), landmarks.size(0))

    interocular_nme = interocular_sum_nme / len(
        test_loader.dataset)  # test_loader.sampler.num_samples
    inter_pupil_nme = inter_pupil_sum_nme / len(
        test_loader.dataset)  # test_loader.sampler.num_samples

    return losses.avg, interocular_nme, inter_pupil_nme


def main(args):

    args, main_cfg, cfgs = Init_mix(args)
    logger = create_logger("Test_{}_{}".format(args.main_data,
                                               args.model_type))
    logger.info(args)

    os.environ[
        'CUDA_VISIBLE_DEVICES'] = args.gpus  #",".join(str(gpu) for gpu in cfg["GPUS"])
    gpu_nums = len(args.gpus.split(','))  # len(cfg["GPUS"])

    # define models
    if args.model_type in ["multi_flan", "MULTI_FLAN"]:
        landmark_nums = [main_cfg["NUM_JOINTS"] * 2]
        for key in cfgs.keys():
            landmark_nums.append(cfgs[key]["NUM_JOINTS"] * 2)

        models = MULTI_FLAN(landmark_nums)  #models[0] is main_
    elif args.model_type in ["multi_flan2", "MULTI_FLAN2"]:
        landmark_nums = [main_cfg["NUM_JOINTS"] * 2]
        for key in cfgs.keys():
            landmark_nums.append(cfgs[key]["NUM_JOINTS"] * 2)

        models = MULTI_FLAN2(landmark_nums)  #models[0] is main_
    elif args.model_type in ["multi_flan3", "MULTI_FLAN3"]:
        landmark_nums = [main_cfg["NUM_JOINTS"] * 2]
        for key in cfgs.keys():
            landmark_nums.append(cfgs[key]["NUM_JOINTS"] * 2)

        models = MULTI_FLAN3(landmark_nums)  #models[0] is main_
    else:
        print("ERROR :Not support {} network".format(args.model_type))
        exit()

    if args.loss_type in ["L2", "l2", "MSE", "mse"]:
        criterion = nn.MSELoss(size_average=True).cuda()
    elif args.loss_type in ["L1", "l1"]:
        criterion = nn.L1Loss(size_average=True).cuda()
    elif args.loss_type in ["Smooth_L1", "SMOOTH_L1"]:
        criterion = nn.SmoothL1Loss(size_average=True).cuda()
    elif args.loss_type in ["WING", "wing"]:
        criterion = WingLoss().cuda()
    elif args.loss_type in ["AWING", "ADAPTIVE_WING"]:
        criterion = AdaptiveWingLoss().cuda()
    else:
        raise ("Not support {} loss now.".format(args.loss_type))

    models = nn.DataParallel(models, range(gpu_nums)).cuda()
    devices = torch.device("cuda:0")
    models.to(devices)

    best_loss = 10
    if os.path.exists(args.resume_checkpoints):
        checkpoint = torch.load(args.resume_checkpoints)
        models.load_state_dict(checkpoint['state_dict'])
        best_nme = checkpoint['best_nme']
        best_loss = best_nme['loss']
        logger.info("restore epoch : {} best nme : {} ".format(
            checkpoint['epoch'], best_nme))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for file_path in test_files[main_cfg["DATA_TYPE"]]:
        main_cfg["Test_csv"] = file_path
        logger.info("Test {}".format(file_path))

        main_test_dataset = SelfDatasetsOnline(main_cfg,
                                               is_train=False,
                                               transforms=transform,
                                               image_size=args.image_size)

        main_test_loader = DataLoader(dataset=main_test_dataset,
                                      batch_size=args.batch_size * gpu_nums,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True)

        start_test_time = time.time()
        loss, ocular_nme, pupil_nme = test_models(models, main_test_loader,
                                                  criterion, 0)
        logger.info(
            "test time :{:<6.2f}s loss :{:.8f} ocular_nme:{:.5f}% pupil_nme:{:.5f}%"
            .format(time.time() - start_test_time, loss, ocular_nme * 100.0,
                    pupil_nme * 100.0))


if __name__ == "__main__":
    args = parse_args()
    main(args)