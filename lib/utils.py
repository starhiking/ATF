import logging
import os
import sys
import time
import torch.optim as optim
import numpy as np
import torch
import json
sys.path.append('..')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_nme(preds,pts):
    
    # shape: N * (Number*2)
    N = preds.size(0)
    L = preds.size(1)//2

    preds = preds.numpy().reshape(N,-1,2) # predict landmark
    pts = pts.numpy().reshape(N,-1,2) # real landmark
    
    interocular_nme = np.zeros(N)
    inter_pupil_nme = np.zeros(N)
    
    for i in range(N):
        lmk_pred , lmk_pts = preds[i,],pts[i,]
        if L == 19 : # aflw
            interocular = 1 # np.linalg.norm(lmk_pts[6, ] - lmk_pts[11, ])
            inter_pupil = np.linalg.norm(lmk_pts[7, ] - lmk_pts[10, ])
        elif L == 29: # cofw
            interocular = np.linalg.norm(lmk_pts[8, ] - lmk_pts[9, ])
            inter_pupil = np.linalg.norm(lmk_pts[10, ] - lmk_pts[11, ])
        elif L == 68:
            interocular = np.linalg.norm(lmk_pts[36, ] - lmk_pts[45, ])
            inter_pupil = np.linalg.norm(np.average(lmk_pts[36:42], axis=0) - np.average(lmk_pts[42:48], axis=0))
        elif L == 98:
            interocular = np.linalg.norm(lmk_pts[60, ] - lmk_pts[72, ])
            inter_pupil = np.linalg.norm(lmk_pts[96, ] - lmk_pts[97, ])
        else:
            print(L,"is not support")
            raise ValueError('Number of landmarks is wrong')
        
        interocular_nme[i] = np.sum(np.linalg.norm(lmk_pred - lmk_pts, axis=1)) / (interocular * L)
        inter_pupil_nme[i] = np.sum(np.linalg.norm(lmk_pred - lmk_pts, axis=1)) / (inter_pupil * L)

    return interocular_nme,inter_pupil_nme


def self_loss(x, y):
    i = (x[:, 6, :] - x[:, 11, :]).norm(dim = 1)
    j = (x - y).norm(dim = 2)
    k = j / i.view(-1, 1)
    return k.sum()


def lr_repr(optim):
    _lr_repr_ = 'lr:'
    for pg in optim.param_groups:
        _lr_repr_ += ' {} '.format(pg['lr'])
    return _lr_repr_

def create_logger(logfile):

    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    log_file = "{}_{}.log".format(logfile , time.strftime('_%m%d_%H%M%S'))

    final_log_file = os.path.join('logs',log_file)
    if os.path.exists(final_log_file):
        print("Current log file is exist")
        raise("Log file alread exist")

    logging.basicConfig(
        format=
        '[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(final_log_file, mode='w'),
            logging.StreamHandler()
        ])                        
    logger = logging.getLogger()
    
    return logger

def train(model,train_loader,criterion,optimizer,epoch):
    losses = AverageMeter()
    losses.reset()
    model.train()
    print_feq = 50
    
    interocular_sum_nme = 0
    inter_pupil_sum_nme = 0

    for i,(imgs,landmarks) in enumerate(train_loader):
        preds = model(imgs).cuda()
        landmarks = landmarks.cuda(non_blocking=True)
        loss = criterion(preds,landmarks)

        interocular_batch_nme , inter_pupil_batch_nme = compute_nme(preds.cpu().detach(),landmarks.cpu().detach())
        interocular_sum_nme += np.sum(interocular_batch_nme)
        inter_pupil_sum_nme += np.sum(inter_pupil_batch_nme)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(),landmarks.size(0))
        if i % print_feq == 0 and i != 0:
            print("[{:<4}/{:<5}] Loss : {:.8f} ocular_nme : {:.4f}% pupil_nme : {:.4f}%".format(epoch,i,losses.val,np.mean(interocular_batch_nme)*100.0,np.mean(inter_pupil_batch_nme)*100.0))

    interocular_nme = interocular_sum_nme / len(train_loader.dataset)# train_loader.sampler.num_samples
    inter_pupil_nme = inter_pupil_sum_nme / len(train_loader.dataset)# train_loader.sampler.num_samples
    return losses.avg,interocular_nme,inter_pupil_nme

def test(model,test_loader,criterion):
    losses = AverageMeter()
    losses.reset()
    model.eval()

    interocular_sum_nme = 0
    inter_pupil_sum_nme = 0

    with torch.no_grad():
        for i,(imgs,landmarks) in enumerate(test_loader):
            preds = model(imgs).cuda()
            landmarks = landmarks.cuda(non_blocking=True)
            loss = criterion(preds,landmarks)

            interocular_batch_nme , inter_pupil_batch_nme = compute_nme(preds.cpu(),landmarks.cpu())
            interocular_sum_nme += np.sum(interocular_batch_nme)
            inter_pupil_sum_nme += np.sum(inter_pupil_batch_nme)
            
            losses.update(loss.item(),landmarks.size(0))

    interocular_nme = interocular_sum_nme / len(test_loader.dataset)# test_loader.sampler.num_samples
    inter_pupil_nme = inter_pupil_sum_nme / len(test_loader.dataset)# test_loader.sampler.num_samples

    return losses.avg ,interocular_nme , inter_pupil_nme

def save_checkpoint(model_state_dict,best_nme,optimizer_state_dict,output_dir):
    
    epoch = best_nme['epoch']
    file_path = os.path.join(output_dir,"{}_checkpoint.pth".format(str(epoch)))
    best_path = os.path.join(output_dir,"best.pth")
    torch.save({
        "state_dict":model_state_dict,
        "best_nme":best_nme,
        "optimizer":optimizer_state_dict,
        "epoch":epoch
    },file_path)
    if os.path.islink(best_path):
        os.remove(best_path)
    
    # symlink is create a relative path file : a is exist file and relative path,b is link and absolute path
    os.symlink(os.path.join("./","{}_checkpoint.pth".format(str(epoch))),best_path)

def Init(args):
    """Init and Update cfg"""
    
    with open(os.path.join('experiments','{}.json'.format(args.data_type))) as f:
        cfg = json.load(f)
        cfg["LR"] = args.lr
        cfg["GPUS"] = args.gpus
        cfg["BATCHSIZE_PERGPU"] = args.batch_size
        cfg["IMAGE_SIZE"] = args.image_size
        cfg['WORKERS'] = args.workers

    logger = create_logger("{}_{}".format(args.data_type,args.model_type))
    logger.info(cfg)
    logger.info(args)
    
    best_nme = {'epoch':-1,'loss':100,'nme':100}
    last_epoch = cfg["BEGIN_EPOCH"]
    end_epoch = cfg["END_EPOCH"]
    model_save_dir = os.path.join('checkpoint',args.data_type,args.model_dir)
    check_mkdir(model_save_dir)

    return cfg,logger,best_nme,model_save_dir,last_epoch,end_epoch

def check_mkdir(str_path):
    paths = str_path.split('/')
    # temp_folder = paths[0]
    temp_folder = ""
    for i in range (len(paths)):
        temp_folder = os.path.join(temp_folder,paths[i])
        if not os.path.exists(temp_folder):
            print("INFO: {} not exist , created.".format(temp_folder))
            os.mkdir(temp_folder)
    
    assert os.path.exists(str_path) , "{} not created success.".format(str_path)
    
