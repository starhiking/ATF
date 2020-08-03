"""
    SelfdatasetOnline:
        args:
            csv_file : csv file saved labels
            transforms: pytorch transform
            is_train  : Bool train or test
            input_size: int
        return : (Dataloader)
            imgs:N*3*input_size*input_size
            landmarks:N*(numx2)
"""

import os
import random
import math
import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.append('.')

from scipy.misc.pilutil import imresize
from cv2 import cv2
from hdf5storage import loadmat
import imgaug as ia
from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.4, aug)

def Rotate_pt(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                            M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

class SelfDatasetsOnline(data.Dataset):
    def __init__(self,cfg,is_train=True,transforms=None,dataset_exp=1,image_size=None,data_aug=True):
        global sometimes
        sometimes = lambda aug: iaa.Sometimes(cfg['RANDOM_SEED'], aug)
        if is_train:
            self.csv_file = cfg['Train_csv']
        else :
            self.csv_file = cfg['Test_csv']
        
        if not os.path.exists(self.csv_file):
            print("{} CSV file not exist ".format(self.csv_file))
            raise("csv file not exist")

        self.is_train = is_train and data_aug
        
        print("read {} may cost time".format(self.csv_file))
        self.data_type= cfg['DATA_TYPE']
        self.landmarks_frame = pd.read_csv(self.csv_file)

        if "left" in self.landmarks_frame.keys():
            self.withBox = True
        else:
            self.withBox = False

        self.transforms = transforms
        self.rot_factor= cfg['ROT_FACTOR']
        self.translation_factor = cfg['TRANSLATION_FACTOR']
        self.mirror_idx = cfg['FLIP_INDEX']
        self.scale = cfg["SCALE"]
        self.scale_factor= cfg['SCALE_FACTOR']
        self.csv_add_path = cfg['csv_add_path']
        self.image_size = image_size if image_size else cfg['IMAGE_SIZE']
        self.dataset_exp = dataset_exp
        self.length = len(self.landmarks_frame)

        self.seq = iaa.Sequential([
            iaa.SomeOf((0,3),
                [  
                    sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.5)),
                    sometimes(iaa.contrast.LinearContrast((0.80,1.4),per_channel=0.5)),
                    sometimes(iaa.Add((-10, 10), per_channel=0.5)),
                    # Random occlusion
                    sometimes(iaa.OneOf([
                        iaa.Dropout((0.001, 0.01), per_channel=0.2),
                        iaa.CoarseDropout((0.03, 0.10), size_percent=(0.02, 0.04), per_channel=0.2)
                    ])),
                    # Add noise
                    sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.5)),
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.5)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.5), direction=(0.0, 1.0)
                        ),
                    ])),

                ],
                random_order=True
            ),

            # Random gaussianBlur
            sometimes(iaa.GaussianBlur((0, 2.0))),
            
        ],random_order=True)

    def __len__(self):
        return self.length * self.dataset_exp

    def __getitem__(self,index):
        # index = 109 # uncomplete image index for test
        # index = 65
        # index = 507
        index = index % self.length

        line_data = self.landmarks_frame.iloc[index]
        image_path = os.path.join(self.csv_add_path,line_data.iloc[0])
        if self.data_type == "AFLW":
            landmark = np.asarray(line_data.iloc[5:].values,dtype=np.float32).reshape(-1,2)
        elif self.data_type == "COFW":
            landmark = np.asarray(line_data.iloc[1:].values,dtype=np.float32).reshape(-1,2)
        else :
            landmark = np.asarray(line_data.iloc[4:].values,dtype=np.float32).reshape(-1,2)
        
        if not os.path.exists(image_path):
            print("{} not exist.".format(image_path))
            raise("image path is wrong or destroy")
        # create a In_landmark which set out_landmark is 0 or shape size
        img = cv2.imread(image_path)  # np.asarray(Image.open(image_path).convert('RGB'),dtype=np.float32)
        img = img[: , : , : : -1] # BGR -> RGB

        xy = np.min(landmark,axis=0)#.astype(np.int32)
        zz = np.max(landmark,axis=0)#.astype(np.int32)
        
        wh = zz-xy + 1
        boxsize = (wh*self.scale)#.astype(np.int32)
        center = (xy+wh/2)#.astype(np.int32)
        
        if self.is_train:
            """
                Data augmentation : scale translation rotation mirror occlusion noise brightness sharpen contrast blur
            """

            # Random scale
            scale = self.scale * (random.uniform(1-self.scale_factor,1+self.scale_factor)) if random.random() <=0.5 else self.scale
            boxsize = (wh*scale)#.astype(np.int32)
            
            
            # Random translation somewhere wrong TODO
            adaptive_pxs = int(max(boxsize) * self.translation_factor)
            translation_pxs = np.random.randint(0,adaptive_pxs,(2,2)) if random.random() <= 0.4 else False
            if type(translation_pxs) is np.ndarray:
                translation_pxs[0,1] = 0
                    # [[left,top],
                    # [right,bottom]]
                
                # print(translation_pxs.sum()/4)
                boxsize = boxsize + (translation_pxs.sum(axis = 0)[1],translation_pxs.sum(axis = 0)[0])
                sub_res = np.subtract(translation_pxs[1],translation_pxs[0]) / 2
                center = np.asarray([center[0] + sub_res[1],center[1]+sub_res[0]],dtype=np.float32)
                # center = (center + np.subtract(translation_pxs[1],translation_pxs[0]) / 2)#.astype(np.int32)
            
            
            # Random mirror
            if  random.random() <=0.5 :
                landmark[:,0] = img.shape[1] - landmark[:,0]
                if self.withBox:
                    landmark[2:] = landmark[2:][self.mirror_idx]
                else :
                    landmark = landmark[self.mirror_idx]
                # img = cv2.flip(img,1)
                img = np.fliplr(img)
                center[0] =  img.shape[1] - center[0]
            

            # Random rotation
            rot = random.uniform(-self.rot_factor,self.rot_factor) if random.random() <=0.6 else 0
            if rot:
                M,landmark = Rotate_pt(rot,center,landmark)

                img = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]))
                xy = np.min(landmark,axis=0)#.astype(np.int32)
                zz = np.max(landmark,axis=0)#.astype(np.int32)

                wh = zz-xy + 1
                center = (xy+wh/2)#.astype(np.int32)
                boxsize = (wh*scale)#.astype(np.int32)


        x1,y1 = center - boxsize / 2
        x2,y2 = (x1,y1) + boxsize

        board_flag = False
        board_size = np.zeros(4).astype(np.int32) # upper bottom left right
        
        if x1 < 0:
            board_flag = True
            board_size[2] = math.ceil(-x1) #left
        
        if y1 < 0 :
            board_flag = True
            board_size[0] = math.ceil(-y1)  #upper
        
        if x2 > img.shape[1]-1:
            board_flag = True
            board_size[3] = math.ceil(x2 - (img.shape[1]-1)) #right
        
        if y2 > img.shape[0] -1 :
            board_flag = True
            board_size[1] = math.ceil(y2 - (img.shape[0] -1))  # bottom

        if board_flag:
            new_img = np.zeros((img.shape[0]+np.sum(board_size[0:2]),img.shape[1]+np.sum(board_size[2:4]),3))
            landmark = landmark + (board_size[2],board_size[0])
            x1, y1= np.min(landmark,axis=0)#.astype(np.int32)
            x2, y2= np.max(landmark,axis=0)#.astype(np.int32)
            new_img[board_size[0] : board_size[0]+img.shape[0] , board_size[2] : board_size[2]+img.shape[1]] = img  
            img = new_img

        x1 = math.floor(x1) if x1 > 0 else 0
        y1 = math.floor(y1) if y1 > 0 else 0

        x2 = math.ceil(x2) if x2 < img.shape[1] else img.shape[1]
        y2 = math.ceil(y2) if y2 < img.shape[0] else img.shape[0]

        crop_img = img[y1:y2,x1:x2]
        landmark = landmark - (x1,y1)
        resize_landmark = (landmark / (crop_img.shape[1],crop_img.shape[0])).astype(np.float32)
        
        resize_img = cv2.resize(crop_img,(self.image_size,self.image_size)) # cv2.resize(crop_img,(self.image_size,self.image_size))
        
        if self.is_train:
            ## not change shape and landmark
            
            seq_det = self.seq.to_deterministic()
            resize_img = seq_det.augment_images([resize_img.astype(np.float32)])[0]       

        resize_img = np.maximum(resize_img,0)
        resize_img = np.minimum(resize_img,255).astype(np.uint8)

        
        if self.transforms:
            resize_img = self.transforms(resize_img)

        if self.withBox:
            resize_landmark = resize_landmark[2:]

        return resize_img,resize_landmark.reshape(-1)

if __name__ == "__main__":
    import json
    with open("experiments/AFLW.json") as f:
        cfg = json.load(f)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    wflwDataset = SelfDatasetsOnline(cfg,is_train=True,transforms=None,dataset_exp=1,image_size=112)
    dataloader = data.DataLoader(wflwDataset,batch_size=128,shuffle=False,num_workers=8,drop_last=False)
    losses = []
    for i,(imgs,landmarks) in enumerate(dataloader):
        imgs = imgs.numpy().astype(np.int32)
        landmarks = landmarks.numpy().astype(np.float32).reshape(imgs.shape[0], -1,2)
        img_landmarks = imgs.shape[1] * landmarks

        # for img,landmark in zip(imgs,img_landmarks):
        #     plt.imshow(img)
        #     # plt.scatter(landmark[:2,0],landmark[:2,1],(255,255,255))
        #     # plt.scatter(landmark[2:,0],landmark[2:,1],1)
        #     plt.scatter(landmark[:,0],landmark[:,1],1)
        #     plt.show()

        print(imgs.shape)
        print(landmarks.shape)