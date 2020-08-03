"""
    Use pfld backbone.but modify last S1 S2 connecting a avgpooling then concat with S3 to decrease params size.
    And set channel as half
"""
import torch
import torch.nn as nn
import time
import collections
import os

BN_MOMENTUM = 0.1

def conv_bn(inp, oup, kernel, stride, padding=1,depth=False):
    """
        inp: input channel
        oup:output channel
    """
    if depth:
        assert oup == inp ,"depthwise conv input channel {} not equal output channel {}.".format(inp,oup)
        group = inp
    else :
        group = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding,groups=group, bias=False),
        nn.BatchNorm2d(oup,momentum=BN_MOMENTUM),
        nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):
    def __init__(self,inp,oup,stride,use_res_connect,expand_ration=6):
        super(InvertedResidual,self).__init__()
        self.stride = stride
        assert stride in [1,2],"stride is {} , exit.".format(stride)

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp,inp*expand_ration,1,1,0,bias=False),
            nn.BatchNorm2d(inp*expand_ration,momentum=BN_MOMENTUM),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp*expand_ration,inp*expand_ration,3,stride,1,groups=inp*expand_ration,bias=False),
            nn.BatchNorm2d(inp*expand_ration,momentum=BN_MOMENTUM),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp*expand_ration,oup,1,1,0,bias=False),
            nn.BatchNorm2d(oup,momentum=BN_MOMENTUM)
        )
    
    def forward(self,x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PFLD_HALF(nn.Module):
    def __init__(self,nums,half=2):
        super(PFLD_HALF,self).__init__()

        self.conv1 = conv_bn(3,64//half,3,2,1) #conv_bn has bn and relu6
        self.conv2 = conv_bn(64//half,64//half,3,1,1,True) # Question TODO dwconv?
        
        self.conv3_1 = InvertedResidual(64//half,64//half,2,False,2)
        self.block3_2 = InvertedResidual(64//half,64//half,1,True,2)
        self.block3_3 = InvertedResidual(64//half,64//half,1,True,2)
        self.block3_4 = InvertedResidual(64//half,64//half,1,True,2)
        self.block3_5 = InvertedResidual(64//half,64//half,1,True,2)

        self.conv4_1 = InvertedResidual(64//half,128//half,2,False,2)
        # Question TODO Conv5_1 False->True
        self.conv5_1 = InvertedResidual(128//half,128//half,1,False,4)
        self.block5_2 = InvertedResidual(128//half,128//half,1,True,4)
        self.block5_3 = InvertedResidual(128//half,128//half,1,True,4)
        self.block5_4 = InvertedResidual(128//half,128//half,1,True,4)
        self.block5_5 = InvertedResidual(128//half,128//half,1,True,4)
        self.block5_6 = InvertedResidual(128//half,128//half,1,True,4)
        
        self.conv6_1 = InvertedResidual(128//half,16//half,1,False,2)

        self.conv7 = conv_bn(16//half,32//half,3,2,1)
        self.conv8 = conv_bn(32//half,128//half,7,1,0)
        
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176//half,nums)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)   
        
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)

        x = self.conv4_1(x)
        
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        
        S1 = self.conv6_1(x)
        S2 = self.conv7(S1)
        S3 = self.conv8(S2)
        
        S1 = self.avg_pool1(S1)
        S2 = self.avg_pool2(S2)

        # need transform 2 dims : NxC
        x = torch.cat([S1.view(S1.size(0),-1),S2.view(S2.size(0),-1),S3.view(S3.size(0),-1)],1)
        x = self.fc(x)

        return x

    def init_weights(self,pretrained=''):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if os.path.isfile(pretrained) or os.path.islink(pretrained):
            
            pretrained_dict = torch.load(pretrained)
            if not isinstance(pretrained_dict,collections.OrderedDict):    
                # suppose state_dict in pretrained_dict 
                if isinstance(pretrained_dict['state_dict'],collections.OrderedDict):
                    pretrained_dict = pretrained_dict['state_dict']
                else :
                    raise("cannot find the state_dict in {}".format(pretrained))

            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


if __name__ == "__main__":
    net_input = torch.randn(1,3,112,112)
    pfld = PFLD_HALF(98*2,2).eval()
    start = time.time()
    for i in range(100):
        landmarks = pfld(net_input)
    print(landmarks.shape)
    print("per image process time : {:.4f}".format((time.time() - start)/100))
    
    from thop import profile,clever_format
    flops, params = profile(pfld,inputs=(net_input, ))    
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: {} Params: {}".format(flops,params))