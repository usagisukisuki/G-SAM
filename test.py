#coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

import os
import argparse
from sklearn.metrics import confusion_matrix
import random
from tqdm import tqdm 

from models.model_dict import get_model
from loader import *
import utils as ut



####### IoU GPU for trans10k ###############
def IoU_10k(y_pred, y_true, num_classes):
    # y_pred.requires_grad_(True)
    smooth = 1e-7 # ゼロ除算回避

    y_pred = F.softmax(y_pred, dim=1) #[4,5,256,256]0から1
    y_true = F.one_hot(y_true, num_classes) #one-hot化
    y_true = torch.permute(y_true, (0, 3, 1, 2)).float() #[4,5,256,256]

    ious = []

    for i in range(num_classes):
        inter = torch.sum(y_pred[:, i] * y_true[:, i])
        cardi = torch.sum(y_pred[:, i] * y_pred[:, i]) + torch.sum(y_true[:, i] * y_true[:, i])
        union = cardi - inter
        iou = (inter + smooth) / (union + smooth)
        ious.append(iou)

    ious = torch.stack(ious)
    miou = torch.mean(ious)

    return miou

####### IoU ##########
def fast_hist(label_true, label_pred, classes):
    mask = (label_true >= 0) & (label_true < classes)
    hist = np.bincount(classes * label_true[mask].astype(int) + label_pred[mask], minlength=classes ** 2,).reshape(classes, classes)
    return hist


def IoU(output, target, label):
    output = torch.stack(output)
    target = torch.stack(target)

    if label==2:
        output = torch.where(torch.sigmoid(output)>=0.5, 1, 0)
        seg = np.array(output[:,0])
        target = np.array(target[:,0])
    else:
        output = F.softmax(output, dim=1)
        _, output_idx = output.max(dim=1)
        seg = np.array(output_idx)
        target = np.array(target)
    

    # onehot
    confusion_matrix = np.zeros((label, label))

    for lt, lp in zip(target, seg):
        confusion_matrix += fast_hist(lt.flatten(), lp.flatten(), label)

    diag = np.diag(confusion_matrix)
    iou_den = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - (diag+1e-7))
    iou = (diag+1e-7) / np.array(iou_den, dtype=np.float32)
    return iou


####### Validation #######
def test():
    model_path = "{}/model/model_bestiou.pth".format(args.out)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predict = []
    answer = []
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            
            if args.num_classes==2:
                targets = targets.unsqueeze(1)
                targets = targets.float()
            else:
                targets = targets.long() 
                
            ##### model input #####
            output = model(inputs, None, None)
            output = output['masks']
            
            
            ##### IoU ######
            iou = IoU_10k(output, targets, args.num_classes)
            # output = output.cpu()
            # targets = targets.cpu()
            # for j in range(output.shape[0]):
            #     predict.append(output[j])
            #     answer.append(targets[j])
                
            # iou1 = IoU(predict, answer, label=args.num_classes)
            total += iou.item()
            num_batches += 1
            # print(iou)

        average_iou = total / num_batches

    return average_iou



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment Anything Model')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--num_epochs',  type=int, default=200)
    parser.add_argument('--dataset',  type=str, default='ISBI2012', help='ISBI2012 or ssTEM')
    parser.add_argument('--datapath',  type=str, default='./Dataset/')
    parser.add_argument('--num_classes',  type=int, default=2)
    parser.add_argument('--multimask_output', type=bool, default=False)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--modelname', default='SAM', type=str, help='SAM, MobileSAM, SAM_LoRA, SAM_ConvLoRA, SAM_AdaptFormer, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='./models/Pretrained_model/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--multi', action='store_true')
    args = parser.parse_args()


    ##### IoU #####
    PATH = "{}/test_iou.txt".format(args.out)
    with open(PATH, mode = 'w') as f:
        pass
    
    
    ##### GPU #####
    gpu_flag = args.gpu
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
        
        
    ########## SAM model ##########
    if 'Mobile' in args.modelname:
        args.sam_ckpt = 'models/Pretrained_model/mobile_sam.pt'
        
    model = get_model(args.modelname, args=args).to(device)

    if args.multi:
        model = nn.DataParallel(model)
        
        
    ######### Dataset ##########
    test_loader = data_loader_test(args)
    
    
    mm = test()

    with open(PATH, mode = 'a') as f:
        f.write("TestmIoU==={:.2f}\n"
                .format(np.mean(mm)*100.))
    

    print("  mIoU   : %.2f" % (np.mean(mm)*100.))
    



