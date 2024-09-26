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



####### Training #######
def train(epoch):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0
    c = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
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
        # print(targets.shape)
        # print(output.shape)
        
        ##### loss #####
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        
    return sum_loss/(batch_idx+1)


####### Validation #######
def test(epoch):
    sum_loss = 0
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, leave=False)):
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
            
            ##### loss #####
            loss = criterion(output, targets)
            
            sum_loss += loss.item()
            
            
            ##### IoU ######
            output = output.cpu()
            targets = targets.cpu()
            for j in range(output.shape[0]):
                predict.append(output[j])
                answer.append(targets[j])
                
        iou = IoU(predict, answer, label=args.num_classes)


    return sum_loss/(batch_idx+1), iou
    
    
    

if __name__ == '__main__':
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
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006 元は0.0005!!!
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--multi', action='store_true')
    args = parser.parse_args()


    print('# GPU    : {}'.format(args.gpu))
    print('# Dataset: {}'.format(args.dataset))
    print('# Batch  : {}'.format(args.batchsize))
    print('# epoch  : {}'.format(args.num_epochs))
    print('# Model  : {}'.format(args.modelname))
    print('# Class  : {}'.format(args.num_classes))
    print('')


    ##### GPU #####
    gpu_flag = args.gpu
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')



    ##### seed #####
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    
    
    ##### Save dir #####
    if not os.path.exists("{}".format(args.out)):
      	os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "model")):
      	os.mkdir(os.path.join("{}".format(args.out), "model"))

    PATH_1 = "{}/trainloss.txt".format(args.out)
    PATH_2 = "{}/testloss.txt".format(args.out)
    PATH_3 = "{}/IoU.txt".format(args.out)
    
    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass



    ########## SAM model ##########
    if 'Mobile' in args.modelname:
        args.sam_ckpt = 'models/Pretrained_model/mobile_sam.pt'
        
    model = get_model(args.modelname, args=args).to(device)

    if args.multi:
        model = nn.DataParallel(model)


    ########## Dataset ##########
    train_loader, val_loader = data_loader_train(args)


    ########## Loss function ##########
    if args.multimask_output == True:
        if args.dataset=='CamVid':
            criterion = nn.CrossEntropyLoss(ignore_index=11)
        if args.dataset=='ADE20k':
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        if args.dataset=='Cityscapes':
            criterion = nn.CrossEntropyLoss(ignore_index=19)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()


    ########## Optimizer ##########
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0.0)



    ########## Traingin & Validation ##########
    sample = 0
    sample_loss = 10000000

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    for epoch in range(args.num_epochs):
        loss_train = train(epoch)
        loss_test, mm = test(epoch)
        scheduler.step() 


        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_train))
        with open(PATH_2, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, loss_test))
        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\t%f\t%f\n" % (epoch+1, mm[0], mm[1], np.mean(mm)))


        ######## Save model ######## 
        if np.mean(mm) >= sample:
            sample = np.mean(mm)
            PATH_best ="{}/model/model_bestiou.pth".format(args.out)
            torch.save(model.state_dict(), PATH_best)

        if loss_train < sample_loss:
           sample_loss = loss_train
           PATH_best ="{}/model/model_bestloss.pth".format(args.out)
           torch.save(model.state_dict(), PATH_best)

        PATH_best ="{}/model/model_train.pth".format(args.out)
        torch.save(model.state_dict(), PATH_best)
        
        print("Epoch{:3d}/{:3d}  TrainLoss:{:.4f}  mIoU:{:.4f}".format(epoch+1, args.num_epochs, loss_train, np.mean(mm)))

