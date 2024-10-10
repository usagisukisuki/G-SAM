import torch
import torchvision
from .Mydataset import ISBICellDataloader, Drosophila_Dataloader, CamvidLoader, MassachusettsDataloader, PolypeDataset, SynapseDataset, Cityscapse_Loader, TransSegmentation
import loader.utils as ut
import os

####### dataset loader ########
def data_loader_train(args):
    if args.dataset=='ISBI2012':
        resize_img = [256,256]
        crop_img = [128,128]
    elif args.dataset=='ssTEM':
        resize_img = [512,512]
        crop_img = [256,256]
    elif args.dataset=='M-Road':
        resize_img = [1024,1024]
        crop_img = [512,512]
    elif args.dataset=='M-Building':
        resize_img = [1024,1024]
        crop_img = [512,512]
    elif args.dataset=='Kvasir':
        resize_img = [224,224]
        crop_img = [224,224]
    elif args.dataset=='Synapse':
        resize_img = [224,224]
        crop_img = [224,224]
    elif args.dataset=='CamVid':
        resize_img = [360,480]
        crop_img = [256,256]
    # elif args.dataset=='ADE20k':
    #     resize_img = [1024,1024]
    #     crop_img = [512,512]
    elif args.dataset=='Cityscapes':
        resize_img = [512,1024]
        crop_img = [512,512]
    elif args.dataset=='Trans10k':
        resize_img = [520,520]
        crop_img = [256,256]
    

    if args.modelname=='SAMUS':
        train_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                        ut.ExtRandomRotation(degrees=90),
                                        ut.ExtRandomHorizontalFlip(),
                                        #  ut.ExtRandomRotation(degrees=5),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])

        val_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])


    elif args.modelname=='GSAM':
        train_transform = ut.ExtCompose([ut.ExtResize((resize_img[0], resize_img[1])),
                                        ut.ExtRandomCrop((crop_img[0], crop_img[1])),
                                        ut.ExtRandomRotation(degrees=90),
                                        ut.ExtRandomHorizontalFlip(),
                                        #  ut.ExtRandomRotation(degrees=5),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])

        val_transform = ut.ExtCompose([ut.ExtResize((resize_img[0], resize_img[1])),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])


    else:
        train_transform = ut.ExtCompose([ut.ExtResize((1024, 1024)),
                                        ut.ExtRandomRotation(degrees=90),
                                        ut.ExtRandomHorizontalFlip(),
                                        #  ut.ExtRandomRotation(degrees=5),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])

        val_transform = ut.ExtCompose([ut.ExtResize((1024, 1024)),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])


    if args.dataset=='ISBI2012':
        data_train = ISBICellDataloader(root = args.datapath+args.dataset , dataset_type='train', transform=train_transform)
        data_val = ISBICellDataloader(root = args.datapath+args.dataset, dataset_type='val', transform=val_transform)
        
    elif args.dataset=='ssTEM':
        data_train = Drosophila_Dataloader(rootdir=args.datapath+args.dataset, val_area=1, split='train', iteration_number=12 * args.batchsize, transform=train_transform)
        data_val = Drosophila_Dataloader(rootdir=args.datapath+args.dataset, val_area=1, split='val', transform=val_transform)

    elif args.dataset=='CamVid':
        data_train = CamvidLoader(path=args.datapath+args.dataset, dataset_type='train', transform=train_transform)
        data_val = CamvidLoader(path=args.datapath+args.dataset, dataset_type='val', transform=val_transform)
        
    elif args.dataset=='M-Road':
        data_train = MassachusettsDataloader(root=args.datapath+args.dataset, dataset_type='train', transform=train_transform)          
        data_val = MassachusettsDataloader(root=args.datapath+args.dataset, dataset_type='val', transform=val_transform)
        
    elif args.dataset=='M-Building':
        data_train = MassachusettsDataloader(root=args.datapath+args.dataset, dataset_type='train', transform=train_transform)          
        data_val = MassachusettsDataloader(root=args.datapath+args.dataset, dataset_type='val', transform=val_transform)
        
    elif args.dataset=='Kvasir':
        data_train = PolypeDataset(root=args.datapath+args.dataset, dataset_type='train', transform=train_transform)
        data_val = PolypeDataset(root=args.datapath+args.dataset, dataset_type='val', transform=val_transform)
        
    elif args.dataset=='Synapse':
        data_train = SynapseDataset(root=args.datapath+args.dataset, dataset_type='train', transform=train_transform)
        data_val = SynapseDataset(root=args.datapath+args.dataset, dataset_type='val', transform=val_transform)

    # elif args.dataset=='ADE20k':
    #     data_train = ADE20KSegmentation(split='train', mode='train', transform=train_transform)
    #     data_val = ADE20KSegmentation(split='val', mode='val', transform=val_transform)

    elif args.dataset=='Cityscapes':
        data_train = Cityscapse_Loader(root_dir=args.datapath+args.dataset, dataset_type='train', transform=train_transform)
        data_val = Cityscapse_Loader(root_dir=args.datapath+args.dataset, dataset_type='val', transform=val_transform)

    elif args.dataset=='Trans10k':
        data_train = TransSegmentation(root=args.datapath+args.dataset, split='train', mode='train', transform=train_transform)
        data_val = TransSegmentation(root=args.datapath+args.dataset, split='val', mode='val', transform=val_transform)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=os.cpu_count())
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=4, shuffle=False, drop_last=True, num_workers=os.cpu_count()) 

    return train_loader, val_loader
    
    
####### dataset loader ########
def data_loader_test(args):
    if args.dataset=='ISBI2012':
        resize_img = [256,256]
    elif args.dataset=='ssTEM':
        resize_img = [512,512]
    elif args.dataset=='M-Road':
        resize_img = [1024,1024]
    elif args.dataset=='M-Building':
        resize_img = [1024,1024]
    elif args.dataset=='Kvasir':
        resize_img = [224,224]
    elif args.dataset=='Synapse':
        resize_img = [224,224]
    elif args.dataset=='CamVid':
        resize_img = [360,480]
    # elif args.dataset=='ADE20k':
    #     resize_img = [512,512]
    elif args.dataset=='Cityscapes':
        resize_img = [512,1024]
    elif args.dataset=='Trans10k':
        resize_img = [520,520]

        
    if args.modelname=='SAMUS':
        test_transform = ut.ExtCompose([ut.ExtResize((256, 256)),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])


    elif args.modelname=='GSAM':
        test_transform = ut.ExtCompose([ut.ExtResize((resize_img[0], resize_img[1])),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])


    else:
        test_transform = ut.ExtCompose([ut.ExtResize((1024, 1024)),
                                        ut.ExtToTensor(),
                                        ut.ExtNormalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
                                        ])


    if args.dataset=='ISBI2012':
        data_test = ISBICellDataloader(root = args.datapath+args.dataset, dataset_type='test', transform=test_transform)
        
    elif args.dataset=='ssTEM':
        data_test = Drosophila_Dataloader(rootdir=args.datapath+args.dataset, val_area=1, split='test', transform=test_transform)

    elif args.dataset=='CamVid':
        data_test = CamvidLoader(path=args.datapath+args.dataset, dataset_type='test', transform=test_transform)
        
    elif args.dataset=='M-Road':     
        data_test = MassachusettsDataloader(root=args.datapath+args.dataset, dataset_type='test', transform=test_transform)
        
    elif args.dataset=='M-Building':       
        data_test = MassachusettsDataloader(root=args.datapath+args.dataset, dataset_type='test', transform=test_transform)
        
    elif args.dataset=='Kvasir':
        data_test = PolypeDataset(root=args.datapath+args.dataset, dataset_type='test', transform=test_transform)
        
    elif args.dataset=='Synapse':
        data_test = SynapseDataset(root=args.datapath+args.dataset, dataset_type='test', transform=test_transform)

    # elif args.dataset=='ADE20k':
    #     data_test = ADE20KSegmentation(split='val', mode='val', transform=test_transform)

    elif args.dataset=='Cityscapes':
        data_test = Cityscapse_Loader(root_dir=args.datapath+args.dataset, dataset_type='val', transform=test_transform)

    elif args.dataset=='Trans10k':
        data_test = TransSegmentation(root=args.datapath+args.dataset, split='test', mode='test', transform=test_transform)


    test_loader = torch.utils.data.DataLoader(data_test, batch_size=4, shuffle=False, drop_last=True, num_workers=os.cpu_count())


    return test_loader
