import numpy as np
import glob
import torch
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms
import os
import random
from PIL import Image
import glob
from natsort import natsorted
import cv2
import loader.utils as ut
import logging
from loader.segbase import SegmentationDataset_10k

################################ ISBI2012 #########################################
class ISBICellDataloader(data.Dataset):
    def __init__(self, root=None, dataset_type='train',  cross=0, K=3, transform=None):
        self.root = root
        self.dataset_type = dataset_type
        self.transform = transform

        self.data = sorted(os.listdir(self.root + "/Image"))
        self.label = sorted(os.listdir(self.root + "/Label"))
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        
        idx = np.arange(0, len(self.data))
        idx_train = idx[np.fmod(idx, K) != cross].astype(np.int32) 
        idx_test = idx[np.fmod(idx, K) == cross].astype(np.int32) 

        if self.dataset_type=='train':
            self.datas = self.data[idx_train]
            self.labels = self.label[idx_train]
        
        else:
            self.datas = self.data[idx_test]
            self.labels = self.label[idx_test]
        
        self.data_path = []
        self.label_path = []
        for i in range(len(self.datas)):
            self.data_num = sorted(os.listdir(self.root + "/Image/{}".format(self.datas[i])))
            self.label_num = sorted(os.listdir(self.root + "/Label/{}".format(self.labels[i])))
            for j in range(len(self.data_num)):
                self.data_path.append("{}/".format(self.datas[i]) + self.data_num[j])
                self.label_path.append("{}/".format(self.labels[i]) + self.label_num[j])
            
        

    def __getitem__(self, index):
        # data
        image_name = self.root + "/Image/" + self.data_path[index]
        label_name = self.root + "/Label/" + self.label_path[index]
        image = Image.open(image_name).convert("RGB")
        label = Image.open(label_name).convert("RGB")
        label = np.array(label)
        label = np.where(label[:,:,0]>=150, 0, 1)
        label = Image.fromarray(np.uint8(label))

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.data_path)



################################ ISBI2012 #########################################
class Drosophila_Dataloader(data.Dataset):
    def __init__(self, rootdir="Dataset", val_area=1, split='train', iteration_number=None, transform=None):
        self.split = split
        self.training = True if split == 'train' else False
        self.transform = transform
        filelist_train = []
        filelist_val = []
        filelist_test = []
        test_area = val_area + 1 if val_area != 5 else 1

        for i in range(1, 6):
            dataset = sorted(glob.glob(os.path.join(rootdir, "data", "5-fold", f"Area_{i}", "*.npy")))
            if i == val_area:
                filelist_test = filelist_test + dataset
            elif i == test_area:
                filelist_val = filelist_val + dataset
            else:
                filelist_train = filelist_train + dataset
        

        if split == 'train':
            self.filelist = filelist_train
        elif split == 'val':
            self.filelist = filelist_val
        elif split == 'test':
            self.filelist = filelist_test

        if self.training:
            self.number_of_run = 1
            self.iterations = iteration_number
        else:
            self.number_of_run = 16
            self.iterations = None

        print(f'val_area : {val_area} test_area : {test_area} ', end='')
        print(f"{split} files : {len(self.filelist)}")

    def __getitem__(self, index):
        # print(index)
        if self.training:
            index = random.randint(0, len(self.filelist) - 1)
            dataset = self.filelist[index]

        else:
            dataset = self.filelist[index // self.number_of_run]

        # load files
        filename_data = os.path.join(dataset)
        inputs = np.load(filename_data)
        #4(12)枚の大きな画像から256*256の画像を生成
        #4(12)*(1024/256)*(1024/256)=64(192)
        #実質64(192)枚学習
        # split features labels
        if self.training:
            x = random.randint(0, inputs.shape[0] - 256)
            y = random.randint(0, inputs.shape[0] - 256)
        else:
            x = index % self.number_of_run//4 * 256
            y = index % self.number_of_run % 4 * 256

        features = inputs[:,:,0:1].astype(np.float32)#transpose(2, 0, 1)
        features = np.repeat(features, 3, 2)
        #features /= 255.0
        
        labels = inputs[:,:, -1].astype(int)
        features = Image.fromarray(np.uint8(features))
        labels = Image.fromarray(np.uint8(labels))
        if self.transform:
            fts, lbs = self.transform(features, labels)
        #fts = torch.from_numpy(features).float()
        #lbs = torch.from_numpy(labels).long()

        return fts, lbs

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations

    def get_class_count(self):
        label = np.array([np.load(data_path)[:, :, -1] for data_path in self.filelist])
        
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]

        return label_count



################################ Camvid dataset #########################################
class CamvidLoader(data.Dataset):
    # 初期設定
    def __init__(self, path=None, dataset_type='train', transform=None):
        self.path = path
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス
        self.image_list = np.loadtxt(path + "/list/{}_d.txt".format(dataset_type), dtype=str)
        self.label_list = np.loadtxt(path + "/list/{}_l.txt".format(dataset_type), dtype=str)
        if self.image_list.size==1:
            self.image_list = (self.image_list,)
            self.label_list = (self.label_list,)

        # self.dataset_type = dataset = 'train' or 'val' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform

    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        image_name = self.image_list[index]
        label_name = self.label_list[index]

        # 画像読み込み  # Image.open = 読み込み .convert("RGB") = RGBで読み込み
        image = Image.open(self.path + "/image/{}".format(image_name)).convert("RGB")
        # ラベル読み込み # Image.open = 読み込み .convert("L") = GrayScaleで読み込み
        label = Image.open(self.path + "/image/{}".format(label_name)).convert("L")

        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)



################################ Massachusetts road or building datasets #########################################
class MassachusettsDataloader(data.Dataset):
    def __init__(self, root=None, dataset_type='train', transform=None):
        self.root = root
        self.dataset_type = dataset_type
        self.transform = transform

        self.data = natsorted(os.listdir(self.root + "/tiff/{}".format(dataset_type)))
        self.label = natsorted(os.listdir(self.root + "/tiff/{}_labels".format(dataset_type)))
        self.data = np.array(self.data)
        self.label = np.array(self.label)
        
        self.data_path = []
        self.label_path = []
        for i in range(len(self.data)):
            self.data_path.append(self.root + "/tiff/{}/{}".format(dataset_type, self.data[i]))
            self.label_path.append(self.root + "/tiff/{}_labels/{}".format(dataset_type, self.label[i]))
            
        

    def __getitem__(self, index):
        # data
        image_name = self.data_path[index]
        label_name = self.label_path[index]
        image = Image.open(image_name).convert("RGB")
        label = Image.open(label_name).convert("RGB")
        label = np.array(label)
        label = np.where(label[:,:,0]>=150, 0, 1)
        label = Image.fromarray(np.uint8(label))

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.data_path) 



################################ Kvasir-SEG #########################################
class PolypeDataset(data.Dataset):
    def __init__(self, root=None, dataset_type='train',cross='1', transform=None):
        #self.h_image_size, self.w_image_size = image_size[0], image_size[1]
        self.dataset_type = dataset_type
        self.transform = transform
        self.cross = cross

        self.item_image = np.load(root + "datamodel/{}_data_{}.npy".format(self.dataset_type, self.cross))        
        self.item_gt = np.load(root + "datamodel/{}_label_{}.npy".format(self.dataset_type, self.cross))  
        print(np.bincount(self.item_gt.flatten()))      


    def __getitem__(self, index):
        items_im = self.item_image
        items_gt = self.item_gt
        img_name = items_im[index]
        label_name = items_gt[index]
        label_name = np.where(label_name>200, 1, 0)

        image = Image.fromarray(np.uint8(img_name))
        mask = Image.fromarray(np.uint8(label_name))

        #mask = np.eye(2)[mask]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.item_image)



################################ Synapse-Multiorgan dataset #########################################
class SynapseDataset(data.Dataset):

    def __init__(self, root=None, dataset_type='train',cross='1', transform=None):
        #self.h_image_size, self.w_image_size = image_size[0], image_size[1]
        self.dataset_type = dataset_type
        self.transform = transform
        self.cross = cross

        self.item_image = np.load(root + "datamodel/{}_data_{}.npy".format(self.dataset_type, self.cross))        
        self.item_gt = np.load(root + "datamodel/{}_label_{}.npy".format(self.dataset_type, self.cross))  
        print(np.bincount(self.item_gt.flatten()))      


    def __getitem__(self, index):
        items_im = self.item_image
        items_gt = self.item_gt
        img_name = items_im[index]
        label_name = items_gt[index]

        image = Image.fromarray(np.uint8(img_name)).convert("RGB")
        mask = Image.fromarray(np.uint8(label_name))

        #mask = np.eye(2)[mask]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.item_image)



################################ Cityscapes dataset #########################################
class Cityscapse_Loader(data.Dataset):

    # 初期設定
    def __init__(self, root_dir=None, dataset_type='train', transform=None):
        # self.image_path = 画像のパス
        # self.label_path = ラベルのパス

        #入力画像とラベルの上位ディレクトリ取得
        image_path = "{}/leftImg8bit/{}".format(root_dir, dataset_type)
        label_path = "{}/gtFine/{}".format(root_dir, dataset_type)

        image_list = sorted(glob.glob(image_path + "/*"))
        label_list = sorted(glob.glob(label_path + "/*"))

        # image_list = 画像のパスのリスト (sorted = 昇順)
        # label_list = ラベルのパスのリスト (sorted = 昇順)
        image_list = [glob.glob(path + "/*.png") for path in image_list]
        label_list = [glob.glob(path + "/*labelIds.png") for path in label_list]

        #2d array => 1d array
        image_list = sorted([path for paths in image_list for path in paths])
        label_list = sorted([path for paths in label_list for path in paths])

        #ignore_label define last class,because one_hot_changer dose not support -1
        ignore_label = 19

        #class id process
        self.label_mapping = {-1: ignore_label,
                            0: ignore_label,
                            1: ignore_label,
                            2: ignore_label,
                            3: ignore_label,
                            4: ignore_label,
                            5: ignore_label,
                            6: ignore_label,
                            7: 0,
                            8: 1,
                            9: ignore_label,
                            10: ignore_label,
                            11: 2,
                            12: 3,
                            13: 4,
                            14: ignore_label,
                            15: ignore_label,
                            16: ignore_label,
                            17: 5,
                            18: ignore_label,
                            19: 6,
                            20: 7,
                            21: 8,
                            22: 9,
                            23: 10,
                            24: 11,
                            25: 12,
                            26: 13,
                            27: 14,
                            28: 15,
                            29: ignore_label,
                            30: ignore_label,
                            31: 16,
                            32: 17,
                            33: 18}


        #画像読み込み
        print("Loading Cityscapse image data")
        #this code is faster than "Image.open(image_name).convert("RGB")"
        self.ToPIL = transforms.ToPILImage()
        self.image_list = [self.ToPIL(cv2.imread(image_name)).convert("RGB") for image_name in (image_list)]

        print("Loading Cityscapse label data")
        self.label_list = [self.ToPIL(self.convert_label(cv2.imread(label_list, -1)))
                        for label_list in (label_list)]

        # self.dataset_type = dataset = 'train' or 'val' or 'test'
        self.dataset_type = dataset_type

        # self.transform = transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        #入力画像、ラベル取得
        image = self.image_list[index]
        label = self.label_list[index]


        # もしself.transformが有効なら
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def convert_label(self, label):
        temp = label.copy()
        for k, v in self.label_mapping.items():
            label[temp == k] = v
        return label

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.image_list)

    def get_class_count(self):
        label = np.array([np.array(label) for label in self.label_list])
        label_num, label_count = np.unique(label, return_counts=True)
        sort_idx = np.argsort(label_num)
        label_num = label_num[sort_idx]
        label_count = label_count[sort_idx]
        return label_count


################################ Trans10k dataset #########################################
class TransSegmentation(SegmentationDataset_10k):
    """Trans10K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Trans10K folder. Default is './datasets/Trans10K'
    split: string
        'train', 'validation', 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = 'Trans10K'
    NUM_CLASS = 3

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        super(TransSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please put dataset in {SEG_ROOT}/datasets/Trans10K"
        self.images, self.mask_paths = _get_trans10k_pairs(root, split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        self.valid_classes = [0,1,2]
        self._key = np.array([0,1,2])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32') + 1

    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)

        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index])
        
        # mask
        mask = np.array(mask)[:,:,:3].mean(-1)
        mask[mask==85.0] = 1
        mask[mask==255.0] = 2
        assert mask.max()<=2, mask.max()
        mask = Image.fromarray(mask)

        # synchrosized transform
        img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('background', 'things', 'stuff')


def _get_trans10k_pairs(folder, split='train'):

    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        imgs = os.listdir(img_folder)

        for imgname in imgs:
            imgpath = os.path.join(img_folder, imgname)
            maskname = imgname.replace('.jpg', '_mask.png')
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                logging.info('cannot find the mask or image:', imgpath, maskpath)

        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths


    if split == 'train':
        img_folder = os.path.join(folder, split, 'images')
        mask_folder = os.path.join(folder, split, 'masks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    else:
        assert split == 'val' or split == 'test'
        easy_img_folder = os.path.join(folder, split, 'easy', 'images')
        easy_mask_folder = os.path.join(folder, split, 'easy', 'masks')
        hard_img_folder = os.path.join(folder, split, 'hard', 'images')
        hard_mask_folder = os.path.join(folder, split, 'hard', 'masks')
        easy_img_paths, easy_mask_paths = get_path_pairs(easy_img_folder, easy_mask_folder)
        hard_img_paths, hard_mask_paths = get_path_pairs(hard_img_folder, hard_mask_folder)
        easy_img_paths.extend(hard_img_paths)
        easy_mask_paths.extend(hard_mask_paths)
        img_paths = easy_img_paths
        mask_paths = easy_mask_paths
    return img_paths, mask_paths