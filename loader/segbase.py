"""Base segmentation dataset"""
import random
import numpy as np
import os
import torchvision

from PIL import Image, ImageOps, ImageFilter



__all__ = ['SegmentationDataset']


class SegmentationDataset_10k(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset_10k, self).__init__()
        # self.root = os.path.join(cfg.ROOT_PATH, root)
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = self.to_tuple(crop_size)

    def to_tuple(self, size):
        if isinstance(size, (list, tuple)):
            return tuple(size)
        elif isinstance(size, (int, float)):
            return tuple((size, size))
        else:
            raise ValueError('Unsupport datatype: {}'.format(type(size)))

    def _val_sync_transform(self, img, mask):
        short_size = self.base_size
        img = img.resize((short_size, short_size), Image.BILINEAR)
        mask = mask.resize((short_size, short_size), Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return Image.fromarray(np.array(img))

    def _mask_transform(self, mask):
        return Image.fromarray(np.array(mask).astype('int32'))

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0