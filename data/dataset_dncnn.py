import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


def norm_change(img, new_norm):
    frac = new_norm / (torch.norm(img))
    return img*frac

class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        self.sigma_range = opt['sigma_range'] if opt['sigma_range'] else 0
        self.baseline = opt['baseline'] if opt['baseline'] else True
        self.new_norm = opt['new_norm'] if opt['new_norm'] else 45.167343

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------

            if not self.baseline:
                img_L = norm_change(img_L, self.new_norm)

            if self.sigma_range > 0:
                sigma = (torch.rand([1])*2-1) * self.sigma_range + self.sigma 
            else:
                sigma = self.sigma
            noise = torch.randn(img_L.size()).mul_(sigma/255.0)

            img_L.add_(noise)


            if not self.baseline:
                img_L.mul_(1.0/(sigma/255.0)**2)

            

        else:


            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = (H - self.patch_size)//2
            rnd_w = (W - self.patch_size)//2
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(patch_H)
            img_L = np.copy(img_H)

            if not self.baseline:
                img_L = norm_change(img_L, self.new_norm)

            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)

            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            if not self.baseline:
                img_L.mul_(1.0/(self.sigma_test/255.0)**2)
            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path, 'sigma': self.sigma}

    def __len__(self):
        return len(self.paths_H)
