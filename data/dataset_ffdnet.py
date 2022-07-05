import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util

def norm_change(img, new_norm):
    frac = new_norm / (torch.norm(img))
    return img*frac

def calc_norm(data, gray_scale=False):
    if gray_scale:
        data_norm = data.norm(dim=(2, 3)).squeeze(1)
    else:
        data_norm = torch.norm(data.norm(dim=(2, 3)), dim=1)
    return data_norm

class DatasetFFDNet(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H/M for denosing on AWGN with a range of sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., FFDNet, H = f(L, sigma), sigma is noise level
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetFFDNet, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 75]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 25

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

        self.new_norm = opt['new_norm'] if opt['new_norm'] else 37.82235769987106
        self.baseline = opt['baseline'] if opt['baseline'] else True
        self.real_norm  = opt['real_norm'] if  opt['real_norm'] else False
        self.constant_norm  = opt['constant_norm'] if  opt['constant_norm'] else True

    def set_test_sigma(self, sigma):
        self.sigma_test = sigma

    def __getitem__(self, index):
        # -------------------------------------
        # get H image
        # -------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path
        gray_scale = self.n_channels == 1

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            """
            H, W = img_H.shape[:2]

            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            if not self.baseline and self.constant_norm:
                img_L = norm_change(img_L, self.new_norm)

            # ---------------------------------
            # get noise level
            # ---------------------------------
            # noise_level = torch.FloatTensor([np.random.randint(self.sigma_min, self.sigma_max)])/255.0
            noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0

            # ---------------------------------
            # add noise
            # ---------------------------------
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)
            ynorm =  torch.norm(img_L)
            xnorm = torch.norm(img_H)
            
            if not self.baseline:
                if self.constant_norm:
                    img_L.mul_(1.0/((noise_level)**2))
                else:
                    d = self.patch_size**2 * self.n_channels

                    if self.real_norm:
                        norm_frac = (xnorm)
                    else:
                        norm_frac = (torch.sqrt(ynorm**2 - d * (noise_level**2)))
                    ##norm_frac = self.new_norm / torch.sqrt(calc_norm(img_L, gray_scale)**2 - d * (noise_level**2))
                    img_L.mul_(norm_frac/(noise_level**2))

        else:
            """
            # --------------------------------
            # get L/H/sigma image pairs
            # --------------------------------
            """

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = (H - self.patch_size)//2
            rnd_w = (W - self.patch_size)//2
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            img_H = util.uint2single(patch_H)
            img_L = np.copy(img_H)
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            ynorm = np.sqrt(np.sum(img_L**2))
            xnorm = np.sqrt(np.sum(img_H**2))
            noise_level = torch.FloatTensor([self.sigma_test/255.0])
            
            if self.constant_norm:
                assert(self.new_norm == 1.0)


            if not self.baseline:
                d = self.patch_size**2 * self.n_channels

                if self.real_norm:
                    norm_frac = (xnorm/self.new_norm)
                else:
                    norm_frac = (torch.sqrt(ynorm**2 - d * (noise_level**2))/self.new_norm)
                ##norm_frac = self.new_norm / torch.sqrt(calc_norm(img_L, gray_scale)**2 - d * (noise_level**2))
                
                img_L.mul_(norm_frac/(noise_level**2))

            # ---------------------------------
            # L/H image pairs
            # ---------------------------------
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        noise_level = noise_level.unsqueeze(1).unsqueeze(1)


        return {'L': img_L, 'H': img_H, 'C': noise_level, 'L_path': L_path, 'H_path': H_path, 'ynorm': ynorm, 'xnorm': xnorm}

    def __len__(self):
        return len(self.paths_H)
