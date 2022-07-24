import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
import wandb


from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model




def norm_change(img, new_norm):
    frac = new_norm / (torch.norm(img))
    return img*frac


'''
# --------------------------------------------
# training code for DnCNN
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
#         https://github.com/cszn/DnCNN
#
# Reference:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

def main(json_path='options/train_dncnn.json'):


    dataset = 'bsd'
    gray_scale = False
    new_norm = 45.167343  # mean of the norm of bsd images with patches of 80*80
    input_channels = 3
    lambd = lambda x: norm_change(x, new_norm)

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    border = 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['train']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)


    opt["Version"] = 0.2
    wandb.init(project="dncnn", entity="dl-projects" )
    wandb.config.update(opt)
    wandb.Table.MAX_ROWS = len(test_loader) * len(opt['tsigma'])

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    ##table_art = wandb.Artifact("table_artifact_" + str(wandb.run.id), type="predictions")



    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            if dataset_type == 'dnpatch' and current_step % 20000 == 0:  # for 'train400'
                train_loader.dataset.update_data()

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)
            # -------------------------------
            # 3) optimize parameters
            # -------------------------------

            if opt['train']['G_lossfn_type'] in ['WL2']:
                loss_weights = train_data['C']
                model.set_loss_weights(loss_weights)

            model.optimize_parameters(current_step)

            # -------------------------------
            # merge bnorm
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss

                log_data = {"Epoch": epoch,
                            "Learning Rate": model.current_learning_rate()}
                
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                    log_data[k] = v
                logger.info(message)

                wandb.log(log_data, step = current_step)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:


                test_results = {}
                my_table = wandb.Table(columns=["noise", "img_idx" , "image" ,"psnr"])

                for sigma_i,tsigma in enumerate(opt['tsigma']):
                
                    test_loader.dataset.set_test_sigma(tsigma)
                    avg_psnr = 0.0
                    idx = 0

                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)

                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(test_data)
                        model.netG.module.feed_sigma(tsigma)
                        model.netG.module.feed_ynorm(test_data['ynorm'])
                        model.netG.module.feed_xnorm(test_data['xnorm'])

                        model.test()

                        visuals = model.current_visuals()
                        E_img = util.tensor2uint(visuals['E'])
                        H_img = util.tensor2uint(visuals['H'])

                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        # save_img_path = os.path.join(img_dir, '{:s}_{:d}_{:d}.png'.format(img_name, tsigma ,current_step))
                        # util.imsave(E_img, save_img_path)


                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                        my_table.add_data(tsigma,idx, wandb.Image(E_img, caption=img_name), current_psnr)

                        ##logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                        avg_psnr += current_psnr
                    avg_psnr = avg_psnr / idx

                    # testing log
                    logger.info('<epoch:{:3d}, iter:{:8,d}, sigma:{:3d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step , tsigma , avg_psnr))
                    test_results['psnr_%.02f' % tsigma] = avg_psnr
                    test_results["denoised images"] =my_table

                wandb.log(test_results, step = current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
