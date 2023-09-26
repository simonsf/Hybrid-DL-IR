import os
import json
import time
import shutil
import matplotlib
matplotlib.use('Agg')

import scipy.misc as misc
from collections import OrderedDict
import torch 
# torch.autograd.set_detect_anomaly(False)
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil

from utils import util
from loss import create_loss
from .base_solver import BaseSolver
from matplotlib import pyplot as plt
import md.image3d.python.image3d_io as cio
from models import create_model, init_weights
from md.mdpytorch.utils.tensor_tools import Image3d

plt.rcParams.update({'font.size': 6})
plt.ioff(); plt.close('all')


class ACSSolver(BaseSolver):
    """
    Slover for training, testing, logging and visualization of aCS project
    """
    def __init__(self, opt):
        super(ACSSolver, self).__init__(opt)
        self.mask = self.Tensor()
        self.sub_img = self.Tensor()
        self.masked_kspace = self.Tensor()
        self.target = self.Tensor()
        self.sub_rss = self.Tensor()
        self.pred_rss = None
        self.num_low_frequencies = 0
        self.mask_direction = 'x'
        self.train_opt = opt['solver']
        self.records = {'train_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': [],
                        'lr': []}

        # create directory
        if self.is_train and self.opt['solver']['pretrain'] is None:
            assert not (os.path.exists(self.checkpoint_dir) and len(os.listdir(self.checkpoint_dir))), "Directory with checkpoints file exist cannot be remove!!!"
            if os.path.exists(self.exp_root):
                shutil.rmtree(self.exp_root)
            # assert not os.path.exists(self.exp_root), "Experimental directory already exist"
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.records_dir)
            os.makedirs(self.visual_dir)

        # save pid
        if self.is_train:
            with open(os.path.join(self.records_dir, 'info.txt'), 'w') as f:
                print('--->pid:%d' % (os.getpid()))
                f.write('pid:\n%d' % (os.getpid()))

        self.model = create_model(opt)

        if self.is_train:
            self.model.train()

            # set cl_loss
            if self.use_cl:
                self.cl_weights = self.opt['solver']['cl_weights']
                assert self.cl_weights, "[Error] 'cl_weights' is not be declared when 'use_cl' is true"

            # set loss
            self.criterion_pix = create_loss(self.train_opt)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            elif optim_type == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(),
                                            lr=self.train_opt['learning_rate'], momentum=self.train_opt['momentum'], weight_decay=weight_decay)

            else:
                raise NotImplementedError('Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Only MultiStepLR scheme is supported!')

        self.load()
        self.print_network()
        self.save_config()
        print('===> Solver Initialized : [%s] || Use CL : [%s] || Use GPU : [%s]'%(self.__class__.__name__,
                                                                                       self.use_cl, self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: %s   gamma: %f"%(self.scheduler.milestones, self.scheduler.gamma))                                                                                      
        

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]'%init_type)
        init_weights(self.model, init_type)


    def load(self):
        """
        load or initialize network
        """
        if (self.is_train and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None: raise ValueError("[Error] The 'pretrained_path' does not declarate in *.json")

            print('===> Loading model from [%s]...' % model_path)
            if self.is_train:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['state_dict'])

                if self.opt['solver']['pretrain'] == 'resume':
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    # self.best_pred = checkpoint['best_pred']
                    # self.best_epoch = checkpoint['best_epoch']
                    # self.records = checkpoint['records']

            else:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'state_dict' in checkpoint.keys(): checkpoint = checkpoint['state_dict']
                load_func = self.model.module.load_state_dict if isinstance(self.model, nn.DataParallel) \
                    else self.model.load_state_dict
                load_func(checkpoint)
                print('===> Load checkpoint for test done!')

        else:
            # self._net_init()
            pass


    def save_config(self):
        """
        Save json format config
        """
        if not os.path.exists(self.records_dir):
            os.makedirs(self.records_dir)

        with open(f"{self.records_dir}/config.json", "w") as file:
            json.dump(self.opt, file, indent=2)
            print("===> saving config\n")
    
    
    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                                 self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'), 'w') as f:
                f.writelines(net_lines)

        print("==================================================")


    def get_current_log(self):
            log = OrderedDict()
            log['epoch'] = self.cur_epoch
            log['best_pred'] = self.best_pred
            log['best_epoch'] = self.best_epoch
            log['records'] = self.records
            return log


    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']


    def feed_data(self, batch, need_target=True):
        # norm data before entering the image finetuning model
        self.mask.resize_(batch.mask.size()).copy_(batch.mask)
        # self.mask = self.mask.to(torch.bool)
        self.sub_img.resize_(batch.sub_img.size()).copy_(batch.sub_img)
        self.masked_kspace.resize_(batch.masked_kspace.size()).copy_(batch.masked_kspace)
        # target shoule be the RSS image for loss calculation
        self.target.resize_(batch.target.size()).copy_(batch.target)
        self.sub_rss.resize_(batch.masked_kspace_rss.size()).copy_(batch.masked_kspace_rss)
        self.num_low_frequencies = batch.num_low_frequencies
        self.mask_direction = batch.mask_direction
        self.fname = batch.fname
        self.rss_range.resize_(batch.max_value.size()).copy_(batch.max_value)


    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()

        # loss_batch = 0.0
        # forward pass
        # output = self.model(self.masked_kspace, self.mask, self.mask_direction, self.num_low_frequencies)
        output = self.model(self.sub_img, self.masked_kspace, self.mask)
        if self.train_opt['loss_type'] == "ssim" or self.train_opt['loss_type'] == "l1ssim":
            loss_batch = self.criterion_pix(output[:,None,:,:], self.target[:,None,:,:], self.rss_range)
        else:
            loss_batch = self.criterion_pix(output[:,None,:,:]/self.rss_range, self.target[:,None,:,:]/self.rss_range) * self.train_opt['loss_lambda_A']
        # with torch.autograd.detect_anomaly():
        loss_batch.backward()

        # loss_batch += (loss_sbatch.item())

        # for stable training
        if loss_batch < self.skip_threshold * self.last_epoch_loss:
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0, norm_type=2)
            self.optimizer.step()
            self.last_epoch_loss = loss_batch
        else:
            print('[Warning] Skip this batch! (Loss: {})'.format(loss_batch))

        self.model.eval()
        return {
            "fname": self.fname,
            "loss": loss_batch.item()
        }

    def update_learning_rate(self):
        self.scheduler.step()

    
    def val(self):
        self.model.eval()
        with torch.no_grad():
            # output = self.model.forward(self.masked_kspace, self.mask, self.mask_direction, self.num_low_frequencies)
            output = self.model(self.sub_img, self.masked_kspace, self.mask)
            # rss
            self.pred_rss = output
            if self.is_train:
                if self.train_opt['loss_type'] == "ssim" or self.train_opt['loss_type'] == "l1ssim":
                    loss_batch = self.criterion_pix(output[:,None,:,:], self.target[:,None,:,:], self.rss_range)
                else:
                    loss_batch = self.criterion_pix(output[:,None,:,:]/self.rss_range, self.target[:,None,:,:]/self.rss_range) * self.train_opt['loss_lambda_A']

            return {
                "fname": self.fname,
                "loss": loss_batch.item()
            }

    
    def test(self, acc_dict):
        self.model.eval()
        with torch.no_grad():
            case_img3d = Image3d()
            for key_acc, slice_dict in acc_dict.items():
                success_cases = 0
                slicenum = len(slice_dict)
                _, h_f, w_f = slice_dict[1].target.shape
                # prepare some matrix
                pre_volume = np.zeros((slicenum, h_f, w_f), dtype='float64')
                GT_volume = np.zeros((slicenum, h_f, w_f), dtype='float64')
                SUB_volume = np.zeros((slicenum, h_f, w_f), dtype='float64')
                total_test_time = 0
                begin_test = time.time()
                torch.cuda.synchronize()
                for key_slice, batch in slice_dict.items():
                    self.feed_data(batch)
                    # print(f"Current test: {self.fname[0]}")
                    output = self.model.forward(self.masked_kspace, self.mask, self.mask_direction, self.num_low_frequencies)
                    self.pred_rss = output

                    torch.cuda.synchronize()
                    net_time = time.time() - begin_test

                    if success_cases != 0:
                        total_test_time = net_time + total_test_time
                        print(' net_test: {:.4f} s, avg test time: {:.4f}'.format(net_time, total_test_time / float(success_cases)))
                    success_cases += 1

                    pre_volume[key_slice-1, :, :] = self.pred_rss.cpu().detach().numpy()
                    GT_volume[key_slice-1, :, :] = self.target.cpu().detach().numpy()
                    SUB_volume[key_slice-1, :, :] = self.sub_rss.cpu().detach().numpy()
                    
                # generate data saving path
                cropidx, organ, protocal = self.fname[0].split('/')[8:11]
                uid = self.fname[0].split('/')[-1]

                out_folder_v = os.path.join(self.test_root, 'visualization', cropidx, organ, protocal, uid, key_acc)
                if not os.path.isdir(out_folder_v):
                    os.makedirs(out_folder_v)

                prefilename = out_folder_v + '/' + 'Pred_varnet.mhd'
                GTfilename = out_folder_v + '/' + 'GT_varnet.mhd'
                subfilename = out_folder_v + '/' + 'Sub_varnet.mhd'

                case_img3d.from_numpy(pre_volume)
                cio.write_image(case_img3d, prefilename, dtype=np.float64, compression=True)
                case_img3d.from_numpy(GT_volume)
                cio.write_image(case_img3d, GTfilename, dtype=np.float64, compression=True)
                case_img3d.from_numpy(SUB_volume)
                cio.write_image(case_img3d, subfilename, dtype=np.float64, compression=True)
                # maskfilename = out_folder_v + '/' 'mask.mhd'
                # mask_np = mask_cpu[0,:,:,:,:][:,0,:,:].numpy()
                # case_img3d.from_numpy(mask_np)
                # cio.write_image(case_img3d, maskfilename, dtype=np.float32, compression=True)
                
    
    def get_current_visual(self, need_GT=True):
        """
        return input_SSR pred_SSR (gt_SSR) images
        """
        out_dict = OrderedDict()
        out_dict['IN_SSR'] = self.sub_rss
        out_dict['PRED_SSR'] = self.pred_rss
        out_dict['VAL_RANGE'] = self.rss_range
        out_dict['UID_ACC'] = self.fname[0].split('/mhd/')[-1]
        if need_GT:
            out_dict['GT_SSR'] = self.target

        return out_dict


    def save_checkpoint(self, epoch, is_best=False):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]'%filename)
        ckp = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'best_pred': self.best_pred,
            # 'best_epoch': self.best_epoch,
            # 'records': self.records
        }
        torch.save(ckp, filename)
        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' % filename.replace('last_ckp','best_ckp'))
            torch.save(ckp, filename.replace('last_ckp','best_ckp'))

        if epoch % self.train_opt['save_ckp_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' % (epoch,
                                                                filename.replace('last_ckp','epoch_%d_ckp'%epoch)))

            torch.save(ckp, filename.replace('last_ckp','epoch_%d_ckp'%epoch))


    def set_current_log(self, log):
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_epoch = log['best_epoch']
        self.records = log['records']


    def save_current_visual(self, epoch, iter, vis_count):
        """
        save visual results for comparison
        """
        # TODO: add acc factor in save PNG name
        if epoch % self.save_vis_step == 0:
            # visuals_list = []
            visuals = self.get_current_visual()
            plt.close()
            plt.figure()
            plt.suptitle(f"{visuals['UID_ACC']}")
            plt.subplot(1, 3, 1)
            plt.imshow((visuals['IN_SSR'][0]/visuals['VAL_RANGE']).cpu().detach().numpy(), vmin=0., vmax=0.99, cmap='gray')
            plt.axis('off'); plt.title(f"INPUT RSS")
            plt.subplot(1, 3, 2)
            plt.imshow((visuals['PRED_SSR'][0]/torch.max(visuals['PRED_SSR'])).cpu().detach().numpy(), vmin=0., vmax=0.99, cmap='gray')
            plt.axis('off'); plt.title('PRED RSS')
            plt.subplot(1, 3, 3)
            plt.imshow((visuals['GT_SSR'][0]/visuals['VAL_RANGE']).cpu().detach().numpy(), vmin=0., vmax=0.99, cmap='gray')
            plt.axis('off'); plt.title('GT RSS')
            plt.tight_layout()
            plt.savefig(os.path.join(self.visual_dir, 'epoch_%d_iteration_%d_img_%d_%s.png' % (epoch, iter + 1, vis_count, visuals['UID_ACC'].split('/')[-1])), dpi=300)
            plt.clf()
            plt.close()

