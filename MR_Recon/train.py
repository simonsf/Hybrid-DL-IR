# import imp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import json
import torch
import random
import argparse
from tqdm import tqdm

from loss.ssim import SSIM
from utils import util
from solver import create_solver
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from data.utils import create_dataloader, create_dataset


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Acs project argparse')
        parser.add_argument('--config_path', type=str, 
             default="/data1/xinglie/ACS_2D_MC/code_share/ACS_2d_15T_sim_wxl/config/share_config.json",
            help='json format config file')
        # the parser
        self.parser = parser
     
    def parse(self):
        args = self.parser.parse_args()
        print(args)
        # get all configuration dictionary
        # remove comments starting with '//'
        json_str = ''
        with open(args.config_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)

        # # export CUDA_VISIBLE_DEVICES
        # if torch.cuda.is_available():
        #     gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        #     print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')
        # else:
        #     print('===> CPU mode is set (NOTE: GPU is recommended)')

        return opt


def main(opt):
    # ------------
    # seed
    # ------------
    manualSeed = opt['solver']['manual_seed']
    if manualSeed is None: seed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = True
    print("===> Random Seed: [%d]"%manualSeed)

    # ------------
    # logging
    # ------------
    writer = SummaryWriter("ACS_reconstruction_log")

    # # ------------
    # # data
    # # ------------
    # create train and val dataloader
    # E2E Varnet中的图像是在图像输入图像优化分支cascade Unet前进行了标准化
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset has total [%d] cases ' % (len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset has total [%d] cases ' % (len(val_set)))

        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)
    
    # ------------
    # solver
    # ------------
    solver = create_solver(opt)
    # scale = opt['scale']
    model_name = opt['networks']['which_model'].upper()

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    print("Method: %s || Epoch Range: (%d ~ %d)"%(model_name, start_epoch, NUM_EPOCH))

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> Training Epoch: [%d/%d]...  Learning Rate: %f'%(epoch,
                                                                      NUM_EPOCH,
                                                                      solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch
         # Train model
        train_loss_list = []
        with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                solver.feed_data(batch)
                result = solver.train_step()
                iter_loss = result["loss"]
                batch_size = batch.masked_kspace.size(0)
                train_loss_list.append(iter_loss*batch_size)
                t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
                t.update()
 
                writer.add_scalar("train/pixel loss", iter_loss, (epoch - 1) * len(train_loader) + iter)
        
        solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f' % (epoch,
                                                    NUM_EPOCH,
                                                    sum(train_loss_list)/len(train_set)))
                                                    
        if epoch % opt['solver']['validation_epoch'] == 0:
            # psnr_list = []
            ssim_list = []
            val_loss_list = []
            print('===>Partially Visualization...',)
            val_idx = [random.randint(0, len(val_loader) - 1) for i in range(opt['solver']['num_slice_vis']) ]
            vis_count = 1
            with tqdm(total=len(val_idx), desc='Epoch: [%d/%d]' % (epoch, NUM_EPOCH), miniters=1) as t:
                for iter, batch in enumerate(val_loader):
                    if iter not in val_idx:
                        continue
                    # forward pass and visualization:
                    solver.feed_data(batch)
                    result = solver.val()
                    iter_loss = result["loss"]

                    # calculate evaluation metrics
                    visuals = solver.get_current_visual()
                    ssim = SSIM(val_range=visuals["VAL_RANGE"]).cuda()
                    # 要把维度扩展成4 也就是[n,c,h,w]的格式
                    ssim = ssim(visuals["PRED_SSR"][:,None, :, :], visuals["GT_SSR"][:,None, :, :])
                    # psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    val_loss_list.append(iter_loss)

                    if opt["save_image"]:
                        solver.save_current_visual(epoch, iter, vis_count)
                        vis_count += 1

            solver_log['records']['ssim'].append(sum(ssim_list)/len(ssim_list))
            solver_log['records']['val_loss'].append(sum(val_loss_list)/len(val_loss_list))
            solver_log['epoch'] = epoch

            print("Sample Average SSIM: %.4f   Loss: %.6f   in Epoch: [%d]" % (sum(ssim_list)/len(ssim_list),
                                                                            sum(val_loss_list)/len(val_loss_list),
                                                                            solver_log['epoch']))

        # TODO
        # solver.set_current_log(solver_log)

        # save checkpoint
        solver.save_checkpoint(epoch)

        # update 
        solver.update_learning_rate()

    # close the writer of tensoroard
    writer.close()
    print('===> Finished !')


if __name__ == '__main__':
    args = Options().parse()
    # import torch
    main(args)
