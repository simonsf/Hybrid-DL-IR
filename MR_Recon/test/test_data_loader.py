# create by
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
import json
import torch
import random
sys.path.insert(0, '.')
import argparse

from solver import create_solver
from collections import OrderedDict
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


    for epoch in range(0, 50 + 1):
        for iter, batch in enumerate(train_loader):
            print(batch.fname)


if __name__ == '__main__':
    args = Options().parse()
    # import torch
    main(args)





