# create by 
import os
import torch


class BaseSolver(object):
    """
    Solver of MRI reconstruction ACS project
    """
    def __init__(self, opt):
        self.opt = opt
        self.fname = None
        # self.scale = opt['scale']
        self.is_train = opt['is_train']
        # self.use_chop = opt['use_chop']
        # self.self_ensemble = opt['self_ensemble']
        self.use_cl = True if opt['use_cl'] else False

        # GPU verify
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        # self.Tensor_bool = torch.cuda.BoolStorage if self.use_gpu else torch.BoolTensor
        self.rss_range = torch.cuda.FloatTensor() if self.use_gpu else torch.FloatTensor()

        # for better training (stablization and less GPU memory usage)
        self.last_epoch_loss = 1e8
        if self.is_train:
            self.skip_threshold = opt['solver']['skip_threshold']
            # save GPU memory during training
            self.split_batch = opt['solver']['split_batch']

        # experimental dirs
        self.exp_root = opt['path']['exp_root']
        if not self.is_train:
            self.test_root = opt['path']['test_root']
        self.checkpoint_dir = os.path.join(self.exp_root, "checkpoints")
        self.records_dir = os.path.join(self.exp_root, "records")
        self.visual_dir = os.path.join(self.exp_root, "visual_dir")
        # self.checkpoint_dir = opt['path']['epochs']
        # self.records_dir = opt['path']['records']
        # self.visual_dir = opt['path']['visual']

        # log and vis scheme
        if self.is_train:
            self.save_ckp_step = opt['solver']['save_ckp_step']
            self.save_vis_step = opt['solver']['save_vis_step']

        self.best_epoch = 0
        self.cur_epoch = 1
        self.best_pred = 0.0

    def initialize(self):
        pass

    def feed_data(self, batch):
        pass

    def train_step(self):
        pass

    def val(self):
        pass
    
    
    def test(self, acc_dict):
        pass

    def get_current_log(self):
        pass


    def get_current_visual(self):
        pass

    def get_current_learning_rate(self):
        pass

    def set_current_log(self, log):
        pass

    def update_learning_rate(self, epoch):
        pass

    def save_checkpoint(self, epoch, is_best):
        pass

    def load_chekpoint(self):
        pass

    def save_current_visual(self, epoch, iter, vis_count):
        pass

    def save_current_log(self):
        pass

    def print_network(self):
        pass

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, torch.nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))

        return s, n