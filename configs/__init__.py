#!/user/bin/python
# -*- encoding: utf-8 -*-

from os.path import join

class Config(object):
    def __init__(self):
        self.data = "bsds"
        # ============== training
        self.resume = "./pretrained/{}.pth".format(self.data)
        self.msg_iter = 500
        self.gpu = '0'
        self.save_pth = join("./output", self.data)
        self.pretrained = "./pretrained/vgg16.pth"
        self.aug = False

        # ============== testing
        self.multi_aug = False # Produce the multi-scale results
        self.side_edge = False # Output the side edges

        # ================ dataset
        self.dataset = "./data/{}".format(self.data)

        # =============== optimizer
        self.batch_size = 1
        self.lr = 1e-6
        self.momentum = 0.9
        self.wd = 2e-4
        self.stepsize = 5
        self.gamma = 0.1
        self.max_epoch = 30
        self.itersize = 10

