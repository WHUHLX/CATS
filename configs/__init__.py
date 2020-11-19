#!/user/bin/python
# -*- encoding: utf-8 -*-

from os.path import join

class Config(object):
    def __init__(self):
        self.data = "bsds"
        # ============== training
        self.resume = "./pretrained/{}.pth".format(self.data)
        self.msg_iter = 500
        self.gpu = '1'
        self.save_pth = join("./output", self.data)
        self.pretrained = "./pretrained/vgg16.pth"

        # ============== testing
        self.multi_aug = False

        # ================ dataset
        self.dataset = "./data/{}".format(self.data)

        # =============== optimizer
        self.batch_size = 1
        self.lr = 1e-6
        self.momentum = 0.9
        self.wd = 2e-4
        self.stepsize = 10
        self.gamma = 0.1
        self.max_epoch = 30
        self.itersize = 10

