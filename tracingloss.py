import numpy as np
import torch
from loss import *

def bdrloss(prediction, label, radius):
	'''
	The boundary tracing loss that handles the confusing pixels.
	'''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.cuda()

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)
    

    
    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)
    
    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()



def textureloss(prediction, label, mask_radius=radius):
	'''
	The texture suppression loss that smooths the texture regions.
	'''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.cuda()
    filt2 = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt2.requires_grad = False
    filt2 = filt2.cuda()
    
    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0
    
    return torch.sum(loss)


def tracingloss(prediction, label, tex_factor=0., bdr_factor=0., balanced_w=1.1):
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()
        
        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        mask[mask == 1] = beta
        mask[mask == 0] = balanced_w * (1 - beta)
        mask[mask == 2] = 0
    
    cost = torch.sum(torch.nn.functional.binary_cross_entropy(
                prediction.float(),label.float(), weight=mask, reduce=False))
    label_w = (label != 0).float()

    textcost = textureloss(prediction.float(),label_w.float(), radius=4)
    bdrcost = bdrloss(prediction.float(),label_w.float(),radius=4)

    return cost + bdr_factor*bdrcost + tex_factor*textcost