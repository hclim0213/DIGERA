import torch.nn.functional as F
from .loss_utils import *
import torch


def point_wise_mse(label, predict):
    loss = mse(predict, label)
    return loss

def point_wise_rmse(label, predict):
    loss = torch.sqrt(mse(predict, label))
    return loss

def classification_cross_entropy(label, predict):
    shape = predict.size()
    label = label.view(shape[0] * shape[1])
    predict = predict.view(shape[0] * shape[1], shape[2])
    loss = ce(predict, label)
    return loss


def list_wise_rankcosine(label, predict):
    loss = torch.mean((1.0 - cos(predict, label)) / 0.5)
    return loss