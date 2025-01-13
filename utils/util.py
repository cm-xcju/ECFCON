import torch.nn as nn
import torch
import csv
from pdb import set_trace as stop
import numpy as np
import scipy
from torch import functional as F
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def override_from_dict(opt, dict_):
    for key, value in dict_.items():
        setattr(opt, key, value)

def get_criterion(opt, emo2idx, cause2idx):
  
    return nn.CrossEntropyLoss(ignore_index=-1), nn.CrossEntropyLoss(ignore_index=-1)
    if opt.for_cause:
        weight = torch.ones(opt.context_length+1)
        oth = cause2idx['oth']
        # weight[oth] = 0.1
        return nn.CrossEntropyLoss()
    elif opt.for_emotion:
        #[1.2,59,37, 18.1,174,906,551]
        # x 0.1  [0.12,5.9,3.7, 1.8,17.1,90,55]
        if opt.emotion_class == 'all':
            weight = [0.12, 5.9, 3.7, 1.8, 17.1, 90, 55]
        elif opt.emotion_class == '3-NPN':
            # 47724 1212 4528 53464
            weight = [0.112, 4.4, 1.18]
        elif opt.emotion_class == '2-EN':
            # 47724 5740 53464
            weight = [0.112, 0.9314]
        weight = torch.Tensor(weight)
        # weight = torch.ones(opt.emo_outlen)
        # neu = emo2idx['中性']
        # weight[neu] = 0.05

        print('\n {} \n'.format(weight))
    return nn.CrossEntropyLoss(weight)

def save_loss_model(opt, epoch_i, model, valid_loss, valid_losses):
    model_state_dict = model.state_dict()
    checkpoint = {'model': model_state_dict, 'settings': opt, 'epoch': epoch_i}
    if opt.save_mode == 'all':
        model_name = opt.model_name + \
            '/accu_{accu:3.3f}.chkpt'.format(accu=100*valid_loss)
        torch.save(checkpoint, model_name)
    elif opt.save_mode == 'best':
        model_name = opt.model_name + '/model.chkpt'
        try:
            if valid_loss >= min(valid_losses):
                torch.save(checkpoint, model_name)
                print('[Info] The checkpoint file has been updated.')
        except:
            pass


def save_acc_model(opt, epoch_i, model, valid_metrics, valid_metrics_all):
    model_state_dict = model.state_dict()
    checkpoint = {'model': model_state_dict, 'settings': opt, 'epoch': epoch_i}
    if opt.save_mode == 'all':
        prefix_path = opt.model_name + '/'+str(epoch_i)+'/'
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        model_name = prefix_path +\
            '/accu_{accu:3.3f}.chkpt'.format(accu=100*valid_metrics['Acc'])
        torch.save(checkpoint, model_name)
    elif opt.save_mode == 'best':
        prefix_path = opt.model_name + '/best_acc/'
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        model_name = prefix_path + 'model.chkpt'
        try:
            if valid_metrics['Acc'] >= max(valid_metrics_all['Acc']):
                torch.save(checkpoint, model_name)
                print('[Info] The checkpoint file has been updated.')
        except:
            pass

def save_f1_model(opt, epoch_i, model, valid_metrics, valid_metrics_all):
    model_state_dict = model.state_dict()
    checkpoint = {'model': model_state_dict, 'settings': opt, 'epoch': epoch_i}
    if opt.save_mode == 'all':
        prefix_path = opt.model_name + '/'+str(epoch_i)+'/'
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        model_name = prefix_path + '/microf1_{accu:3.3f}.chkpt'.format(accu=100 *
                                                                       valid_metrics['micro_f1'])
        torch.save(checkpoint, model_name)
    elif opt.save_mode == 'best':
        prefix_path = opt.model_name + '/best_microf1/'
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        model_name = prefix_path+'model.chkpt'
        try:
            if valid_metrics['micro_f1'] >= max(valid_metrics_all['micro_f1']):
                torch.save(checkpoint, model_name)
                print('[Info] The checkpoint file has been updated.')
        except:
            pass
