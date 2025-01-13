import argparse
import math
import time
import os

import copy
import numpy as np
import os.path as path
import utils.evals as evals
import utils.logging as logging
import utils.util as util

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as stop
from tqdm import tqdm
from train import train_epoch
from test import test_epoch
import warnings
warnings.filterwarnings("ignore")


def run_model(model, train_data, valid_data, test_data, emo2idx, cos2dix, crition_foremo,crition_forcos, optimizer, adv_optimizer, opt):
    logger = logging.Logger(opt)

    valid_losses = []

    train_metrics_all = {'micro_f1': [],
                         'micro_R': [], 'micro_P': [], 'micro_emo_f1':[],'micro_emo_R':[],'micro_emo_P':[]}
    valid_metrics_all = { 'micro_f1': [],
                         'micro_R': [], 'micro_P': [],'micro_emo_f1':[],'micro_emo_R':[],'micro_emo_P':[]}
    test_metrics_all = { 'micro_f1': [],
                        'micro_R': [], 'micro_P': [],'micro_emo_f1':[],'micro_emo_R':[],'micro_emo_P':[]}

    losses = []
 
    if opt.test_only:
        # test_data = valid_data
        start = time.time()
        
        all_predTgtDict, test_loss = test_epoch(
            model, test_data, emo2idx, cos2dix, opt, crition_foremo,crition_forcos, '(Testing)')
        elapsed = ((time.time()-start)/60)
        print('(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
        test_loss = test_loss/len(test_data)
        print('B : '+str(test_loss)+'\n')

        evals.make_test_samples(all_predTgtDict,opt,test_data,emo2idx,cos2dix)

        test_metrics = evals.compute_metrics(
            all_predTgtDict, 0, emo2idx, cos2dix, opt, elapsed, all_metrics=True)
        # logger.test_print(test_data,test_metrics,all_predTgtDict,emo2idx,cos2dix)
        print(test_metrics)
        return
    

    loss_file = open(path.join(opt.model_name, 'losses.txt'), 'w')
    loss_file.write(f'epoch,train,valid,test,lr,lrb')
    if not os.path.exists(opt.model_name+'/epochs/'):
        os.makedirs(opt.model_name+'/epochs/')
    for epoch_i in range(opt.epoch):
        print('================= Epoch', epoch_i+1, '=================')
      
        ################################## TRAIN ###################################
        start = time.time()
        all_predTgtDict, train_loss = train_epoch(
            model, train_data, emo2idx, cos2dix, crition_foremo,crition_forcos, optimizer, adv_optimizer, (epoch_i+1), opt)
       
        elapsed = ((time.time()-start)/60)
        print('(Training) elapse: {elapse:3.3f} min '.format(
            elapse=elapsed))
        print('train dia avg loss : '+str(train_loss)+'\n')
        
        train_metrics = evals.compute_metrics(
            all_predTgtDict, 0, emo2idx, cos2dix, opt, elapsed, all_metrics=True)
       
        for item_key in train_metrics:
            if item_key in train_metrics_all.keys():
                train_metrics_all[item_key].append(train_metrics[item_key])
       
        ################################### VALID ###################################
        start = time.time()
        with torch.no_grad():
            all_predTgtDict, valid_loss = test_epoch(
                model, valid_data, emo2idx, cos2dix, opt, crition_foremo,crition_forcos, '(Validation)')
        elapsed = ((time.time()-start)/60)
        print('(Validation) elapse: {elapse:3.3f} min'.format(
            elapse=elapsed))
        print('valid dia avg loss : '+str(valid_loss)+'\n')

        valid_metrics = evals.compute_metrics(
            all_predTgtDict, 0, emo2idx, cos2dix, opt, elapsed, all_metrics=True)
        valid_losses += [valid_loss]
        for item_key in valid_metrics:
            if item_key in valid_metrics_all.keys():
                valid_metrics_all[item_key].append(valid_metrics[item_key])
        ################################## TEST ###################################
        start = time.time()
        with torch.no_grad():
            all_predTgtDict, test_loss = test_epoch(
                model, test_data,  emo2idx, cos2dix, opt, crition_foremo,crition_forcos, '(Testing)')
        elapsed = ((time.time()-start)/60)
        print('(Testing) elapse: {elapse:3.3f} min'.format(
            elapse=elapsed))
       
        print('test dia avg loss '+str(test_loss)+'\n')
       

        test_metrics = evals.compute_metrics(
            all_predTgtDict, 0, emo2idx, cos2dix, opt, elapsed, all_metrics=True)
        for item_key in test_metrics.keys():
            if item_key in test_metrics_all.keys():
                test_metrics_all[item_key].append(test_metrics[item_key])

        logger.print_metrics_all(epoch=epoch_i+1, train_metrics=train_metrics, valid_metrics=valid_metrics, test_metrics=test_metrics,
                                 train_metrics_all=train_metrics_all, valid_metrics_all=valid_metrics_all, test_metrics_all=test_metrics_all)
        losses.append([epoch_i+1, train_loss, valid_loss, test_loss])
        if not 'test' in opt.model_name and not opt.test_only:
            if opt.save_loss_best:
                util.save_loss_model(
                    opt, epoch_i+1, model, valid_loss, valid_losses)
            if opt.save_f1_best:
                util.save_f1_model(opt, epoch_i+1, model,
                                    valid_metrics, valid_metrics_all)
            # best acc and F1
     
        loss_file.write(str(int(epoch_i+1)))
        loss_file.write(','+str(train_loss))
        loss_file.write(','+str(valid_loss))
        loss_file.write(','+str(test_loss))
        loss_file.write(','+str(optimizer.optimizer_list[0].param_groups[0]['lr']))
        loss_file.write(','+str(optimizer.optimizer_list[0].param_groups[1]['lr']))
        loss_file.write('\n')
    logger.print_metrics_best(train_metrics_all=train_metrics_all,
                              valid_metrics_all=valid_metrics_all, test_metrics_all=test_metrics_all)
