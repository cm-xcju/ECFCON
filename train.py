import argparse
import math
import time
import warnings
import copy
import numpy as np
import os.path as path
# import utils.evals as evals
import utils.util as util
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def train_epoch(model, train_data, emo2idx, cos2idx, crition_foremo,crition_forcos,  optimizer, adv_optimizer, epoch, opt):
    model.train()
    all_predictions = []
    all_targets = []
    loss_total = 0
    batch_idx=0
    all_predictions = []
    all_targets = []
    all_predTgtDict = {'emo_pred': [], 'emo_tgt': [],
                       'cos_pred': [], 'cos_tgt': []}
    for dia_item in tqdm(train_data, mininterval=0.5, desc='(Training)', leave=False):
       
        loss_item, dia_predTgtDict = train_batch_dias(
            model, emo2idx, cos2idx, crition_foremo,crition_forcos, optimizer, opt, epoch, dia_item)
        if not math.isnan(loss_item):
            batch_idx+=1
            loss_total += loss_item
        
        all_predTgtDict['emo_pred']+=dia_predTgtDict[0]
        all_predTgtDict['emo_tgt']+=dia_predTgtDict[1]
        all_predTgtDict['cos_pred']+=dia_predTgtDict[2]
        all_predTgtDict['cos_tgt']+=dia_predTgtDict[3]
        
   
    return all_predTgtDict, loss_total/(batch_idx+1e-8)
 

# split dia into mindatas
def train_batch_dias(model, emo2idx, cos2idx, crition_foremo,crition_forcos, optimizer, opt, epoch, dia_item):
    """due to our task is real-time, we need to split the dia"""
    start = time.time()
    
    accum_count = math.ceil(dia_item['Speaker_input_ids'].shape[0]/opt.MIN_BATCH_SIZE)


    loss_dia = 0
    loss_dia_n = 0
    emo_preds, emo_tgts, cos_preds, cos_tgts = [], [], [], []

    current_accum = 0

    optimizer.zero_grad()

    for i in range(accum_count):

    
        min_batch = {key:value[i*opt.MIN_BATCH_SIZE:(i+1)*opt.MIN_BATCH_SIZE] for key, value in dia_item.items()}
        # if min_batch['Speaker_input_ids'].shape[0] == 0:
        #     stop()
        loss, emo_pred,emo_tgt,cos_pred,cos_tgt = train_batch(
            model, crition_foremo,crition_forcos, opt, epoch, min_batch)
       
        if opt.multi_gpu:
            loss = loss.mean()
        if loss:
            loss.backward()
        current_accum += 1
    
        if loss:
            loss_dia += loss.item()
            loss_dia_n += 1

        emo_preds+=emo_pred
        emo_tgts+=emo_tgt
        cos_preds+=cos_pred
        cos_tgts+=cos_tgt
 
    dia_predTgtDict = [emo_preds, emo_tgts, cos_preds, cos_tgts]
    # if remain minbatches
    optimizer.step()
    optimizer.zero_grad()
    if loss_dia > 0:
        loss_dia = loss_dia/loss_dia_n
    elapsed = ((time.time()-start)/60)
    print('\n-*10(train one dia) elapse: {elapse:3.3f} min'.format(
        elapse=elapsed))

    return loss_dia, dia_predTgtDict  # pred_out_dia, tgt_out_dia


def train_batch(model, crition_foremo,crition_forcos, opt, epoch, minbatch):

   
    minbatch = {key:value.cuda() for key,value in minbatch.items()}
    # speaker_src, text_src, img_src, tgt, idxs = minbatch
 
    # speaker_src_, text_src_, img_src_, tgt_ = trans_src_to_cuda(
    #     speaker_src, text_src, img_src, tgt, opt)
  
    loss, emo_pred,emo_tgt,cos_pred,cos_tgt = model(**minbatch, crition_foremo=crition_foremo,crition_forcos=crition_forcos, opt=opt, epoch=epoch)
 
    # loss, PredTgtlist = model(speaker_src_, text_src_,
    #                           img_src_, tgt_, idxs, crit)

    emo_pred = emo_pred.detach().cpu().numpy().tolist()
    emo_tgt = emo_tgt.detach().cpu().numpy().tolist()
    cos_pred = cos_pred.detach().cpu().numpy().tolist()
    cos_tgt = cos_tgt.detach().cpu().numpy().tolist()
    return loss, emo_pred,emo_tgt,cos_pred,cos_tgt

def get_dia_minbatch(dia_item, start_idx, end_idx):
    dict_keys = dia_item.keys()
    min_batch = {}
    for k in dict_keys:
        min_batch[k] = dia_item[k][start_idx:end_idx+1]
    return min_batch


def trans_src_to_cuda(speaker_src, text_src, img_src, tgt, opt):
    if not opt.return_pt_data:
        speaker_src_cuda, text_src_cuda, tgt_cuda = {}, {}, {}
        for item_key in speaker_src.keys():

            if opt.cuda:
                speaker_src_cuda[item_key] = torch.Tensor(
                    speaker_src[item_key]).long().cuda()
            else:
                speaker_src_cuda[item_key] = torch.Tensor(
                    speaker_src[item_key]).long()
        for item_key in text_src.keys():

            if opt.cuda:
                text_src_cuda[item_key] = torch.Tensor(
                    text_src[item_key]).long().cuda()
            else:
                text_src_cuda[item_key] = torch.Tensor(
                    text_src[item_key]).long()

        if opt.cuda:
            img_src_cuda = torch.cat([item.unsqueeze(0)
                                      for item in img_src], 0).cuda()
        else:
            img_src_cuda = torch.cat([item.unsqueeze(0)
                                     for item in img_src], 0)
        for item_key in tgt.keys():

            if opt.cuda:
                tgt_cuda[item_key] = torch.Tensor(
                    tgt[item_key]).long().cuda()
            else:
                tgt_cuda[item_key] = torch.Tensor(
                    tgt[item_key]).long()

        return speaker_src_cuda, text_src_cuda, img_src_cuda, tgt_cuda
    elif opt.cuda:
       
        speaker_src = {ke: it.squeeze(0).cuda() for ke, it in speaker_src.items()}
        text_src = {ke: it.squeeze(0).cuda() for ke, it in text_src.items()}
        tgt = {ke: it.squeeze(0).cuda() for ke, it in tgt.items()}
        img_src = img_src.squeeze(0).cuda()
        return speaker_src, text_src, img_src, tgt
    else:
        return speaker_src, text_src, img_src, tgt
