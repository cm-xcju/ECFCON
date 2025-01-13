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


def test_epoch(model, test_data, emo2idx, cos2idx, opt,  crition_foremo,crition_forcos, description):
    model.eval()

    all_predictions = []
    all_targets = []
    loss_total = 0
    batch_idx=0
    all_predictions = []
    all_targets = []
    all_predTgtDict = {'emo_pred': [], 'emo_tgt': [],
                       'cos_pred': [], 'cos_tgt': []}
    for dia_item in tqdm(test_data, mininterval=0.5, desc='(testing)', leave=False):
      

        loss_item, dia_predTgtDict = test_batch_dias(
            model, emo2idx, cos2idx, opt,  crition_foremo,crition_forcos, dia_item)
        if not math.isnan(loss_item):
            batch_idx+=1
            loss_total += loss_item

        all_predTgtDict['emo_pred']+=dia_predTgtDict[0]
        all_predTgtDict['emo_tgt']+=dia_predTgtDict[1]
        all_predTgtDict['cos_pred']+=dia_predTgtDict[2]
        all_predTgtDict['cos_tgt']+=dia_predTgtDict[3]

  
    return all_predTgtDict, loss_total/(batch_idx+1e-8)


def test_batch_dias(model, emo2idx, cos2idx, opt, crition_foremo,crition_forcos, dia_item):  # split dia into mindatas
    """due to our task is real-time, we need to split the dia"""
    start = time.time()
 
   
    loss_dia = 0
   
    emo_preds, emo_tgts, cos_preds, cos_tgts = [], [], [], []
    min_batch = dia_item
    
    loss, emo_pred,emo_tgt,cos_pred,cos_tgt = test_batch(
        model, opt, crition_foremo,crition_forcos, min_batch)
    
    if loss:
        loss_dia += loss.item()
    emo_preds+=emo_pred
    emo_tgts+=emo_tgt
    cos_preds+=cos_pred
    cos_tgts+=cos_tgt


    

    elapsed = ((time.time()-start)/60)
    print('\n-*10(valid and test one dia) elapse: {elapse:3.3f} min'.format(
        elapse=elapsed))
    dia_predTgtDict = [emo_pred, emo_tgt, cos_pred, cos_tgt]
    return loss_dia, dia_predTgtDict


def test_batch(model, opt, crition_foremo,crition_forcos, minbatch):

    minbatch = {key:value.cuda() for key,value in minbatch.items()}

    with torch.no_grad():
        loss, emo_pred,emo_tgt,cos_pred,cos_tgt = model(**minbatch, crition_foremo=crition_foremo,crition_forcos=crition_forcos, opt=opt,epoch=None)
    
    emo_pred = emo_pred.detach().cpu().numpy().tolist()
    emo_tgt = emo_tgt.detach().cpu().numpy().tolist()
    cos_pred = cos_pred.detach().cpu().numpy().tolist()
    cos_tgt = cos_tgt.detach().cpu().numpy().tolist()
    return loss, emo_pred,emo_tgt,cos_pred,cos_tgt