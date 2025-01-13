import numpy as np
import scipy.sparse as sp
import logging
from six.moves import xrange
from collections import OrderedDict
import sys
import pdb
from sklearn import metrics
from threading import Lock
from threading import Thread
import torch
import math
from pdb import set_trace as stop
import os
import pandas as pd
# import pylab as pl
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
import json
from tqdm import tqdm
from pdb import set_trace as stop
from utils.load_data import get_test_data

def compute_metrics(all_predTgtDict, loss, emo2idx, cause2idx, opt, elapsed, all_metrics=True, verbose=True):
    if opt.task =='CF':
        emo_targets = all_predTgtDict['emo_tgt']
        cos_preds = all_predTgtDict['cos_pred'] # (N,seq_len,sent_len)
        cos_targets = all_predTgtDict['cos_tgt']
       
        emo_targets_mask = [[1 if emo>0 else 0 for emo in emo_seq] for emo_seq in emo_targets]

        cos_target_mask = [[[1 if cos>0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_targets]
        cos_pred_mask = [[[1 if cos>0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_preds]
        pred_cos_nums =sum([sum([sum(cos_seq) for cos_seq in cos_pred]) for cos_pred in cos_pred_mask])
        tgt_cos_nums = sum([sum([sum(cos_seq) for cos_seq in cos_tgt]) for cos_tgt in cos_target_mask])
        correct_cos_nums = sum([sum([sum([1 if cos==cos_pred and cos==1 else 0 for cos,cos_pred in zip(cos_seq,cos_pred_seq)]) for cos_seq,cos_pred_seq in zip(cos_tgt,cos_pred)]) for cos_tgt,cos_pred in zip(cos_target_mask,cos_pred_mask)])


        # # only need to compute cos metrics
        # cos_mask = [[[1 if cos>=0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_targets]
        # filtered_cos_mask = [[[ emo*cos for cos in cos_seq ] for emo,cos_seq in zip(emo_tgt,cos_pred)] for emo_tgt,cos_pred in zip(emo_targets_mask,cos_mask)]
        # cos_used_preds = [[[cos if cos_mask>0 else 0 for cos,cos_mask in zip(cos_seq,cos_mask_seq)] for cos_seq,cos_mask_seq in zip(cos_pred,cos_mask)] for cos_pred,cos_mask in zip(cos_preds,filtered_cos_mask)]
        # cos_used_targets = [[[cos if cos_mask>0 else 0 for cos,cos_mask in zip(cos_seq,cos_mask_seq)] for cos_seq,cos_mask_seq in zip(cos_tgt,cos_mask_seq)] for cos_tgt,cos_mask_seq in zip(cos_targets,filtered_cos_mask)]
        # # calculate
        # pred_cos_nums =sum([sum([sum(cos_seq) for cos_seq in cos_pred]) for cos_pred in cos_used_preds])
        # tgt_cos_nums = sum([sum([sum(cos_seq) for cos_seq in cos_tgt]) for cos_tgt in cos_used_targets])
        # correct_cos_nums = sum([sum([sum([1 if cos==cos_pred and cos==1 else 0 for cos,cos_pred in zip(cos_seq,cos_pred_seq)]) for cos_seq,cos_pred_seq in zip(cos_tgt,cos_pred)]) for cos_tgt,cos_pred in zip(cos_used_targets,cos_used_preds)])
        p = correct_cos_nums/(pred_cos_nums+1e-8)
        r = correct_cos_nums/(tgt_cos_nums+1e-8)
        f1 = 2*p*r/(p+r+1e-8)
      
        print(f'pred_cos_nums: {pred_cos_nums}, tgt_cos_nums: {tgt_cos_nums}, correct_cos_nums: {correct_cos_nums}')
        return {'micro_f1':f1,'micro_R':r,'micro_P':p}
      
    elif opt.task =='ECPF':
        emo_preds = all_predTgtDict['emo_pred']
        emo_targets = all_predTgtDict['emo_tgt']
        cos_targets = all_predTgtDict['cos_tgt']
        cos_preds = all_predTgtDict['cos_pred']
        
        # compute pair metrics only hasEmotion + cos
        emo_targets_mask = [[1 if emo>0 else 0 for emo in emo_seq] for emo_seq in emo_targets]
        emo_preds_mask = [[1 if emo>0 else 0 for emo in emo_seq] for emo_seq in emo_preds]

        # first calculate emo metrics
        pred_emo_nums =sum([sum([emo for emo in emo_seq]) for emo_seq in emo_preds_mask])
        tgt_emo_nums = sum([sum([emo for emo in emo_seq]) for emo_seq in emo_targets_mask])
        correct_emo_nums = sum([sum([1 if pred==tgt and pred!=emo2idx['中性'] else 0 for pred,tgt in zip(emo_pred,emo_tgt)]) for emo_pred,emo_tgt in zip(emo_preds,emo_targets)])
        emo_p = correct_emo_nums/(pred_emo_nums+1e-8)
        emo_r = correct_emo_nums/(tgt_emo_nums+1e-8)
        emo_f1 = 2*emo_p*emo_r/(emo_p+emo_r+1e-8)
     
        # then calculate the cos pair metrics
        
        cos_target_mask = [[[1 if cos>0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_targets]
        cos_pred_mask = [[[1 if cos>0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_preds]
        pred_cos_nums =sum([sum([sum(cos_seq) for cos_seq in cos_pred]) for cos_pred in cos_pred_mask])
        tgt_cos_nums = sum([sum([sum(cos_seq) for cos_seq in cos_tgt]) for cos_tgt in cos_target_mask])
        correct_cos_nums = sum([sum([sum([1 if cos==cos_pred and cos==1 else 0 for cos,cos_pred in zip(cos_seq,cos_pred_seq)]) for cos_seq,cos_pred_seq in zip(cos_tgt,cos_pred)]) for cos_tgt,cos_pred in zip(cos_target_mask,cos_pred_mask)])


        # cos_target_mask = [[[1 if cos>=0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_targets]
        # cos_pred_mask = [[[1 if cos>=0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_preds]
        # filtered_pred_emo_cos_mask = [[[ emo*cos for cos in cos_seq ] for emo,cos_seq in zip(emo_tgt,cos_pred)] for emo_tgt,cos_pred in zip(emo_preds_mask,cos_pred_mask)]
        # filtered_tgt_emo_cos_mask = [[[ emo*cos for cos in cos_seq ] for emo,cos_seq in zip(emo_tgt,cos_pred)] for emo_tgt,cos_pred in zip(emo_targets_mask,cos_target_mask)]
        # cos_used_preds = [[[cos if cos_mask>0 else 0 for cos,cos_mask in zip(cos_seq,cos_mask_seq)] for cos_seq,cos_mask_seq in zip(cos_preds,cos_masks)] for cos_preds,cos_masks in zip(cos_preds,filtered_pred_emo_cos_mask)]
        # cos_used_targets = [[[cos if cos_mask>0 else 0 for cos,cos_mask in zip(cos_seq,cos_mask_seq)] for cos_seq,cos_mask_seq in zip(cos_tgts,cos_masks)] for cos_tgts,cos_masks in zip(cos_targets,filtered_tgt_emo_cos_mask)]
        
        # pred_cos_nums =sum([sum([sum(cos_seq) for cos_seq in cos_pred]) for cos_pred in cos_used_preds])
        # tgt_cos_nums = sum([sum([sum(cos_seq) for cos_seq in cos_tgt]) for cos_tgt in cos_used_targets])
        # correct_cos_nums = sum([sum([sum([1 if cos==cos_pred and cos==1 else 0 for cos,cos_pred in zip(cos_seq,cos_pred_seq)]) for cos_seq,cos_pred_seq in zip(cos_tgt,cos_pred)]) for cos_tgt,cos_pred in zip(cos_used_targets,cos_used_preds)])
    
        p = correct_cos_nums/(pred_cos_nums+1e-8)
        r = correct_cos_nums/(tgt_cos_nums+1e-8)
        f1 = 2*p*r/(p+r+1e-8)
        print(f'pred_emo_nums: {pred_emo_nums}, tgt_emo_nums: {tgt_emo_nums}, correct_emo_nums: {correct_emo_nums}')
        print(f'pred_cos_nums: {pred_cos_nums}, tgt_cos_nums: {tgt_cos_nums}, correct_cos_nums: {correct_cos_nums}')
        return {'micro_f1':f1,'micro_R':r,'micro_P':p,'micro_emo_f1':emo_f1,'micro_emo_R':emo_r,'micro_emo_P':emo_p}
    
    elif opt.task =='ECPF-C':
        emo_preds = all_predTgtDict['emo_pred']
        emo_targets = all_predTgtDict['emo_tgt']
        cos_targets = all_predTgtDict['cos_tgt']
        cos_preds = all_predTgtDict['cos_pred']
        
        # compute pair metrics only hasEmotion + cos
        emo_targets_mask = [[1 if emo>0 else 0 for emo in emo_seq] for emo_seq in emo_targets]
        emo_preds_mask = [[1 if emo>0 else 0 for emo in emo_seq] for emo_seq in emo_preds]

        # first calculate emo metrics
        pred_emo_nums =sum([sum([emo for emo in emo_seq]) for emo_seq in emo_preds_mask])
        tgt_emo_nums = sum([sum([emo for emo in emo_seq]) for emo_seq in emo_targets_mask])
        correct_emo_nums = sum([sum([1 if pred==tgt and pred!=emo2idx['中性'] else 0 for pred,tgt in zip(emo_pred,emo_tgt)]) for emo_pred,emo_tgt in zip(emo_preds,emo_targets)])
        emo_p = correct_emo_nums/(pred_emo_nums+1e-8)
        emo_r = correct_emo_nums/(tgt_emo_nums+1e-8)
        emo_f1 = 2*emo_p*emo_r/(emo_p+emo_r+1e-8)


        # then calculate the cos pair metrics
        cos_target_mask = [[[1 if cos>0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_targets]
        cos_pred_mask = [[[1 if cos>0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_preds]
        pred_cos_nums =sum([sum([sum(cos_seq) for cos_seq in cos_pred]) for cos_pred in cos_pred_mask])
        tgt_cos_nums = sum([sum([sum(cos_seq) for cos_seq in cos_tgt]) for cos_tgt in cos_target_mask])
        correct_cos_nums = sum([sum([sum([1 if cos==cos_pred and cos==1 and epred==etgt and epred!=emo2idx['中性'] else 0 for cos,cos_pred in zip(cos_seq,cos_pred_seq)]) for cos_seq,cos_pred_seq, epred,etgt in zip(cos_tgt,cos_pred,emo_pred,emo_tgt)]) for cos_tgt,cos_pred,emo_pred,emo_tgt in zip(cos_target_mask,cos_pred_mask,emo_preds,emo_targets)])
        
        # cos_mask = [[[1 if cos>=0 else 0  for cos in cos_seq] for cos_seq in cos_tgt] for cos_tgt in cos_targets]
        # filtered_pred_emo_cos_mask = [[[ emo*cos for cos in cos_seq ] for emo,cos_seq in zip(emo_tgt,cos_pred)] for emo_tgt,cos_pred in zip(emo_preds_mask,cos_mask)]
        # filtered_tgt_emo_cos_mask = [[[ emo*cos for cos in cos_seq ] for emo,cos_seq in zip(emo_tgt,cos_pred)] for emo_tgt,cos_pred in zip(emo_targets_mask,cos_mask)]
        # cos_used_preds = [[[cos if cos_mask>0 else 0 for cos,cos_mask in zip(cos_seq,cos_mask_seq)] for cos_seq,cos_mask_seq in zip(cos_preds,cos_masks)] for cos_preds,cos_masks in zip(cos_preds,filtered_pred_emo_cos_mask)]
        # cos_used_targets = [[[cos if cos_mask>0 else 0 for cos,cos_mask in zip(cos_seq,cos_mask_seq)] for cos_seq,cos_mask_seq in zip(cos_tgts,cos_masks)] for cos_tgts,cos_masks in zip(cos_targets,filtered_tgt_emo_cos_mask)]
        
        # pred_cos_nums =sum([sum([sum(cos_seq) for cos_seq in cos_pred]) for cos_pred in cos_used_preds])
        # tgt_cos_nums = sum([sum([sum(cos_seq) for cos_seq in cos_tgt]) for cos_tgt in cos_used_targets])
        # correct_cos_nums = sum([sum([sum([1 if cos==cos_pred and cos==1 else 0 for cos,cos_pred in zip(cos_seq,cos_pred_seq)]) for cos_seq,cos_pred_seq in zip(cos_tgt,cos_pred)]) for cos_tgt,cos_pred in zip(cos_used_targets,cos_used_preds)])
        p = correct_cos_nums/(pred_cos_nums+1e-8)
        r = correct_cos_nums/(tgt_cos_nums+1e-8)
        f1 = 2*p*r/(p+r+1e-8)
        print(f'pred_cos_nums: {pred_cos_nums}, tgt_cos_nums: {tgt_cos_nums}, correct_cos_nums: {correct_cos_nums}')
        return {'micro_f1':f1,'micro_R':r,'micro_P':p,'micro_emo_f1':emo_f1,'micro_emo_R':emo_r,'micro_emo_P':emo_p}
    



def make_test_samples(all_predTgtDict, opt, test_data, emo2idx, cos2idx):
    def get_cos_pred_num(cp):
        sinal_1 = False
        num=0
        for c in cp:
            if c>0 and not sinal_1:
                sinal_1 = True
                num+=1
            if sinal_1 and c>0:
                num+=1
            elif sinal_1 and c<=0:
                break

        return num
    def save_json_data(data, tgt_path):
        with open(tgt_path, "w",encoding='utf8') as fp:
            fp.write(json.dumps(data, indent=4, ensure_ascii=False))
    
                
    idx2emo= {idx:emo for emo,idx in emo2idx.items()}
    emo_pred= all_predTgtDict['emo_pred']
    emo_tgt= all_predTgtDict['emo_tgt']
    cos_pred= all_predTgtDict['cos_pred']
    cos_tgt= all_predTgtDict['cos_tgt']
    dialog_list = get_test_data(opt)
    for d_ep,d_et,d_cp,d_ct,dialog in zip(emo_pred,emo_tgt,cos_pred,cos_tgt,dialog_list):
       for ep,et,cp,ct,turn in zip(d_ep,d_et,d_cp,d_ct,dialog):
            UttId=int(turn['UttId'])
            
            ep_=idx2emo[ep]
            turn['Emotion_pre'] = ep_
            if ep_!='中性':
                cp_num=get_cos_pred_num(cp)
                turn['Consequence_pre'] = str(cp_num+UttId)

    save_json_data(dialog_list, opt.model_name+'/test_samples.json')

    

              
    