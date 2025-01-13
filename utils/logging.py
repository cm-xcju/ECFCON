
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


class Logger:
    def __init__(self, opt):
        self.opt = opt
        self.model_path = opt.model_name

    def print_metrics_all(self, epoch, train_metrics, valid_metrics, test_metrics, train_metrics_all, valid_metrics_all, test_metrics_all):
        metrics_save = self.model_path+'/metrics.txt'
        with open(metrics_save, 'a+', encoding='utf-8') as fp:  # ensure_ascii=False
            fp.write('epoch:{}, train,valid,test'.format(epoch))
            fp.write(json.dumps(train_metrics, indent=4, ensure_ascii=False))
            fp.write(json.dumps(valid_metrics, indent=4, ensure_ascii=False))
            test_metrics_most = {}
            for key in test_metrics:
                if key != 'multi_class_f1':
                    test_metrics_most[key] = test_metrics[key]
            fp.write(json.dumps(test_metrics_most,
                     indent=4, ensure_ascii=False))
            # fp.write('multiple class:\n'+test_metrics['multi_class_f1'])
            fp.write('\n=*30\n')

        print('\n')
        print('**********************************epoch:{}*******************************'.format(epoch))
        if self.opt.task == 'CF':
           
            # print('valid ACC:  '+str(valid_metrics['Acc']))
            print('valid miF1: '+str(valid_metrics['micro_f1']))
            # print('valid maF1: '+str(valid_metrics['macro_f1']))
            # print('test ACC:  '+str(test_metrics['Acc']))
            print('test miF1: '+str(test_metrics['micro_f1']))
            # print('test maF1: '+str(test_metrics['macro_f1']))
        elif self.opt.task == 'ECPF':
            print('valid cospair miF1:'+str(valid_metrics['micro_f1'])+', emo miF1:'+str(valid_metrics['micro_emo_f1']))
            print('test cospair miF1: '+str(test_metrics['micro_f1'])+', emo miF1:'+str(test_metrics['micro_emo_f1']))
        print('***********************************************************************')

    def print_metrics_best(self, train_metrics_all, valid_metrics_all, test_metrics_all):
        metrics_save = self.model_path+'/metrics.txt'
        with open(metrics_save, 'a+', encoding='utf-8') as fp:
            # best_acc = max(valid_metrics_all['Acc'])

            # epoch_acc = valid_metrics_all['Acc'].index(best_acc)
            # fp.write('\n='*10+'epoch:{}, Acc valid best\n'.format(epoch_acc)+'='*10)

            best_mif1 = max(valid_metrics_all['micro_f1'])
            epoch_mif1 = valid_metrics_all['micro_f1'].index(best_mif1)+1
            fp.write(
                '='*10+'epoch:{}, micro_f1 valid best\n'.format(epoch_mif1)+'='*10)

            # best_maf1 = max(valid_metrics_all['macro_f1'])
            # epoch_maf1 = valid_metrics_all['macro_f1'].index(best_maf1)
            # fp.write(
            #     '='*10+'epoch:{}, micro_f1 valid best\n'.format(epoch_maf1)+'='*10)

    def test_print(self, test_data, test_metrics, all_predTgtDict, emo2idx, cause2idx):
        # what
        idx2emo = {id: key for key, id in emo2idx.items()}
        idx2cause = {id: key for key, id in cause2idx.items()}
        metrics_save = self.model_path+'/metrics_test.txt'
        with open(metrics_save, 'a+', encoding='utf-8') as fp:  # ensure_ascii=False
            fp.write('\n'+'='*10+'test-only'+'='*10+'\n')
            test_metrics_most = {}
            for key in test_metrics:
                if key != 'multi_class_f1':
                    test_metrics_most[key] = test_metrics[key]
            fp.write(json.dumps(test_metrics_most,
                     indent=4, ensure_ascii=False))
            fp.write('multiple class:\n'+test_metrics['multi_class_f1'])
            fp.write('\n=*30\n')
        
        testsample_save = self.model_path+'/test_pre_samples.json'
        raw_data = test_data.dataset.src_dias
        new_dia_dict = []
        # test
        allpreds_print = [ len(all_predTgtDict['emo_tgt'][i]) for i in range(5)] # [20, 15, 40, 30, 25]
        alltgts_print = [len(raw_data[i]['Utterances']) for i in range(5)] # [17, 12, 38, 27, 25]

        
        for i, raw in enumerate(raw_data):
            new_dia = raw.copy()
            emo_tgt = all_predTgtDict['emo_tgt'][i]
            cause_pred = all_predTgtDict['cause_pred'][i]
            cause_tgt = all_predTgtDict['cause_tgt'][i]
           
            for j, (utt, etgt,ctgt,cpre) in enumerate(zip(new_dia['Utterances'],emo_tgt, cause_tgt,cause_pred)):
                
                assert emo2idx[utt['emotion']] == etgt[0]
                assert cause2idx[utt['cause_pos']] == ctgt[0]
                new_dia['Utterances'][j]['cause_pred']=idx2cause[cpre[0]]
            
            new_dia_dict.append(new_dia)
      
        #save 
        with open(testsample_save, 'w',encoding='utf-8') as fp :
            fp.write(json.dumps(new_dia_dict,indent=4,ensure_ascii=False))
        
            