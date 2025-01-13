import random
from pdb import set_trace as stop
from typing import ValuesView
import numpy as np
import pandas as pd
import json
import joblib
import os
import logging
import pickle
import argparse
import torch
from sklearn.metrics import classification_report

from utils.load_data import process_data
from configs.config_args import config_args, get_args
import utils.util as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from runner import run_model
from models.model import EmoCos
from models.STGraph import STGraph
from models.LSmodel import LSmodel
from models.Roberta import myRoberta
from models.MECPE import MECPE
import warnings
warnings.filterwarnings("ignore")
from utils.optim import get_optim_for_emocas
from pdb import set_trace as stop
from runner import run_model

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)  # 所有设别设置随机种子
torch.cuda.manual_seed_all(opt.seed)  # 设置所有GPU的随机种子,如果没有GPU，会默认忽视

def main(opt):
   
    train_data, valid_data, test_data, emo2idx, cos2idx = process_data(
        opt)
    
    if opt.select_model =='LSmodel':
        model=LSmodel(
            opt=opt,
            emo2idx=emo2idx,
            cos2idx=cos2idx,
        )
    elif opt.select_model == 'EmoCos':
        model = EmoCos(
            opt=opt,
            emo2idx=emo2idx,
            cos2idx=cos2idx,
        )
    elif opt.select_model == 'STGraph':
        model = STGraph(
            opt=opt,
            emo2idx=emo2idx,
            cos2idx=cos2idx,
        )
    elif opt.select_model == 'Roberta':
        model = myRoberta(
            opt=opt,
            emo2idx=emo2idx,
            cos2idx=cos2idx,
        )
    elif opt.select_model == 'MECPE':
        model = MECPE(
            opt=opt,
            emo2idx=emo2idx,
            cos2idx=cos2idx,
        )
    elif opt.select_model == 'LLMprompting':
        # maybe we can get some clues from llm to help the model. like image decription, logic reasoning, analysis, etc.
        pass
    else:
        raise ValueError('Please select a model')

    opt.total_num_parameters = int(utils.count_parameters(model))
    
    
    if opt.test_only:
        model.load_state_dict(torch.load(opt.model_name+'/best_microf1/model.chkpt')['model'])

    # torch.optim.Adam(
    #     model.parameters(), betas=(0.9, 0.98), lr=opt.lr)
    # freeze_layers = ['layer.0', 'layer.1.', 'layer.2', 'layer.3',
    #                  'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8']
    # for name, param in model.named_parameters():
    #     if "SpeakerEncoder" in name:
    #         param.requires_grad = False
    #     for free in freeze_layers:
    #         if free in name:
    #             param.requires_grad = False
    #             print(name)
    optimizer= get_optim_for_emocas(opt, model)
    # scheduler = torch.torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay, last_epoch=-1)
    crition_foremo,crition_forcos = utils.get_criterion(opt, emo2idx, cos2idx)
 
  
    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        # torch.distributed.init_process_group(backend="nccl")
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        # crition_foremo=nn.DataParallel(crition_foremo)
        # crition_forcos=nn.DataParallel(crition_forcos)
   
    if torch.cuda.is_available() and opt.cuda:

        # if opt.gpu_id != -1:
        #     # torch.cuda.set_device(opt.gpu_id)
        #     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
        
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)
        model = model.cuda()
        crition_foremo = crition_foremo.cuda()
        crition_forcos = crition_forcos.cuda()

    if opt.load_pretrained:

        checkpoint = torch.load(opt.model_name+'/best_microf1/model.chkpt')
        model.load_state_dict(checkpoint['model'])
    adv_optimizer = None
    try:
        run_model(model, train_data, valid_data, test_data, emo2idx, cos2idx, crition_foremo,crition_forcos,
                  optimizer, adv_optimizer, opt)
    except KeyboardInterrupt:
        print('-' * 89+'\nManual Exit')  # it is so good for print
        exit()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main(opt)
