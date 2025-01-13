from pandas import value_counts
import torch.multiprocessing
from PIL import Image
# pip3 install pillow
from cgi import test
import enum
import json
import random
import numpy as np
import torch
import torch.utils.data as torch_data
import math
from pdb import set_trace as stop
import utils
from os import path
import os
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import joblib


def get_test_data(opt):
    text_data_root= os.path.join(opt.dataset_basepath,'split_videos_scripts/coarse')
    scripts_files = os.listdir(text_data_root)
    test_scripts_files =['S1E71_S1E75', 'S1E51_S1E55']
    scripts_files = [(folder,file) for folder in test_scripts_files for file in os.listdir(os.path.join(text_data_root,folder))]
    dialog_list = []
    for src in scripts_files:
        folder,file = src
        dialog_json= json.load(open(os.path.join(text_data_root,folder,file),'r',encoding='utf-8'))
        dialog_list.append(dialog_json)
    return dialog_list

def process_data(opt):
   
    text_data_root= os.path.join(opt.dataset_basepath,'split_videos_scripts/coarse')
    scripts_files = os.listdir(text_data_root)
    # scripts_files.remove('S1E86_S1E90')
    # scripts_files.remove('S1E81_S1E85')
    # scripts_files.remove('S1E91_S1E95')

    
    train_scripts_files = scripts_files[:int(len(scripts_files)*0.8)]
    valid_scripts_files = scripts_files[int(len(scripts_files)*0.8):int(len(scripts_files)*0.9)]
    test_scripts_files = scripts_files[int(len(scripts_files)*0.9):]
    train_scripts_files=['S1E21_S1E25', 'S1E31_S1E35', 'S1E36_S1E40', 'S1E01_S1E05', 'S1E56_S1E60', 'S1E41_S1E45', 'S1E66_S1E70', 'S1E81_S1E85', 'S1E76_S1E80', 'S1E61_S1E65', 'S1E11_S1E15', 'S1E86_S1E90', 'S1E46_S1E50', 'S1E06_S1E10', 'S1E96_S1E100', 'S1E26_S1E30']
    valid_scripts_files = ['S1E91_S1E95', 'S1E16_S1E20']
    test_scripts_files =['S1E71_S1E75', 'S1E51_S1E55']
 

    # audio video root
    if opt.audio_extract == 'opensmile':
        audio_feature_root = os.path.join(opt.dataset_basepath,'audio_features/target')
    elif opt.audio_extract == 'hubert':
        audio_feature_root = os.path.join(opt.dataset_basepath,'audio_features/source_hubert')
    else:
        raise ValueError("audio_extract error")
    video_feature_root = os.path.join(opt.dataset_basepath,'video_features/target')

    if opt.have_describe:
        video_description_root = os.path.join(opt.dataset_basepath,'video_features/Captions_13B_mingpt4')
    else:
        video_description_root = None

 
    
    
    # Neutral Happy Depressed Angry Superised Sad Fear 
    if opt.emotion_class == 'all':
        emo2idx = {'中性': 0, '快乐': 1, '厌恶': 2,
                   '愤怒': 3, '惊讶': 4, '悲伤': 5, '恐惧': 6}
    elif opt.emotion_class == 'NPN':
        emo2idx = {'中性': 0, '消极': 1, '积极': 2}
    elif opt.emotion_class == 'EN':
        emo2idx = {'中性': 0, '情绪': 1}
    else:
        raise ValueError("emo2idx error")
  
    cos2idx={'结果':1,'其他':0}
    train_data_custom = CustomDataset(
        src_dias=train_scripts_files,
        text_data_root=text_data_root,
        video_feature_root=video_feature_root,
        audio_feature_root=audio_feature_root,
        video_description_root=video_description_root,
        emo2idx=emo2idx,
        cos2idx=cos2idx,
        shuffle=True,
        cuda=opt.cuda,
        opt=opt,
        name='train'

    )
    valid_data_custom = CustomDataset(
        src_dias=valid_scripts_files,
        text_data_root=text_data_root,
        video_feature_root=video_feature_root,
        audio_feature_root=audio_feature_root,
        video_description_root=video_description_root,
        emo2idx=emo2idx,
        cos2idx=cos2idx,
        shuffle=False,
        cuda=opt.cuda,
        opt=opt,
        name='valid'

    )
    test_data_custom = CustomDataset(
        src_dias=test_scripts_files,
        text_data_root=text_data_root,
        video_feature_root=video_feature_root,
        audio_feature_root=audio_feature_root,
        video_description_root=video_description_root,
        emo2idx=emo2idx,
        cos2idx=cos2idx,
        shuffle=False,
        cuda=opt.cuda,
        opt=opt,
        name='test'

    )

    # [i for i in train_data_custom]
    # [i for i in valid_data_custom]
    # [i for i in test_data_custom]
    
    train_data_custom.get_small_size(opt.small_size)
   
    # print(f'the size of train dataset is {len(train_data_custom)}\n\n')
    train_data = torch_data.DataLoader(
        train_data_custom, batch_size=opt.BATCH_SIZE, num_workers=4, shuffle=True,collate_fn=collate_align)
    test_data = torch_data.DataLoader(test_data_custom , batch_size=opt.EVAL_BATCH_SIZE, num_workers=4, shuffle=False,collate_fn=collate_align)
    valid_data = torch_data.DataLoader(
        valid_data_custom , batch_size=opt.EVAL_BATCH_SIZE, num_workers=4, shuffle=False,collate_fn=collate_align)
    # print(len([i for i in train_data])) 
    # train_data.__iter__()
    
    print(len(train_data_custom))
    print(len(valid_data_custom))
    print(len(test_data_custom))
    max_cos_length = max(train_data_custom.max_cos_length,valid_data_custom.max_cos_length,test_data_custom.max_cos_length)
    
    print(f'max_cos_length:{max_cos_length}') # max_cos_length:31
    print(f'emo_nums:{train_data_custom.emo_nums},{valid_data_custom.emo_nums},{test_data_custom.emo_nums}') # emo_nums: 0
    print(f'cos_nums:{train_data_custom.cos_nums},{valid_data_custom.cos_nums},{test_data_custom.cos_nums}') # cos_nums: 0
  
    return train_data, valid_data, test_data, emo2idx, cos2idx


def collate_align(data):
    # print(type(data),len(data))
    data.sort(key=lambda x: len(x['Emotion_input_ids']),reverse=True)

   
    max_seq_len = len(data[0]['Emotion_input_ids'])
    word_lens=[len(d) for da in data for d in da['input_ids']]
    max_word_len = max(word_lens)
    # max_word_len=min(max_word_len,100)
    spk_lens = [len(d) for da in data for d in da['Speaker_input_ids']]
    max_spk_len = max(spk_lens)
    # max_spk_len=min(max_spk_len,8)
    max_spk_and_word_len = max([a+b for a,b in zip(spk_lens,word_lens)])
    max_spk_and_word_len = max_spk_and_word_len+3 # two ids for [CLS] and [SEP]

    max_describe_len = max([len(d) for da in data for d in da['describe_input_ids']])
    max_emo_clue_len = max([len(d) for da in data for d in da['emo_clue_input_ids']])
    max_cos_clue_why_len = max([len(d) for da in data for d in da['cos_clue_why_input_ids']])
    max_cos_clue_impact_len = max([len(d) for da in data for d in da['cos_clue_impact_input_ids']])
  
    # print(max_spk_len,max_word_len,max_seq_len,max_spk_and_word_len)
    sep_token_id = 102
    cls_token_id = 101
    pad_Speaker_input_ids,pad_Speaker_input_attention_mask,pad_input_ids,\
    pad_input_attention_mask,pad_describe_input_ids,pad_describe_attention_masks,\
    pad_describe_token_type_ids, \
    pad_emo_clue_input_ids,pad_emo_clue_attention_mask,pad_emo_clue_token_type_ids,\
    pad_cos_clue_why_input_ids,pad_cos_clue_why_attention_mask,pad_cos_clue_why_token_type_ids,\
    pad_cos_clue_impact_input_ids,pad_cos_clue_impact_attention_mask,pad_cos_clue_impact_token_type_ids,\
    pad_Emotion_input_ids,\
    pad_Emotion_input_attention_mask,pad_Cosequence_input_ids,\
    pad_Cosequence_input_attention_mask,pad_video_feature,\
    pad_video_feature_attention_mask,pad_audio_feature,\
    pad_audio_feature_ateention_mask,pad_spk_and_input_ids,\
    pad_spk_and_input_ids_attention_mask,pad_spk_and_input_token_type_ids = [[] for i in range(27)]
    
    for i_dict in data:
        Speaker_input_ids,Speaker_input_attention_mask,input_ids,\
            input_attention_mask,describe_input_ids,describe_attention_masks,\
                describe_token_type_ids,\
                    emo_clue_input_ids,emo_clue_attention_mask,emo_clue_token_type_ids,\
                        cos_clue_why_input_ids,cos_clue_why_attention_mask,cos_clue_why_token_type_ids,\
                            cos_clue_impact_input_ids,cos_clue_impact_attention_mask,cos_clue_impact_token_type_ids,\
                                Emotion_input_ids,\
                                Emotion_input_attention_mask,Cosequence_input_ids,\
                                    Cosequence_input_attention_mask,video_feature,\
                                        video_feature_attention_mask,audio_feature,\
                                            audio_feature_attention_mask = [i_dict[key] for key in i_dict.keys()]
        
        pad_dia_Speaker_input_ids = [ids+[0]*(max_spk_len-len(ids)) for ids in Speaker_input_ids]+ [[0]*max_spk_len]*(max_seq_len-len(Speaker_input_ids))   
        pad_dia_Speaker_input_attention_mask = [mask+[0]*(max_spk_len-len(mask)) for mask in Speaker_input_attention_mask]+ [[0]*max_spk_len]*(max_seq_len-len(Speaker_input_attention_mask))
        pad_dia_input_ids = [ids+[0]*(max_word_len-len(ids)) for ids in input_ids]+ [[0]*max_word_len]*(max_seq_len-len(input_ids))
        pad_dia_input_attention_mask = [mask+[0]*(max_word_len-len(mask)) for mask in input_attention_mask]+ [[0]*max_word_len]*(max_seq_len-len(input_attention_mask))
        
        pad_dia_describe_input_ids = [ids+[0]*(max_describe_len-len(ids)) for ids in describe_input_ids]+ [[0]*max_describe_len]*(max_seq_len-len(describe_input_ids))
        pad_dia_describe_attention_masks = [mask+[0]*(max_describe_len-len(mask)) for mask in describe_attention_masks]+ [[0]*max_describe_len]*(max_seq_len-len(describe_attention_masks))
        pad_dia_describe_token_type_ids = [token_type_ids+[0]*(max_describe_len-len(token_type_ids)) for token_type_ids in describe_token_type_ids]+ [[0]*max_describe_len]*(max_seq_len-len(describe_token_type_ids))
        
        pad_dia_emo_clue_input_ids = [ids+[0]*(max_emo_clue_len-len(ids)) for ids in emo_clue_input_ids]+ [[0]*max_emo_clue_len]*(max_seq_len-len(emo_clue_input_ids))
        pad_dia_emo_clue_attention_mask = [mask+[0]*(max_emo_clue_len-len(mask)) for mask in emo_clue_attention_mask]+ [[0]*max_emo_clue_len]*(max_seq_len-len(emo_clue_attention_mask))
        pad_dia_emo_clue_token_type_ids = [token_type_ids+[0]*(max_emo_clue_len-len(token_type_ids)) for token_type_ids in emo_clue_token_type_ids]+ [[0]*max_emo_clue_len]*(max_seq_len-len(emo_clue_token_type_ids))
        pad_dia_cos_clue_why_input_ids = [ids+[0]*(max_cos_clue_why_len-len(ids)) for ids in cos_clue_why_input_ids]+ [[0]*max_cos_clue_why_len]*(max_seq_len-len(cos_clue_why_input_ids))
        pad_dia_cos_clue_why_attention_mask = [mask+[0]*(max_cos_clue_why_len-len(mask)) for mask in cos_clue_why_attention_mask]+ [[0]*max_cos_clue_why_len]*(max_seq_len-len(cos_clue_why_attention_mask))
        pad_dia_cos_clue_why_token_type_ids = [token_type_ids+[0]*(max_cos_clue_why_len-len(token_type_ids)) for token_type_ids in cos_clue_why_token_type_ids]+ [[0]*max_cos_clue_why_len]*(max_seq_len-len(cos_clue_why_token_type_ids))
        pad_dia_cos_clue_impact_input_ids = [ids+[0]*(max_cos_clue_impact_len-len(ids)) for ids in cos_clue_impact_input_ids]+ [[0]*max_cos_clue_impact_len]*(max_seq_len-len(cos_clue_impact_input_ids))
        pad_dia_cos_clue_impact_attention_mask = [mask+[0]*(max_cos_clue_impact_len-len(mask)) for mask in cos_clue_impact_attention_mask]+ [[0]*max_cos_clue_impact_len]*(max_seq_len-len(cos_clue_impact_attention_mask))
        pad_dia_cos_clue_impact_token_type_ids = [token_type_ids+[0]*(max_cos_clue_impact_len-len(token_type_ids)) for token_type_ids in cos_clue_impact_token_type_ids]+ [[0]*max_cos_clue_impact_len]*(max_seq_len-len(cos_clue_impact_token_type_ids))

        
        

        pad_dia_Emotion_input_ids = Emotion_input_ids+[0]*(max_seq_len-len(Emotion_input_ids))
        pad_dia_Emotion_input_attention_mask = Emotion_input_attention_mask+[0]*(max_seq_len-len(Emotion_input_attention_mask))
        pad_dia_Cosequence_input_ids = [ids+[0]*(max_seq_len-len(ids)) for ids in Cosequence_input_ids]+ [[0]*max_seq_len]*(max_seq_len-len(Cosequence_input_ids))
        pad_dia_Cosequence_input_attention_mask = [[0]*(i+1)+mask[i+1:]+[0]*(max_seq_len-len(mask)) for i,mask in enumerate(Cosequence_input_attention_mask)]+ [[0]*max_seq_len]*(max_seq_len-len(Cosequence_input_attention_mask))
        pad_dia_video_feature = video_feature+[np.zeros(video_feature[0].shape)]*(max_seq_len-len(video_feature))
        pad_dia_video_feature_attention_mask = video_feature_attention_mask+[0]*(max_seq_len-len(video_feature_attention_mask))
        pad_dia_audio_feature = audio_feature+[np.zeros(audio_feature[0].shape)]*(max_seq_len-len(audio_feature))
        pad_dia_audio_feature_attention_mask = audio_feature_attention_mask+[0]*(max_seq_len-len(audio_feature_attention_mask))
        
        spk_and_input_ids = [[cls_token_id]+a+[sep_token_id]+b+[sep_token_id] for a,b in zip(Speaker_input_ids,input_ids)]
        spk_and_input_ids_attention_mask = [[1]*len(ids) for ids in spk_and_input_ids]
        spk_and_input_token_type_ids =  [[0]*len(ids) for ids in spk_and_input_ids]
        pad_dia_spk_and_input_ids = [ids+[0]*(max_spk_and_word_len-len(ids)) for ids in spk_and_input_ids]+ [[0]*max_spk_and_word_len]*(max_seq_len-len(spk_and_input_ids))
        
        pad_dia_spk_and_input_ids_attention_mask = [mask+[0]*(max_spk_and_word_len-len(mask)) for mask in spk_and_input_ids_attention_mask]+ [[0]*max_spk_and_word_len]*(max_seq_len-len(spk_and_input_ids_attention_mask))
        pad_dia_spk_and_input_token_type_ids = [token_type_ids+[0]*(max_spk_and_word_len-len(token_type_ids)) for token_type_ids in spk_and_input_token_type_ids]+ [[0]*max_spk_and_word_len]*(max_seq_len-len(spk_and_input_token_type_ids))

        pad_Speaker_input_ids.append(pad_dia_Speaker_input_ids)
        pad_Speaker_input_attention_mask.append(pad_dia_Speaker_input_attention_mask)
        pad_input_ids.append(pad_dia_input_ids)
        pad_input_attention_mask.append(pad_dia_input_attention_mask)
        pad_describe_input_ids.append(pad_dia_describe_input_ids)
        pad_describe_attention_masks.append(pad_dia_describe_attention_masks)
        pad_describe_token_type_ids.append(pad_dia_describe_token_type_ids)

        pad_emo_clue_input_ids.append(pad_dia_emo_clue_input_ids)
        pad_emo_clue_attention_mask.append(pad_dia_emo_clue_attention_mask)
        pad_emo_clue_token_type_ids.append(pad_dia_emo_clue_token_type_ids)
        pad_cos_clue_why_input_ids.append(pad_dia_cos_clue_why_input_ids)
        pad_cos_clue_why_attention_mask.append(pad_dia_cos_clue_why_attention_mask)
        pad_cos_clue_why_token_type_ids.append(pad_dia_cos_clue_why_token_type_ids)
        pad_cos_clue_impact_input_ids.append(pad_dia_cos_clue_impact_input_ids)
        pad_cos_clue_impact_attention_mask.append(pad_dia_cos_clue_impact_attention_mask)
        pad_cos_clue_impact_token_type_ids.append(pad_dia_cos_clue_impact_token_type_ids)
        
        pad_Emotion_input_ids.append(pad_dia_Emotion_input_ids)
        pad_Emotion_input_attention_mask.append(pad_dia_Emotion_input_attention_mask)
        pad_Cosequence_input_ids.append(pad_dia_Cosequence_input_ids)
        pad_Cosequence_input_attention_mask.append(pad_dia_Cosequence_input_attention_mask)
        pad_video_feature.append(pad_dia_video_feature)
        pad_video_feature_attention_mask.append(pad_dia_video_feature_attention_mask)
        pad_audio_feature.append(pad_dia_audio_feature)
        pad_audio_feature_ateention_mask.append(pad_dia_audio_feature_attention_mask)

        pad_spk_and_input_ids.append(pad_dia_spk_and_input_ids)
        pad_spk_and_input_ids_attention_mask.append(pad_dia_spk_and_input_ids_attention_mask)
        pad_spk_and_input_token_type_ids.append(pad_dia_spk_and_input_token_type_ids)

 
   
    

    # print(pad_Speaker_input_ids)
    pad_Speaker_input_ids = torch.LongTensor(pad_Speaker_input_ids)
    pad_Speaker_input_attention_mask = torch.LongTensor(pad_Speaker_input_attention_mask)
    pad_input_ids = torch.LongTensor(pad_input_ids)
    pad_input_attention_mask = torch.LongTensor(pad_input_attention_mask)
    pad_describe_input_ids = torch.LongTensor(pad_describe_input_ids)
    pad_describe_attention_masks = torch.LongTensor(pad_describe_attention_masks)
    pad_describe_token_type_ids = torch.LongTensor(pad_describe_token_type_ids)

    pad_emo_clue_input_ids = torch.LongTensor(pad_emo_clue_input_ids)
    pad_emo_clue_attention_mask = torch.LongTensor(pad_emo_clue_attention_mask)
    pad_emo_clue_token_type_ids = torch.LongTensor(pad_emo_clue_token_type_ids)
    pad_cos_clue_why_input_ids = torch.LongTensor(pad_cos_clue_why_input_ids)
    pad_cos_clue_why_attention_mask = torch.LongTensor(pad_cos_clue_why_attention_mask)
    pad_cos_clue_why_token_type_ids = torch.LongTensor(pad_cos_clue_why_token_type_ids)
    pad_cos_clue_impact_input_ids = torch.LongTensor(pad_cos_clue_impact_input_ids)
    pad_cos_clue_impact_attention_mask = torch.LongTensor(pad_cos_clue_impact_attention_mask)
    pad_cos_clue_impact_token_type_ids = torch.LongTensor(pad_cos_clue_impact_token_type_ids)

    pad_Emotion_input_ids = torch.LongTensor(pad_Emotion_input_ids)
    pad_Emotion_input_attention_mask = torch.LongTensor(pad_Emotion_input_attention_mask)
    pad_Cosequence_input_ids = torch.LongTensor(pad_Cosequence_input_ids)
    pad_Cosequence_input_attention_mask = torch.LongTensor(pad_Cosequence_input_attention_mask)
    pad_video_feature = torch.FloatTensor(pad_video_feature).squeeze(-2)
    pad_video_feature_attention_mask = torch.LongTensor(pad_video_feature_attention_mask)
    pad_audio_feature = torch.FloatTensor(pad_audio_feature)
    pad_audio_feature_ateention_mask = torch.LongTensor(pad_audio_feature_ateention_mask)

    # speaker and input ids 
    pad_spk_and_input_ids = torch.LongTensor(pad_spk_and_input_ids)
    pad_spk_and_input_ids_attention_mask = torch.LongTensor(pad_spk_and_input_ids_attention_mask)
    pad_spk_and_input_token_type_ids = torch.LongTensor(pad_spk_and_input_token_type_ids)
  


    return_dict = {
        'Speaker_input_ids':pad_Speaker_input_ids,
        'Speaker_input_attention_mask':pad_Speaker_input_attention_mask,
        'input_ids':pad_input_ids,
        'input_attention_mask':pad_input_attention_mask,
        'describe_input_ids':pad_describe_input_ids,
        'describe_attention_masks':pad_describe_attention_masks,
        'describe_token_type_ids':pad_describe_token_type_ids,
        'emo_clue_input_ids':pad_emo_clue_input_ids,
        'emo_clue_attention_mask':pad_emo_clue_attention_mask,
        'emo_clue_token_type_ids':pad_emo_clue_token_type_ids,
        'cos_clue_why_input_ids':pad_cos_clue_why_input_ids,
        'cos_clue_why_attention_mask':pad_cos_clue_why_attention_mask,
        'cos_clue_why_token_type_ids':pad_cos_clue_why_token_type_ids,
        'cos_clue_impact_input_ids':pad_cos_clue_impact_input_ids,
        'cos_clue_impact_attention_mask':pad_cos_clue_impact_attention_mask,
        'cos_clue_impact_token_type_ids':pad_cos_clue_impact_token_type_ids,
        'Emotion_input_ids':pad_Emotion_input_ids,
        'Emotion_input_attention_mask':pad_Emotion_input_attention_mask,
        'Cosequence_input_ids':pad_Cosequence_input_ids,
        'Cosequence_input_attention_mask':pad_Cosequence_input_attention_mask,
        'video_feature':pad_video_feature,
        'video_feature_attention_mask':pad_video_feature_attention_mask,
        'audio_feature':pad_audio_feature,
        'audio_feature_attention_mask':pad_audio_feature_ateention_mask,
        'spk_and_input_ids':pad_spk_and_input_ids,
        'spk_and_input_ids_attention_mask':pad_spk_and_input_ids_attention_mask,
        'spk_and_input_token_type_ids':pad_spk_and_input_token_type_ids
    }

    return return_dict
     


class CustomDataset(torch_data.Dataset):
    def __init__(self, src_dias,text_data_root,video_feature_root,audio_feature_root,video_description_root,emo2idx, cos2idx, shuffle, cuda, opt,name):
        super(CustomDataset, self).__init__()
        # src_dias = src_dias[:5]
        self.src_dias = src_dias
        self.text_data_root = text_data_root
        self.video_feature_root = video_feature_root
        self.audio_feature_root = audio_feature_root
        self.video_description_root = video_description_root
        self.emo2idx = emo2idx
        self.cos2idx = cos2idx
        self.cuda = cuda
        self.opt = opt
        self._need_shuffle = shuffle
      
        dialog_nums = [len(os.listdir(os.path.join(text_data_root,dia_name))) for dia_name in self.src_dias]
        self._n_batch = sum(dialog_nums)
        self.max_cos_length = 0
        self.emo_nums=0
        self.cos_nums=0

        self.name = name
        emotion_clues_ = json.load(open(os.path.join(opt.clue_path_root,f'emo_clues_{name}.json'),'r',encoding='utf-8'))
        consequence_clues_ =json.load(open(os.path.join(opt.clue_path_root,f'cos_clues_wAc_{name}.json'),'r',encoding='utf-8'))
        self.emotion_clues = {item['emo_key']:item for item in emotion_clues_}
        self.consequence_clues = {item['cos_key']:item for item in consequence_clues_}
       


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.opt.bert_path)
        self.describe_tokenizer = AutoTokenizer.from_pretrained(
            self.opt.decribe_bert_path)
        self.scripts_files = [(folder,file) for folder in self.src_dias for file in os.listdir(os.path.join(text_data_root,folder))]
        
        self.test_seq_len(self.scripts_files)

       
        dialog_file = self.scripts_files[0]
        dialog_tensor_dict = self.get_preUtt(dialog_file)
    def get_small_size(self,small_size):
        if small_size>0:
            self.scripts_files = self.scripts_files[:small_size]
            self._n_batch = len(self.scripts_files)
        
        
    def test_seq_len(self,scripts_files):
        for item in scripts_files:
            jsonname=item[1]
 
            _,beg,end = jsonname.replace('.json','').split('_') 
            if int(end)-int(beg)>50:
                print(item)
    def __len__(self):
        
        return self._n_batch
    def pre_emotion2dix(self,emotion):
        emotion = emotion.strip()
        if emotion in ['快乐', '厌恶', '愤怒', '惊讶', '悲伤', '恐惧']:
            self.emo_nums+=1
        if self.opt.emotion_class == 'all':
            if emotion in self.emo2idx.keys():
                return self.emo2idx[emotion]
            else:
                print(emotion,'not in emo2idx load_data.py pre_emotion2dix')
                
                emotion = '中性'
                return self.emo2idx[emotion]
        elif self.opt.emotion_class == 'NPN':
            if emotion in ['快乐']:
                return self.emo2idx['积极']
            elif emotion in ['厌恶','愤怒','悲伤','恐惧','惊讶']:
                return self.emo2idx['消极']
            else:
                return self.emo2idx['中性']
        elif self.opt.emotion_class == 'EN':
            if emotion in ['快乐','厌恶','愤怒','悲伤','恐惧','惊讶']:
                return self.emo2idx['情绪']
            else:
                return self.emo2idx['中性']
        else:
            raise ValueError("emo2idx error")
        
    def pre_cos2dix(self,whole_UttId,cos,dialog_json):
        cos_list = [0]*len(dialog_json)
        first_UttId = int(dialog_json[0]['UttId'])
       
        if cos.strip().isdigit():
            cos = int(cos)
            for i in range(whole_UttId-first_UttId+1,cos-first_UttId+1):
                try:
                    cos_list[i] = self.cos2idx['结果']
                except:
                    pass
            
        self.cos_nums+=sum(cos_list)
        return cos_list
    
    def get_preUtt(self,dialog_file):
        folder,file = dialog_file
        dialog_json= json.load(open(os.path.join(self.text_data_root,folder,file),'r',encoding='utf-8'))
        Speaker_input_ids=[]
        Speaker_input_attention_mask=[]
        input_ids=[]
        input_attention_mask=[]
        describe_input_ids=[]
        describe_attention_masks=[]
        describe_token_type_ids=[]
        emo_clue_input_ids=[]
        emo_clue_attention_mask=[]
        emo_clue_token_type_ids=[]
        cos_clue_why_input_ids=[]
        cos_clue_why_attention_mask=[]
        cos_clue_why_token_type_ids=[]
        cos_clue_impact_input_ids=[]
        cos_clue_impact_attention_mask=[]
        cos_clue_impact_token_type_ids=[]
        Emotion_input_ids=[]
        Emotion_input_attention_mask=[]
        Cosequence_input_ids=[]
        Cosequence_input_attention_mask=[]
        video_feature=[]
        video_feature_attention_mask=[]
        audio_feature=[]
        audio_feature_attention_mask=[]

        for i,item in enumerate(dialog_json):
            Speaker = item['Speaker']
            Speaker = Speaker.strip().replace('#','')
            Utterance = item['Utterance']
            Emotion = item['Emotion']
            whole_UttId = int(item['UttId'])
            current_UttId= i+1
            Cosequence = item['Cosequence']
            utt_Speaker_input_ids = self.tokenizer.encode(Speaker,add_special_tokens=False)
            utt_Speaker_input_attention_mask = [1]*len(utt_Speaker_input_ids)
            Utt_input_ids = self.tokenizer.encode(Utterance,add_special_tokens=False)
            Utt_input_attention_mask = [1]*len(Utt_input_ids)
            utt_Emotion_input_ids = self.pre_emotion2dix(Emotion)
            utt_Cosequence_input_ids = self.pre_cos2dix(whole_UttId,Cosequence,dialog_json)
            utt_Cosequence_input_attention_mask = [1]*len(utt_Cosequence_input_ids)

            # video feature
            video_feature_path = os.path.join(self.video_feature_root,folder,file.split('_')[0]+f'_{whole_UttId}_m_fea.joblib')
            utt_video_feature = joblib.load(video_feature_path)

            # audio feature
            if self.opt.audio_extract == 'opensmile':
                audio_feature_path = os.path.join(self.audio_feature_root,folder,file.split('_')[0]+f'_{whole_UttId}_fea.joblib')
                utt_audio_feature = joblib.load(audio_feature_path)
            elif self.opt.audio_extract == 'hubert':
                audio_feature_path = os.path.join(self.audio_feature_root,folder,file.split('_')[0]+f'_{whole_UttId}.joblib')
                try:
                    utt_audio_feature = joblib.load(audio_feature_path)
                    utt_audio_feature= utt_audio_feature.transpose(0,1).squeeze(0)
                except:
                    print(audio_feature_path)
                
                # if utt_audio_feature.
                # print(utt_audio_feature.)
                # assert 0
           
            emo_key = f'{folder}#{file[:-5]}#{current_UttId}#emo'
            cos_key = f'{folder}#{file[:-5]}#{current_UttId}#cos'
            emo_item =self.emotion_clues[emo_key]
            cos_item = self.consequence_clues[cos_key]
            emo_clue = emo_item['emo_answer'][:150]
         
            cos_clue_why = cos_item['cos_answer_why'][:150]
            cos_clue_impact = cos_item['cos_answer_impact'][:150]
            emo_clue_inputs = self.tokenizer(emo_clue,max_length=510)
            cos_clue_why_inputs = self.tokenizer(cos_clue_why,max_length=510)
            cos_clue_impact_inputs = self.tokenizer(cos_clue_impact,max_length=510)

            utt_emo_clue_input_ids = emo_clue_inputs['input_ids']
            utt_emo_clue_attention_mask = emo_clue_inputs['attention_mask']
            utt_emo_clue_token_type_ids = emo_clue_inputs['token_type_ids']
            utt_cos_clue_why_input_ids = cos_clue_why_inputs['input_ids']
            utt_cos_clue_why_attention_mask = cos_clue_why_inputs['attention_mask']
            utt_cos_clue_why_token_type_ids = cos_clue_why_inputs['token_type_ids']
            utt_cos_clue_impact_input_ids = cos_clue_impact_inputs['input_ids']
            utt_cos_clue_impact_attention_mask = cos_clue_impact_inputs['attention_mask']
            utt_cos_clue_impact_token_type_ids = cos_clue_impact_inputs['token_type_ids']

            emo_clue_input_ids.append(utt_emo_clue_input_ids)
            emo_clue_attention_mask.append(utt_emo_clue_attention_mask)
            emo_clue_token_type_ids.append(utt_emo_clue_token_type_ids)
            cos_clue_why_input_ids.append(utt_cos_clue_why_input_ids)
            cos_clue_why_attention_mask.append(utt_cos_clue_why_attention_mask)
            cos_clue_why_token_type_ids.append(utt_cos_clue_why_token_type_ids)
            cos_clue_impact_input_ids.append(utt_cos_clue_impact_input_ids)
            cos_clue_impact_attention_mask.append(utt_cos_clue_impact_attention_mask)
            cos_clue_impact_token_type_ids.append(utt_cos_clue_impact_token_type_ids)




            # video decription
            if self.video_description_root is not None:
                utt_description_path = os.path.join(self.video_description_root,folder,file.split('_')[0]+f'_{whole_UttId}_m.json')
                utt_description = json.load(open(utt_description_path,'r',encoding='utf-8'))
                utt_description=utt_description[:100]
                utt_describe_inputs = self.describe_tokenizer(utt_description,max_length=510)

                utt_describe_input_ids = utt_describe_inputs['input_ids']
                utt_describe_attention_mask = utt_describe_inputs['attention_mask']
                utt_describe_token_type_ids = utt_describe_inputs['token_type_ids']
            else:
                utt_describe_input_ids=None
                utt_describe_attention_mask=None
                utt_describe_token_type_ids=None

            # clues
            
            # emo_clue_key=f'{sea_folder}#{dialogue_file[:-5]}#{current_UttId}'

                

            Speaker_input_ids.append(utt_Speaker_input_ids)
            Speaker_input_attention_mask.append(utt_Speaker_input_attention_mask)
            input_ids.append(Utt_input_ids)
            input_attention_mask.append(Utt_input_attention_mask)
            describe_input_ids.append(utt_describe_input_ids)
            describe_attention_masks.append(utt_describe_attention_mask)
            describe_token_type_ids.append(utt_describe_token_type_ids)
            Emotion_input_ids.append(utt_Emotion_input_ids)
            Cosequence_input_ids.append(utt_Cosequence_input_ids)
            Cosequence_input_attention_mask.append(utt_Cosequence_input_attention_mask)
            video_feature.append(utt_video_feature)
            audio_feature.append(utt_audio_feature)




        Emotion_input_attention_mask = [1]*len(Emotion_input_ids)
        video_feature_attention_mask = [1]*len(video_feature)
        audio_feature_attention_mask = [1]*len(audio_feature)

        max_cos_len = max([sum(cos) for cos in Cosequence_input_ids])
        self.max_cos_length = max(self.max_cos_length,max_cos_len)
       
        input_dict = {
            'Speaker_input_ids':Speaker_input_ids,
            'Speaker_input_attention_mask':Speaker_input_attention_mask,
            'input_ids':input_ids,
            'input_attention_mask':input_attention_mask,
            'describe_input_ids':describe_input_ids,
            'describe_attention_masks':describe_attention_masks,
            'describe_token_type_ids':describe_token_type_ids,
            'emo_clue_input_ids':emo_clue_input_ids,
            'emo_clue_attention_mask':emo_clue_attention_mask,
            'emo_clue_token_type_ids':emo_clue_token_type_ids,
            'cos_clue_why_input_ids':cos_clue_why_input_ids,
            'cos_clue_why_attention_mask':cos_clue_why_attention_mask,
            'cos_clue_why_token_type_ids':cos_clue_why_token_type_ids,
            'cos_clue_impact_input_ids':cos_clue_impact_input_ids,
            'cos_clue_impact_attention_mask':cos_clue_impact_attention_mask,
            'cos_clue_impact_token_type_ids':cos_clue_impact_token_type_ids,
            'Emotion_input_ids':Emotion_input_ids,
            'Emotion_input_attention_mask':Emotion_input_attention_mask,
            'Cosequence_input_ids':Cosequence_input_ids,
            'Cosequence_input_attention_mask':Cosequence_input_attention_mask,
            'video_feature':video_feature,
            'video_feature_attention_mask':video_feature_attention_mask,
            'audio_feature':audio_feature,
            'audio_feature_attention_mask':audio_feature_attention_mask
        }
       
        return input_dict


    def __getitem__(self, index):
        dialog_file = self.scripts_files[index]
        dialog_tensor_dict = self.get_preUtt(dialog_file)

       
        return dialog_tensor_dict
    
