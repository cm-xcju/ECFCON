import os
from re import L
from turtle import forward
import torch
from torch import nn
import numpy as np
import math
import torch.nn.init as init
from torch.nn import functional as F
from torch.autograd import Variable
from pdb import set_trace as stop
from transformers import BertModel,AutoModelForMaskedLM,AutoTokenizer
import warnings
warnings.filterwarnings("ignore")


class MECPE(nn.Module):
    def __init__(self, opt,emo2idx,cos2idx) -> None:
        super(MECPE,self).__init__()
        self.opt=opt
        self.emo2idx=emo2idx
        self.cos2idx=cos2idx
        self.bert = BertModel.from_pretrained(opt.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)
        self.hsize=500
        if opt.select_modality =='T':
            self.simple_fusion = nn.Sequential(
                nn.LayerNorm(opt.hidden_size),
                nn.Linear(opt.hidden_size, 2*opt.hidden_size),
                nn.Tanh(),
                nn.Linear(2*opt.hidden_size, opt.hidden_size),
            )
        elif opt.select_modality =='TA':
            self.simple_fusion = nn.Sequential(
                nn.LayerNorm(opt.hidden_size+opt.audio_size),
                nn.Linear(opt.hidden_size+opt.audio_size, 2*opt.hidden_size),
                nn.Tanh(),
                nn.Linear(2*opt.hidden_size, opt.hidden_size),
            )
        elif opt.select_modality =='TV':
            self.simple_fusion = nn.Sequential(
                nn.LayerNorm(opt.hidden_size+opt.video_size),
                nn.Linear(opt.hidden_size+opt.video_size, 2*opt.hidden_size),
                nn.Tanh(),
                nn.Linear(2*opt.hidden_size, opt.hidden_size),
            )
        elif opt.select_modality =='TAV' or opt.select_modality =='TAVC':
            self.simple_fusion = nn.Sequential(
                nn.LayerNorm(opt.hidden_size+opt.video_size+opt.audio_size),
                nn.Linear(opt.hidden_size+opt.video_size+opt.audio_size, 2*opt.hidden_size),
                nn.Tanh(),
                nn.Linear(2*opt.hidden_size, opt.hidden_size),
            )
        else:
            raise ValueError('Please select a modality')
        self.rnn = nn.LSTM(input_size=opt.hidden_size,hidden_size=opt.hidden_size,num_layers=1,batch_first=True,bidirectional=True)

        if self.opt.task == 'CF':
            self.emo_embedding = nn.Embedding(len(emo2idx), opt.hidden_size)
            if self.opt.select_modality !='TAVC':
                self.fea_emo = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size, opt.hidden_size),
                )
                self.fea_utt = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size, opt.hidden_size),
                )
                self.cos_MLP = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                    nn.ReLU(),
                    nn.Linear(opt.hidden_size, len(cos2idx)),
                )
            else:
                self.fea_emo = nn.Sequential(
                    nn.Linear(opt.hidden_size*4, opt.hidden_size*2),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                )
                self.fea_utt = nn.Sequential(
                    nn.Linear(opt.hidden_size*3, opt.hidden_size*2),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size*2, opt.hidden_size*1),
                )
                self.cos_MLP = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size*2),
                    nn.ReLU(),
                    nn.Linear(opt.hidden_size*2, len(cos2idx)),
                )
         
        if self.opt.task == 'ECPF' or self.opt.task == 'ECPF-C':
            if self.opt.select_modality !='TAVC':
                self.fea_emo = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size, opt.hidden_size),
                )
                self.fea_utt = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size, opt.hidden_size),
                )
                self.emo_MLP = nn.Sequential(
                    nn.Linear(opt.hidden_size, opt.hidden_size),
                    nn.ReLU(),
                    nn.Linear(opt.hidden_size, len(emo2idx)),
                )
                self.cos_MLP = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                    nn.ReLU(),
                    nn.Linear(opt.hidden_size, len(cos2idx)),
                )
            else:
                self.fea_emo = nn.Sequential(
                    nn.Linear(opt.hidden_size*3, opt.hidden_size*2),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                )
                self.fea_emo_impact = nn.Sequential(
                    nn.Linear(opt.hidden_size*3, opt.hidden_size*2),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                )
                self.fea_utt = nn.Sequential(
                    nn.Linear(opt.hidden_size*3, opt.hidden_size*2),
                    nn.Tanh(),
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                )
                self.emo_MLP = nn.Sequential(
                    nn.Linear(opt.hidden_size, opt.hidden_size),
                    nn.ReLU(),
                    nn.Linear(opt.hidden_size, len(emo2idx)),
                )
                self.cos_MLP = nn.Sequential(
                    nn.Linear(opt.hidden_size*2, opt.hidden_size),
                    nn.ReLU(),
                    nn.Linear(opt.hidden_size, len(cos2idx)),
                )

    def forward(self,Speaker_input_ids, Speaker_input_attention_mask, input_ids, input_attention_mask,\
                describe_input_ids, describe_attention_masks, describe_token_type_ids,\
                emo_clue_input_ids, emo_clue_attention_mask, emo_clue_token_type_ids,\
                cos_clue_why_input_ids, cos_clue_why_attention_mask, cos_clue_why_token_type_ids,\
                cos_clue_impact_input_ids, cos_clue_impact_attention_mask, cos_clue_impact_token_type_ids,\
                Emotion_input_ids, Emotion_input_attention_mask, Cosequence_input_ids,\
                Cosequence_input_attention_mask, video_feature, video_feature_attention_mask,\
                audio_feature, audio_feature_attention_mask,spk_and_input_ids,\
                spk_and_input_ids_attention_mask,spk_and_input_token_type_ids,\
                crition_foremo,crition_forcos, opt, epoch) -> None:
       
        pooler_output = self.get_pooler_output(spk_and_input_ids,spk_and_input_ids_attention_mask,spk_and_input_token_type_ids)

        if opt.select_modality =='TAVC':
            # for clues 
            if self.opt.task != 'CF':
                emo_clue_pooler_output= self.get_pooler_output(emo_clue_input_ids,emo_clue_attention_mask,emo_clue_token_type_ids)
            cos_clue_why_pooler_output= self.get_pooler_output(cos_clue_why_input_ids,cos_clue_why_attention_mask,cos_clue_why_token_type_ids)
            cos_clue_impact_pooler_output= self.get_pooler_output(cos_clue_impact_input_ids,cos_clue_impact_attention_mask,cos_clue_impact_token_type_ids)
            # for clues end

        batch_size,seq_len,_ =spk_and_input_ids.shape
        # text+video+audio
        mm_feature_cat = self.get_modality_cat(pooler_output,video_feature,audio_feature)
      
        mm_feature = self.simple_fusion(mm_feature_cat)
        lstm_mm_feature,*_ = self.rnn(mm_feature)

        

        if self.opt.task == 'CF':
            emo_emb = self.emo_embedding(Emotion_input_ids)
            # for emo emo_clues, to CF no emo clues neeed
            if opt.select_modality !='TAVC':
               
                emo_feature = self.fea_emo(lstm_mm_feature)
                utt_feature = self.fea_utt(lstm_mm_feature)
            else:
                emo_feature = self.fea_emo(torch.cat((lstm_mm_feature,emo_emb,cos_clue_impact_pooler_output),dim=2))
                utt_feature = self.fea_utt(torch.cat((lstm_mm_feature,cos_clue_why_pooler_output),dim=2))

            

            # for cos cos_clues
          
            emo_feature_repeated = emo_feature.unsqueeze(2).repeat(1,1,seq_len,1)
            utt_feature_repeated = utt_feature.unsqueeze(1).repeat(1,seq_len,1,1)
            emo_used_mask = (torch.gt(Emotion_input_ids,0)*1).unsqueeze(2)
            emo_used_mask_reapeated = emo_used_mask.repeat(1,1,seq_len)
            cos_feature = torch.cat((emo_feature_repeated,utt_feature_repeated),dim=3)
            cos_score = self.cos_MLP(cos_feature)
            pred_cos = cos_score.argmax(dim=3)
            mask = Cosequence_input_attention_mask*emo_used_mask_reapeated
            Cosequence_input_ids[mask==0]=-1
            dense_cos_score = cos_score.reshape(-1,cos_score.shape[-1])
            dense_Cosequence_input_ids=Cosequence_input_ids.reshape(-1)
           
            loss = crition_forcos(dense_cos_score,dense_Cosequence_input_ids)
            
            ones = torch.ones_like(pred_cos)
            cos_used_mask4 = torch.triu(ones,diagonal=1)
            mask4 = emo_used_mask_reapeated*cos_used_mask4
            pred_cos[mask4==0]=-1
           
            return loss,Emotion_input_ids,Emotion_input_ids,pred_cos,Cosequence_input_ids





        
        if self.opt.task =='ECPF' or self.opt.task =='ECPF-C':
            # very similar
            if opt.select_modality !='TAVC':
                # emo_feature = self.fea_emo(lstm_mm_feature)
                utt_feature = self.fea_utt(lstm_mm_feature)
                emo_feature =utt_feature
            else:
                emo_feature = self.fea_emo(torch.cat((lstm_mm_feature,emo_clue_pooler_output),dim=2))
                emo_feature_impact = self.fea_emo_impact(torch.cat((lstm_mm_feature,cos_clue_impact_pooler_output),dim=2))
                utt_feature = self.fea_utt(torch.cat((lstm_mm_feature,cos_clue_why_pooler_output),dim=2))

            
            emo_scores = self.emo_MLP(emo_feature)
            mask1 = Emotion_input_attention_mask
            Emotion_input_ids[mask1==0]=-1
            dense_emo_scores = emo_scores.reshape(-1,emo_scores.shape[-1])
            dense_Emotion_input_ids=Emotion_input_ids.reshape(-1)
            loss_emo = crition_foremo(dense_emo_scores,dense_Emotion_input_ids)
            pred_emos = emo_scores.argmax(dim=2)

            pred_emos[mask1==0]=0
          
            # for cos
            if opt.select_modality !='TAVC':
                emo_feature_repeated = emo_feature.unsqueeze(2).repeat(1,1,seq_len,1)
            else:
                emo_feature_repeated = emo_feature_impact.unsqueeze(2).repeat(1,1,seq_len,1)
            utt_feature_repeated = utt_feature.unsqueeze(1).repeat(1,seq_len,1,1)
            emo_used_mask = (torch.gt(pred_emos,0)*1).unsqueeze(2)
            emo_used_mask_reapeated = emo_used_mask.repeat(1,1,seq_len)
            cos_feature = torch.cat((emo_feature_repeated,utt_feature_repeated),dim=3)
            cos_score = self.cos_MLP(cos_feature)
            pred_cos = cos_score.argmax(dim=3)
     
            mask2 = Cosequence_input_attention_mask*emo_used_mask_reapeated
            cos_mask_input_ids = Cosequence_input_ids.clone()
            cos_mask_input_ids[mask2==0]=-1
            dense_cos_score = cos_score.reshape(-1,cos_score.shape[-1])
            dense_Cosequence_input_ids=cos_mask_input_ids.reshape(-1)
           
            loss_cos = crition_forcos(dense_cos_score,dense_Cosequence_input_ids)
            loss = (loss_emo+loss_cos)/2

            emo_used_mask3 = (torch.gt(Emotion_input_ids,0)*1).unsqueeze(2)
            emo_used_mask3_reapeated = emo_used_mask3.repeat(1,1,seq_len)
            mask3 = Cosequence_input_attention_mask*emo_used_mask3_reapeated
            Cosequence_input_ids[mask3==0]=-1

           
            ones = torch.ones_like(pred_cos)
            cos_used_mask4 = torch.triu(ones,diagonal=1)
            mask4 = emo_used_mask_reapeated*cos_used_mask4
            pred_cos[mask4==0]=-1
            
            # if math.isnan(loss):
            #     stop()
            return loss,pred_emos,Emotion_input_ids,pred_cos,Cosequence_input_ids
        


    def get_pooler_output(self,input_ids,attention_mask,token_type_ids):
        batch_size,seq_len,_ =input_ids.shape
        dense_input_ids = input_ids.reshape(-1,input_ids.shape[-1])
        dense_attention_mask = attention_mask.reshape(-1,attention_mask.shape[-1])
        dense_token_type_ids = token_type_ids.reshape(-1,token_type_ids.shape[-1])
        outputs = self.bert(input_ids=dense_input_ids,token_type_ids=dense_token_type_ids,attention_mask=dense_attention_mask)
        pooler_output=outputs.pooler_output
     
        pooler_output = pooler_output.reshape(batch_size,seq_len,-1)
        return pooler_output
    def  get_modality_cat(self,pooler_output,video_feature,audio_feature):
        if self.opt.select_modality =='T':
            mm_feature_cat = pooler_output
        elif self.opt.select_modality =='TA':
            mm_feature_cat = torch.cat((pooler_output,audio_feature),dim=2)
        elif self.opt.select_modality =='TV':
            mm_feature_cat = torch.cat((pooler_output,video_feature),dim=2)
        elif self.opt.select_modality in ['TAV','TAVC']:
            mm_feature_cat = torch.cat((pooler_output,video_feature,audio_feature),dim=2)
        else:
            raise ValueError('Please select a modality')
        return mm_feature_cat
        



# import os
# from re import L
# from turtle import forward
# import torch
# from torch import nn
# import numpy as np
# import math
# import torch.nn.init as init
# from torch.nn import functional as F
# from torch.autograd import Variable
# from pdb import set_trace as stop
# from transformers import BertModel,AutoModelForMaskedLM,AutoTokenizer
# import warnings
# warnings.filterwarnings("ignore")


# class MECPE(nn.Module):
#     def __init__(self, opt,emo2idx,cos2idx) -> None:
#         super(MECPE,self).__init__()
#         self.opt=opt
#         self.emo2idx=emo2idx
#         self.cos2idx=cos2idx
#         self.bert = BertModel.from_pretrained(opt.bert_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)
#         if opt.select_modality =='T':
#             self.simple_fusion = nn.Sequential(
#                 nn.LayerNorm(opt.hidden_size),
#                 nn.Linear(opt.hidden_size, 2*opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(2*opt.hidden_size, opt.hidden_size),
#             )
#         elif opt.select_modality =='TA':
#             self.simple_fusion = nn.Sequential(
#                 nn.LayerNorm(opt.hidden_size+opt.audio_size),
#                 nn.Linear(opt.hidden_size+opt.audio_size, 2*opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(2*opt.hidden_size, opt.hidden_size),
#             )
#         elif opt.select_modality =='TV':
#             self.simple_fusion = nn.Sequential(
#                 nn.LayerNorm(opt.hidden_size+opt.video_size),
#                 nn.Linear(opt.hidden_size+opt.video_size, 2*opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(2*opt.hidden_size, opt.hidden_size),
#             )
#         elif opt.select_modality =='TAV':
#             self.simple_fusion = nn.Sequential(
#                 nn.LayerNorm(opt.hidden_size+opt.video_size+opt.audio_size),
#                 nn.Linear(opt.hidden_size+opt.video_size+opt.audio_size, 2*opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(2*opt.hidden_size, opt.hidden_size),
#             )
#         else:
#             raise ValueError('Please select a modality')
#         self.rnn = nn.LSTM(input_size=opt.hidden_size,hidden_size=opt.hidden_size,num_layers=1,batch_first=True,bidirectional=True)

#         if self.opt.task == 'CF':
#             self.emo_embedding = nn.Embedding(len(emo2idx), opt.hidden_size)
#             self.fea_emo = nn.Sequential(
#                 nn.Linear(opt.hidden_size*2, opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(opt.hidden_size, opt.hidden_size),
#             )
#             self.fea_utt = nn.Sequential(
#                 nn.Linear(opt.hidden_size*2, opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(opt.hidden_size, opt.hidden_size),
#             )
#             self.cos_MLP = nn.Sequential(
#                 nn.Linear(opt.hidden_size*2, opt.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(opt.hidden_size, len(cos2idx)),
#             )
#         if self.opt.task == 'ECPF' or self.opt.task == 'ECPF-C':
#             self.fea_emo = nn.Sequential(
#                 nn.Linear(opt.hidden_size*2, opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(opt.hidden_size, opt.hidden_size),
#             )
#             self.fea_utt = nn.Sequential(
#                 nn.Linear(opt.hidden_size*2, opt.hidden_size),
#                 nn.Tanh(),
#                 nn.Linear(opt.hidden_size, opt.hidden_size),
#             )
#             self.emo_MLP = nn.Sequential(
#                 nn.Linear(opt.hidden_size, opt.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(opt.hidden_size, len(emo2idx)),
#             )
#             self.cos_MLP = nn.Sequential(
#                 nn.Linear(opt.hidden_size*2, opt.hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(opt.hidden_size, len(cos2idx)),
#             )

#     def forward(self,Speaker_input_ids, Speaker_input_attention_mask, input_ids, input_attention_mask,\
#                 describe_input_ids, describe_attention_masks, describe_token_type_ids,\
#                 Emotion_input_ids, Emotion_input_attention_mask, Cosequence_input_ids,\
#                 Cosequence_input_attention_mask, video_feature, video_feature_attention_mask,\
#                 audio_feature, audio_feature_attention_mask,spk_and_input_ids,\
#                 spk_and_input_ids_attention_mask,spk_and_input_token_type_ids,\
#                 crition_foremo,crition_forcos, opt, epoch) -> None:
#         batch_size,seq_len,_ =spk_and_input_ids.shape
#         dense_spk_and_input_ids = spk_and_input_ids.reshape(-1,spk_and_input_ids.shape[-1])
#         dense_spk_and_input_ids_attention_mask = spk_and_input_ids_attention_mask.reshape(-1,spk_and_input_ids_attention_mask.shape[-1])
#         dense_spk_and_input_token_type_ids = spk_and_input_token_type_ids.reshape(-1,spk_and_input_token_type_ids.shape[-1])
        
#         outputs = self.bert(input_ids=dense_spk_and_input_ids,token_type_ids=dense_spk_and_input_token_type_ids,attention_mask=dense_spk_and_input_ids_attention_mask)
      
#         # last_hidden_states = outputs.last_hidden_state
#         pooler_output=outputs.pooler_output
     
#         pooler_output = pooler_output.reshape(batch_size,seq_len,-1)
       
       
#         # raw method 
#         # text+video+audio
#         if opt.select_modality =='T':
#             mm_feature_cat = pooler_output
#         elif opt.select_modality =='TA':
#             mm_feature_cat = torch.cat((pooler_output,audio_feature),dim=2)
#         elif opt.select_modality =='TV':
#             mm_feature_cat = torch.cat((pooler_output,video_feature),dim=2)
#         elif opt.select_modality =='TAV':
#             mm_feature_cat = torch.cat((pooler_output,video_feature,audio_feature),dim=2)
#         else:
#             raise ValueError('Please select a modality')
        
#         mm_feature = self.simple_fusion(mm_feature_cat)
#         lstm_mm_feature,*_ = self.rnn(mm_feature)

        

#         if self.opt.task == 'CF':
#             # emo_emb = self.emo_embedding(Emotion_input_ids)
#             # for cos
#             emo_feature = self.fea_emo(lstm_mm_feature)
#             utt_feature = self.fea_utt(lstm_mm_feature)
          
#             emo_feature_repeated = emo_feature.unsqueeze(2).repeat(1,1,seq_len,1)
#             utt_feature_repeated = utt_feature.unsqueeze(1).repeat(1,seq_len,1,1)
#             emo_used_mask = (torch.gt(Emotion_input_ids,0)*1).unsqueeze(2)
#             emo_used_mask_reapeated = emo_used_mask.repeat(1,1,seq_len)
#             cos_feature = torch.cat((emo_feature_repeated,utt_feature_repeated),dim=3)
#             cos_score = self.cos_MLP(cos_feature)
#             pred_cos = cos_score.argmax(dim=3)
#             mask = Cosequence_input_attention_mask*emo_used_mask_reapeated
#             Cosequence_input_ids[mask==0]=-1
#             dense_cos_score = cos_score.reshape(-1,cos_score.shape[-1])
#             dense_Cosequence_input_ids=Cosequence_input_ids.reshape(-1)
           
#             loss = crition_forcos(dense_cos_score,dense_Cosequence_input_ids)
            
#             ones = torch.ones_like(pred_cos)
#             cos_used_mask4 = torch.triu(ones,diagonal=1)
#             mask4 = emo_used_mask_reapeated*cos_used_mask4
#             pred_cos[mask4==0]=-1
           
#             return loss,Emotion_input_ids,Emotion_input_ids,pred_cos,Cosequence_input_ids





        
#         if self.opt.task =='ECPF' or self.opt.task =='ECPF-C':
#             # very similar
       
#             # emo_feature = self.fea_emo(lstm_mm_feature)
#             utt_feature = self.fea_utt(lstm_mm_feature)
#             emo_feature =utt_feature
       

#             emo_scores = self.emo_MLP(emo_feature)
#             mask1 = Emotion_input_attention_mask
#             Emotion_input_ids[mask1==0]=-1
#             dense_emo_scores = emo_scores.reshape(-1,emo_scores.shape[-1])
#             dense_Emotion_input_ids=Emotion_input_ids.reshape(-1)
#             loss_emo = crition_foremo(dense_emo_scores,dense_Emotion_input_ids)
#             pred_emos = emo_scores.argmax(dim=2)

#             pred_emos[mask1==0]=0
          
#             # for cos
#             emo_feature_repeated = emo_feature.unsqueeze(2).repeat(1,1,seq_len,1)
#             utt_feature_repeated = utt_feature.unsqueeze(1).repeat(1,seq_len,1,1)
#             emo_used_mask = (torch.gt(pred_emos,0)*1).unsqueeze(2)
#             emo_used_mask_reapeated = emo_used_mask.repeat(1,1,seq_len)
#             cos_feature = torch.cat((emo_feature_repeated,utt_feature_repeated),dim=3)
#             cos_score = self.cos_MLP(cos_feature)
#             pred_cos = cos_score.argmax(dim=3)
     
#             mask2 = Cosequence_input_attention_mask*emo_used_mask_reapeated
#             cos_mask_input_ids = Cosequence_input_ids.clone()
#             cos_mask_input_ids[mask2==0]=-1
#             dense_cos_score = cos_score.reshape(-1,cos_score.shape[-1])
#             dense_Cosequence_input_ids=cos_mask_input_ids.reshape(-1)
           
#             loss_cos = crition_forcos(dense_cos_score,dense_Cosequence_input_ids)
#             loss = (loss_emo+loss_cos)/2

#             emo_used_mask3 = (torch.gt(Emotion_input_ids,0)*1).unsqueeze(2)
#             emo_used_mask3_reapeated = emo_used_mask3.repeat(1,1,seq_len)
#             mask3 = Cosequence_input_attention_mask*emo_used_mask3_reapeated
#             Cosequence_input_ids[mask3==0]=-1

           
#             ones = torch.ones_like(pred_cos)
#             cos_used_mask4 = torch.triu(ones,diagonal=1)
#             mask4 = emo_used_mask_reapeated*cos_used_mask4
#             pred_cos[mask4==0]=-1
            
#             # if math.isnan(loss):
#             #     stop()
#             return loss,pred_emos,Emotion_input_ids,pred_cos,Cosequence_input_ids