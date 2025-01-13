
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
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")


class STGraph(nn.Module):
    def __init__(self, opt,emo2idx,cos2idx) -> None:
        super(STGraph,self).__init__()
        self.opt=opt
        self.emo2idx=emo2idx
        self.cos2idx=cos2idx
        self.bert = BertModel.from_pretrained(opt.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)
        self.rnn = nn.LSTM(input_size=2*opt.hidden_size,hidden_size=opt.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        self.layer_norm = nn.LayerNorm(opt.hidden_size)
        if opt.select_modality =='T':
            num_node=1
        elif opt.select_modality =='TV':
            num_node=2
            self.flatten_VF=nn.Linear(opt.video_size, opt.hidden_size)
        elif opt.select_modality =='TA':
            num_node=2
            self.flatten_AF=nn.Linear(opt.audio_size, opt.hidden_size)
        elif opt.select_modality =='TAV':
            num_node=3
            self.flatten_VF=nn.Linear(opt.video_size, opt.hidden_size)
            self.flatten_AF=nn.Linear(opt.audio_size, opt.hidden_size)
        elif opt.select_modality =='TAVC':
            if self.opt.task == 'CF':
                num_node=5
            else:
                num_node=6
            self.flatten_VF=nn.Linear(opt.video_size, opt.hidden_size)
            self.flatten_AF=nn.Linear(opt.audio_size, opt.hidden_size)
        self.simple_fusion = nn.Sequential(
            nn.Linear(num_node*opt.hidden_size, 2*opt.hidden_size),
            nn.Tanh(),
            nn.Linear(2*opt.hidden_size, opt.hidden_size),
        )

        pooling_choice='mean'
        decay=0.7
        stride=[1,1,1,1,1]
        moving_window=[1,3,5,7,9]
        time_length=None
        

        self.positional_encoding = PositionalEncoding(opt.hidden_size,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(opt.hidden_size, opt.hidden_size, num_node, time_length, time_window_size=moving_window[0], stride=stride[0], decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(opt.hidden_size, opt.hidden_size, num_node, time_length, time_window_size=moving_window[1], stride=stride[1], decay = decay, pool_choice=pooling_choice)
        self.MPNN3 = GraphConvpoolMPNN_block_v6(opt.hidden_size, opt.hidden_size, num_node, time_length, time_window_size=moving_window[2], stride=stride[2], decay = decay, pool_choice=pooling_choice)
        self.MPNN4 = GraphConvpoolMPNN_block_v6(opt.hidden_size, opt.hidden_size, num_node, time_length, time_window_size=moving_window[3], stride=stride[3], decay = decay, pool_choice=pooling_choice)
        self.MPNN5 = GraphConvpoolMPNN_block_v6(opt.hidden_size, opt.hidden_size, num_node, time_length, time_window_size=moving_window[4], stride=stride[4], decay = decay, pool_choice=pooling_choice)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(opt.hidden_size * len(moving_window) , 2*opt.hidden_size)),
            ('th1', nn.Tanh()),
            ('fc2', nn.Linear(2*opt.hidden_size, 2*opt.hidden_size)),
        ]))



        if self.opt.task == 'CF':
            self.emo_embedding = nn.Embedding(len(emo2idx), opt.hidden_size)
            self.fea_emo = nn.Sequential(
                nn.Linear(opt.hidden_size*3, opt.hidden_size),
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
        if self.opt.task == 'ECPF' or self.opt.task == 'ECPF-C':
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
       
       
        # raw method 
        # make sure the shape of video_feature and audio_feature is the same as the shape of pooler_output
        if opt.select_modality =='T':
            mm_features = pooler_output.unsqueeze(2)
            mm_features= self.layer_norm(mm_features)
        elif opt.select_modality =='TV':
            flatten_video_feature = self.flatten_VF(video_feature)
            mm_features = torch.cat((pooler_output.unsqueeze(2),flatten_video_feature.unsqueeze(2)),dim=2)
            mm_features= self.layer_norm(mm_features)
        elif opt.select_modality =='TA':
            flatten_audio_feature = self.flatten_AF(audio_feature)
            mm_features = torch.cat((pooler_output.unsqueeze(2),flatten_audio_feature.unsqueeze(2)),dim=2)
            mm_features= self.layer_norm(mm_features)
        elif opt.select_modality =='TAV':
            flatten_video_feature = self.flatten_VF(video_feature)
            flatten_audio_feature = self.flatten_AF(audio_feature)
            # text+video+audio
            mm_features = torch.cat((pooler_output.unsqueeze(2),flatten_video_feature.unsqueeze(2),flatten_audio_feature.unsqueeze(2)),dim=2)
            mm_features= self.layer_norm(mm_features)
        elif opt.select_modality =='TAVC':
            flatten_video_feature = self.flatten_VF(video_feature)
            flatten_audio_feature = self.flatten_AF(audio_feature)
            # text+video+audio
            if self.opt.task == 'CF':
                mm_features = torch.cat((pooler_output.unsqueeze(2),flatten_video_feature.unsqueeze(2),flatten_audio_feature.unsqueeze(2),cos_clue_why_pooler_output.unsqueeze(2),cos_clue_impact_pooler_output.unsqueeze(2)),dim=2)
            else:
                mm_features = torch.cat((pooler_output.unsqueeze(2),flatten_video_feature.unsqueeze(2),flatten_audio_feature.unsqueeze(2),emo_clue_pooler_output.unsqueeze(2),cos_clue_why_pooler_output.unsqueeze(2),cos_clue_impact_pooler_output.unsqueeze(2)),dim=2)
            mm_features= self.layer_norm(mm_features)
           

        # encoding
      
        _,_,node_num,_=mm_features.shape
        mf_=mm_features.transpose(1,2)
        mf_ = mf_.reshape(-1,seq_len,opt.hidden_size)
        mf_ = self.positional_encoding(mf_)
        mf_ = mf_.reshape(batch_size,node_num,seq_len,opt.hidden_size)
        mm_features = mf_.transpose(1,2)


        


        mmf_1 = self.MPNN1(mm_features)
        mmf_2 = self.MPNN2(mm_features)
        mmf_3 = self.MPNN2(mm_features)
        mmf_4 = self.MPNN2(mm_features)
        mmf_5 = self.MPNN2(mm_features)

        features1 = torch.reshape(mmf_1, [batch_size,seq_len, -1])
        features2 = torch.reshape(mmf_2, [batch_size,seq_len, -1])
        features3 = torch.reshape(mmf_3, [batch_size,seq_len, -1])
        features4 = torch.reshape(mmf_4, [batch_size,seq_len, -1])
        features5 = torch.reshape(mmf_5, [batch_size,seq_len, -1])

        mm_feature1 = self.simple_fusion(features1)
        mm_feature2 = self.simple_fusion(features2)
        mm_feature3 = self.simple_fusion(features3)
        mm_feature4 = self.simple_fusion(features4)
        mm_feature5 = self.simple_fusion(features5)

        mm_features = torch.cat([mm_feature1,mm_feature2,mm_feature3,mm_feature4,mm_feature5],-1)

        mm_feas = self.fc(mm_features)
     
        # mm_feature_cat = torch.cat((pooler_output,video_feature,audio_feature),dim=2)
        # mm_feature = self.simple_fusion(mm_feature_cat)
        lstm_mm_feature,*_ = self.rnn(mm_feas)

        # lstm_mm_feature =mm_feas
        if self.opt.task == 'CF':
            emo_emb = self.emo_embedding(Emotion_input_ids)
            # for cos
            emo_feature = self.fea_emo(torch.cat((lstm_mm_feature,emo_emb),dim=2))
            utt_feature = self.fea_utt(lstm_mm_feature)
          
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





        
        elif self.opt.task =='ECPF' or self.opt.task =='ECPF-C':
            # very similar
       
            emo_feature = self.fea_emo(lstm_mm_feature)
            utt_feature = self.fea_utt(lstm_mm_feature)

            emo_scores = self.emo_MLP(emo_feature)
            mask1 = Emotion_input_attention_mask
            Emotion_input_ids[mask1==0]=-1
            dense_emo_scores = emo_scores.reshape(-1,emo_scores.shape[-1])
            dense_Emotion_input_ids=Emotion_input_ids.reshape(-1)
            loss_emo = crition_foremo(dense_emo_scores,dense_Emotion_input_ids)
            pred_emos = emo_scores.argmax(dim=2)
            pred_emos[mask1==0]=0
            # for cos
            emo_feature_repeated = emo_feature.unsqueeze(2).repeat(1,1,seq_len,1)
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




class GraphConvpoolMPNN_block_v6(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, time_window_size, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block_v6, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.BN = nn.BatchNorm1d(input_dim)

        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)

        self.pre_relation = Mask_Matrix(num_sensors,time_window_size,decay)

        self.pool_choice = pool_choice

    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)
        
        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        input_con_ = torch.transpose(input_con, 2,3)
        input_con_ = torch.reshape(input_con_, [bs*num_windows, time_window_size*num_sensors, feature_dim])
        
        A_input = self.graph_construction(input_con_)
        # print(A_input.size())
        # print(self.pre_relation.size())
       
        A_input = A_input*self.pre_relation


        input_con_ = torch.transpose(input_con_, -1, -2)
        input_con_ = self.BN(input_con_)
        input_con_ = torch.transpose(input_con_, -1, -2)
        X_output = self.MPNN(input_con_, A_input)


        X_output = torch.reshape(X_output, [bs, num_windows, time_window_size,num_sensors, self.output_dim])
        # print(X_output.size())

        if self.pool_choice == 'mean':
            X_output = torch.mean(X_output, 2)
        elif self.pool_choice == 'max':

            X_output, ind = torch.max(X_output, 2)
        else:
            print('input choice for pooling cannot be read')
        # X_output = torch.reshape(X_output, [bs, num_windows*time_window_size,num_sensors, self.output_dim])
        # print(X_output.size())

        return X_output

class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):

        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = torch.transpose(node_features, 1, 2)

        Adj = torch.bmm(node_features, node_features_1)

        eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj

class MPNN_mk_v2(nn.Module):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk_v2, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, outpuut_dinmension))
        self.theta = nn.ModuleList(theta)
        self.bn1 = nn.BatchNorm1d(outpuut_dinmension)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
       
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_,A)
            out_k = self.theta[kk](torch.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = torch.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        GCN_output_ = torch.transpose(GCN_output_, -1, -2)
        GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = torch.transpose(GCN_output_, -1, -2)

        return F.leaky_relu(GCN_output_)
    
def Mask_Matrix(num_node, time_length, decay_rate):
    Adj = torch.ones(num_node * time_length, num_node * time_length).cuda()
    for i in range(time_length):
        v = 0
        for r_i in range(i,time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
        v=0
        for r_i in range(i+1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i-r_i) * num_node
            idx_e_col = (i-r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1

    return Adj

def Conv_GraphST(input, time_window_size, stride):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = torch.transpose(input, 1, 3)
    pd=(time_window_size-1)//2
    if (time_window_size-1)%2!=0:
        raise ValueError('time_window_size should be odd')
   
    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride, padding=(0,pd))

    y_ = torch.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = torch.transpose(y_, 1,-1) 

    return y_

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x = x + torch.Tensor(self.pe[:, :x.size(1)],
        #                  requires_grad=False)
        # print(self.pe[0, :x.size(1),2:5])
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x
