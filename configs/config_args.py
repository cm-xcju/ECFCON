from email.policy import default
import os.path as path
import os
from pdb import set_trace as stop
from typing import List
import yaml
from utils.util import override_from_dict

def get_args(parser):

    # parser.add_argument('--epoch', type=int, default=12,
    #                     metavar='E', help='number of epochs')
    parser.add_argument('-bert', default='False', help='wheather use bert')
    parser.add_argument('-position', type=str, default='None',
                        metavar='F', help='which position')
    parser.add_argument('--cuda', type=int, default=3,
                        metavar='C', help='cuda device')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-multi_gpu', action='store_true')
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-test_only', action='store_true')
    parser.add_argument('-lr_decay', type=float, default=0)
    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-n_layers_enc', type=int, default=5)
    parser.add_argument('-load_pretrained', action='store_true')
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-lr_step_size', type=int, default=1)

    parser.add_argument('-Tail', type=str, default='1')


    parser.add_argument(
        '--dataset_basepath', default='../../datasets/labeled_data', help=' dataset base path')
    parser.add_argument(
        '--clue_path_root', default='../Video-LLaVA/my_model/results/prompt.ECPF-C_all.clue3', help=' dataset base path')
    
    parser.add_argument('--bert_path', type=str, default='../../TransModels/bert-base-chinese',
                        help='use context, the length contains the utterance itself')
    parser.add_argument('--decribe_bert_path', type=str, default='../../TransModels/bert-base-chinese',
                        help='use context, the length contains the utterance itself')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='which dataset')
    parser.add_argument('-emotion_class', type=str,
                        choices=['all', 'NPN', 'EN'], default='EN')
    parser.add_argument('--cfg_file', type=str, default='configs/EmoCos.yaml',)
    parser.add_argument('--task', type=str, default='CF', choices=['ER','CF', 'ECPF','ECPF-C'])
    parser.add_argument('--audio_extract', type=str, default='hubert', choices=['opensmile', 'hubert'])
    parser.add_argument('--select_model', type=str, default='EmoCos', choices=['LSmodel','EmoCos','Roberta', 'STGraph','PoolGraph','MLLMs','MECPE'])
    parser.add_argument('--select_modality', type=str, default='TAV', choices=['T', 'TA','TV','TAV','TAVC'])
    parser.add_argument('-have_describe', action='store_false')
    parser.add_argument('-save_loss_best', action='store_true')
    parser.add_argument('-save_acc_best', action='store_true')
    parser.add_argument('-save_f1_best', action='store_false')
    parser.add_argument('-save_mode', type=str,
                        choices=['all', 'best'], default='best')
    parser.add_argument('-small_size', type=int, default=100000)
    
    opt = parser.parse_args()
    return opt
    



def config_args(opt):
    opt.cuda = not opt.no_cuda
    with open(opt.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        override_from_dict(opt, yaml_dict)
    if opt.task=='ECPF-C':
        opt.emotion_class = 'all'
    opt.cuda = not opt.no_cuda
    opt.model_name =opt.select_model
    opt.model_name += '.'+opt.task+'_'+opt.emotion_class
    if opt.audio_extract =='hubert':
        opt.audio_size = 1024
        opt.model_name += '.hub'
    elif opt.audio_extract =='opensmile':
        opt.audio_size = 1582
        opt.model_name += '.ops'
    if opt.have_describe:
        opt.model_name += '.hd'

    if opt.select_model in ['EmoCos','STGraph']:
        opt.bert_path='../../TransModels/bert-base-chinese'
    elif opt.select_model in ['Roberta']:
        opt.bert_path = '../../TransModels/roberta-base-finetuned-chinanews-chinese'
        # opt.bert_path = '../../TransModels/chinese-roberta-wwm-ext'
    


    
    


    opt.model_name += '.s'+str(opt.seed)
    opt.model_name += '.mb'+str(opt.MIN_BATCH_SIZE)
    opt.model_name += '.lb'+str(opt.LR_BASE)
    opt.model_name +='.s'+str(opt.small_size)
    opt.model_name += '.'+str(opt.Tail)
    opt.model_name+='.'+opt.select_modality
    opt.model_name = path.join(opt.results_dir, opt.model_name)

    # opt.model_name =  path.join(opt.results_dir,'EmoCos.ECPF_EN.hub.hd.s42.mb16.lb5e-05.t5.TAV')
    if not path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    with open(opt.model_name+'/config.yaml', 'w') as f:
        yaml.dump(opt, f, default_flow_style=False)

    print(opt)



    return opt

