
# for baseline with no decription and graph
# for task in 'CF' #'ECPF' 'ER' 'ECPE-C'
# STGraph EmoCos LSmodel multi_gpu
# -test_only  
tail=clue2
seed=1234 #42 1234, 243
size=10000
for modality in 'TAVC' #'T' 'TA' 'TV' 'TAV' 'TAVC' 
do
for task in 'ECPF-C' #'ECPF' 'ECPF-C'
do
for select_model in 'EmoCos' # 'EmoCos' 'STGraph' 'LSmodel' 'Roberta' 'MECPE' 
do
    # echo python main.py  --task $task --select_model $select_model --select_modality $modality -Tail $tail
    CUDA_VISIBLE_DEVICES=1 python -u main.py  --task  $task --select_model  $select_model --select_modality $modality -small_size $size -Tail $tail -seed $seed -test_only
done
done
done