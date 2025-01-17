# [ECFCON: Emotion Consequence Forecasting in Conversations](https://dl.acm.org/doi/10.1145/3664647.3681413)

![image](https://github.com/user-attachments/assets/3a8e0491-01c9-4200-b795-ef44595b3120)

## Datas #This need some time to upload.
### checkpoint
* **result** the checkpoints of the main model (ECFCON-BERT), i.e., Emocos [baidu](https://pan.baidu.com/s/1cTRjlx1XrF9jP8aXzmV5Sw) code: 1t2b
### clues 
* **Video_LLaVA\prompt.ECPF-C_all.clue3** The clues from MLLMs   [baidu](https://pan.baidu.com/s/1Gm52oG5XlyxbPOwirHMIqg) code:1t2b
#### dataset 
[baidu](https://pan.baidu.com/s/1yW5mXeU7chbw96aVSQpZbg) code:1t2b. This data is too large to share.
This link will only exist for one day, so ask for an update at github if needed. This will be addressed later.
* **datasets\audio_features\source_hubert** The features of the audio, extracted by HuBert 
* **video_features\Captions_13B_mingpt4**  The Caption of the visual information
* **video_features\Captions_13B_mingpt4** The feature of the visual modal.
* **datasets\split_videos_scripts\coarse** the textual Scripts of the videos.   
   

## Process
### step 1
Select the parameter to train  
``` bash train.sh ```

## More details 
We will update the details later. If any questions, please contact me by email.
## Citation
``` @inproceedings{DBLP:conf/mm/JuZZLLZ24,
  author       = {Xincheng Ju and
                  Dong Zhang and
                  Suyang Zhu and
                  Junhui Li and
                  Shoushan Li and
                  Guodong Zhou},
  editor       = {Jianfei Cai and
                  Mohan S. Kankanhalli and
                  Balakrishnan Prabhakaran and
                  Susanne Boll and
                  Ramanathan Subramanian and
                  Liang Zheng and
                  Vivek K. Singh and
                  Pablo C{\'{e}}sar and
                  Lexing Xie and
                  Dong Xu},
  title        = {{ECFCON:} Emotion Consequence Forecasting in Conversations},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Multimedia,
                  {MM} 2024, Melbourne, VIC, Australia, 28 October 2024 - 1 November
                  2024},
  pages        = {2233--2241},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3664647.3681413},
  doi          = {10.1145/3664647.3681413},
  timestamp    = {Wed, 06 Nov 2024 22:17:27 +0100},
  biburl       = {https://dblp.org/rec/conf/mm/JuZZLLZ24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
