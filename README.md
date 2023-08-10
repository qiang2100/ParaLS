# ParaLS: Paraphraser-based Lexical Substitution

Lexical substitution (LS) aims at finding appropriate substitutes for a target word in a sentence. Recently,  LS methods based on pretrained language models have made remarkable progress, generating potential substitutes for a target word through analysis of its contextual surroundings. However, these methods tend to overlook the preservation of the sentence's meaning when generating the substitutes. This study explores how to generate the substitute candidates from a paraphraser, as the generated paraphrases from a paraphraser contain variations in word choice and preserve the sentence's meaning. Since we cannot directly generate the substitutes via commonly used decoding strategies, we propose two simple decoding strategies that focus on the variations of the target word during decoding. Experimental results show that our methods outperform state-of-the-art LS methods based on pre-trained language models on three benchmarks.



# Requirements and Installation

*  Our code is mainly based on [Fairseq](https://github.com/pytorch/fairseq) version=10.2 with customized modification of scripts, To start, you need to clone this repo and install fairseq firstly using pip install -e .
* [PyTorch](http://pytorch.org/) version = 1.7.1
* Python version >= 3.7
* Other dependencies: pip install -r requirements.txt
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

## Step 1: Downlaod the pretrained model

You need to download the paraphraser(Transformer) from [here](https://drive.google.com/file/d/1o5fUGJnTxMe9ASQWTxIlbWmbEqN_RQ6D/view?usp=sharing) and paraphraser(BART) from [here](https://drive.google.com/file/d/1hgUtUHmxw8R4tsnGsp8usS2vr773DJ6o/view?usp=drive_link), and put it into folder "checkpoints/⁨para⁩/transformer/⁩" and "checkpoints/⁨para⁩/bart/⁩" seperately. For candidates ranking, we use [BLEURT](https://huggingface.co/Elron/bleurt-large-512) and BARTscore(https://github.com/neulab/BARTScore).

## Step 2: Run our code 

(1) run ParaLS for lexical substitute dataset LS07

sh run_LS_Paraphraser.multi.ls07.sh # Transformer
sh run_LS_Paraphraser.multi.ls07.bart.sh # BART

(2)run ParaLS for lexical substitute dataset LS14(Default BART)

sh run_LS_Paraphraser.multi.ls14.sh # Transformer
sh run_LS_Paraphraser.multi.ls14.bart.sh # BART


# Citation
Jipeng Qiang and Kang Liu contributed the code. 
Please cite as:

@inproceedings{qiang-etal-2023-ParaLS,
    title = "ParaLS: Lexical Substitution via Pretrained Paraphraser",
    author = "Qiang, Jipeng  and Liu, Kang  and Li, Yun  and Yuan, Yunhao  and Zhu, Yi",
    booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics",
    year = "2023"
    }
# Contact 
If you have any question about the code. Please contact yzunlplk@163.com
