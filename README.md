# ParaLS: Paraphraser-based Lexical Substitution

Lexical substitution (LS) aims at finding appropriate substitutes for a target word in a sentence. Recently,  LS methods based on pretrained language models have made remarkable progress, generating potential substitutes for a target word through analysis of its contextual surroundings. However, these methods tend to overlook the preservation of the sentence's meaning when generating the substitutes. This study explores how to generate the substitute candidates from a paraphraser, as the generated paraphrases from a paraphraser contain variations in word choice and preserve the sentence's meaning. Since we cannot directly generate the substitutes via commonly used decoding strategies, we propose two simple decoding strategies that focus on the variations of the target word during decoding. Experimental results show that our methods outperform state-of-the-art LS methods based on pre-trained language models on three benchmarks.



# Requirements and Installation

*  Our code is based on [Fairseq](https://github.com/pytorch/fairseq) version=10.2
* [PyTorch](http://pytorch.org/) version = 1.7.1
* Python version >= 3.7
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

## Step 1: Downlaod the pretrained paraphraser modeling

You need to download the paraphraser from [here](https://drive.google.com/file/d/1o5fUGJnTxMe9ASQWTxIlbWmbEqN_RQ6D/view?usp=sharing), and put it into folder "checkpoints/⁨para⁩/transformer/⁩"

## Step 2: Run our code 

(1) run ParaLS for lexical substitute dataset LS07

input "run_LS_Paraphraser.multi.ls07.sh"

(2)run ParaLS for lexical substitute dataset LS14

input "run_LS_Paraphraser.multi.ls14.sh"


# Citation

Please cite as:

``` bibtex

```
