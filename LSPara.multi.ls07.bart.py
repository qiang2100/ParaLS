#!/usr/bin/python
# -*- coding: UTF-8 -*-

from operator import index
import os
from pyexpat import model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import csv
import logging
import os
import random
import math
import sys
import re

from sklearn.metrics.pairwise import cosine_similarity as cosine

import numpy as np
import torch
import nltk

import pdb
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import openpyxl
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from fairseq.models.transformer import TransformerModel

from transformers import AutoTokenizer, AutoModelWithLMHead

from nltk.stem import PorterStemmer

from bert_score.scorer import BERTScorer
from reader import Reader_lexical
from metrics.evaluation import evaluation
from bert_score.scorer import BERTScorer
from bart_score import BARTScorer
from bleurt import score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fairseq.models.bart import BARTModel
# scorer = BERTScorer(lang="en", rescale_with_baseline=True)
import string
punctuation = string.punctuation
qutos=["<unk>",'']
def gen_gpt2_string(tgt_dict,input):
    if type(input)==int:
        input=torch.tensor([input])
    return tgt_dict.bpe.decode(tgt_dict.task.source_dictionary.string(input))

bart_scorer=BARTScorer(device="cuda",checkpoint="/home/yz/liukang/liukang/huggingface/facebook/bart-large-cnn")
bart_scorer.load(path="/home/yz/liukang/liukang/huggingface/facebook/bart-large-cnn/bart.pth")


bleurt_tokenizer = AutoTokenizer.from_pretrained("bleurt-large-512")
bleurt_scorer = AutoModelForSequenceClassification.from_pretrained("bleurt-large-512").cuda()
bleurt_scorer.eval()



import json
word_pos_fp="../../Gloss/LS_infer/vocab/word_pos.json"
with open(word_pos_fp,"r") as f:
    pos_vocab = json.loads( f.read().strip() )

from tqdm import tqdm
def lemma_word(target, target_pos):
    lemmatizer = WordNetLemmatizer()
    to_wordnet_pos = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
    from_lst_pos = {'j': 'J', 'a': 'J', 'v': 'V', 'n': 'N', 'r': 'R'}
    try:
        pos_initial = to_wordnet_pos[from_lst_pos[target_pos]]
    except KeyError:
        pos_initial = to_wordnet_pos[target_pos]

    return lemmatizer.lemmatize(target, pos=pos_initial)

ps = PorterStemmer()

#scorer = BERTScorer(model_type="bert-base-uncased",lang="en", rescale_with_baseline=True)
scorer = BERTScorer(lang="en", rescale_with_baseline=True)
# scorer=None
import re
def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    # string = re.sub(" ' ", "'", string)
    return string



def give_embedding_scores(outputs,tokens_embedding,complex_tokens,temperature=None,prefix_len=None):
    complex_embed=tokens_embedding[complex_tokens[:-1]]
    outputs_embed=tokens_embedding[outputs[:,1:]]

    sim_cal=nn.CosineSimilarity(dim=-1, eps=1e-6) 
    if complex_embed.size(0)==1:
        sim_matrix=sim_cal(outputs_embed,complex_embed)
        if temperature!=None:
            sim_matrix=sim_matrix/temperature
        sim_matrix=F.log_softmax(sim_matrix,dim=0)
    else:
        sim_matrix=torch.zeros(outputs.size(0),(outputs.size(1)-1))
    return sim_matrix


def give_embedding_scores_v2(complex_word,candis,glove_model,temperature=None,tokens_embedding=None):
    if glove_model==None or tokens_embedding==None:
        return torch.zeros(len(candis))
    # try:
    if len(complex_word)>=2:
        return torch.zeros(len(candis))
    else:
        complex_embed=tokens_embedding[complex_word[0]]
    


    sim_cal=nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim_matrix=[]
    for candi1 in candis:
        try:
            candi1_token=glove_model.encode(candi1)[0].tolist()
            candi_embed=tokens_embedding[candi1_token]
            sim_candi_complex=sim_cal(complex_embed,candi_embed).tolist()
            sim_matrix.append(sim_candi_complex)
        except:
            sim_matrix.append(-1000)
    sim_matrix=torch.tensor(sim_matrix)
    if temperature!=None:
        sim_matrix=sim_matrix/temperature
    sim_matrix=F.log_softmax(sim_matrix,dim=0)
    return sim_matrix  

def give_embedding_scores_v4(complex_word,candis,glove_model,temperature=None,tokens_embedding=None):
    if glove_model==None or tokens_embedding==None:
        return torch.zeros(len(candis))
    # try:
    if len(complex_word)>=2:
        return torch.zeros(len(candis))
    else:
        complex_embed=tokens_embedding[complex_word[0]]
    
    # except:
    #     print("complex word not in the glove model")

    sim_cal=nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim_matrix=[]
    
    for candi1 in candis:
        try:
            all_candi_tokens=glove_model.encode(candi1)[1:-1].tolist()
            candi1_token=all_candi_tokens[0]
            candi_embed=tokens_embedding[candi1_token]
            for k1 in range(1,len(all_candi_tokens)):
                candi_embed+=tokens_embedding[all_candi_tokens[k1]]
            candi_embed=candi_embed/len(all_candi_tokens)

            sim_candi_complex=sim_cal(complex_embed,candi_embed).tolist()
            sim_matrix.append(sim_candi_complex)
        except:
            print("should not happend in line 202!!!")
            sim_matrix.append(-1000)
    sim_matrix=torch.tensor(sim_matrix)
    if temperature!=None:
        sim_matrix=sim_matrix/temperature
    sim_matrix=F.log_softmax(sim_matrix,dim=0)
    return sim_matrix


def change_embedding_scores(outputs,sim_matrix,prefix_len=None,max_ahead=5):
    beam_size,max_len=outputs.size()
    first_index=prefix_len
    last_index=min(prefix_len+max_ahead,max_len-1)
    for i in range(first_index+1,last_index):
        sim_matrix[:,i]=sim_matrix[:,first_index]
    return sim_matrix    

def get_glove_embedding(complex_word,candis,glove_model,temperature=None):
    try:
        complex_word=complex_word.lower()
        complex_embed=glove_model[complex_word]
    except:
        print("complex word not in the glove model")
        return torch.zeros(len(candis))
    sim_cal=nn.CosineSimilarity(dim=-1, eps=1e-6)
    sim_matrix=[]
    for candi1 in candis:
        try:
            candi_embed=glove_model[candi1.lower()]
            sim_candi_complex=sim_cal(torch.tensor(complex_embed),torch.tensor(candi_embed)).tolist()
            sim_matrix.append(sim_candi_complex)
        except:
            sim_matrix.append(-1000)
    sim_matrix=torch.tensor(sim_matrix)
    if temperature!=None:
        sim_matrix=sim_matrix/temperature
    sim_matrix=F.log_softmax(sim_matrix,dim=0)
    return sim_matrix   




@torch.no_grad()
def give_dynamic_embedding_scores(model,candis,prefix,ori_sentence,ori_index,layer_index=-1):
    sim_cal=nn.CosineSimilarity(dim=-1, eps=1e-6)
    import copy
    encoder=model.models[0].encoder
    ori_word_list=ori_sentence.split()
    ori_word=ori_word_list[ori_index]
    prefix_tokens=model.encode(prefix)[:-1]
    ori_sentence_tokens=model.encode(ori_sentence)
    ori_word_tokens=model.encode(ori_word)[1:-1]


    this_input_tokens=ori_sentence_tokens
    this_first_index=len(prefix_tokens)
    this_last_index=this_first_index+len(ori_word_tokens)


    if layer_index!=None:
        this_encoder_outs=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][layer_index].cpu().squeeze()
        this_ori_outs=torch.sum(this_encoder_outs[:,1:-1],dim=0)/(len(this_encoder_outs)-2)

    else:
        this_encoder_outs_1=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][-1].cpu().squeeze()[this_first_index:this_last_index]
        this_encoder_outs_2=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][-2].cpu().squeeze()[this_first_index:this_last_index]
        this_encoder_outs_3=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][-3].cpu().squeeze()[this_first_index:this_last_index]
        this_encoder_outs=torch.cat((this_encoder_outs_1,this_encoder_outs_2,this_encoder_outs_3),dim=-1)
        this_ori_outs=this_encoder_outs.squeeze()
    
    sim_matrix=[]
    for i in range(len(candis)):
        this_candi=candis[i]
        this_word_list=copy.deepcopy(ori_word_list)
        this_word_list[ori_index]=this_candi
        this_sentence=" ".join(this_word_list)
        this_input_tokens=model.encode(this_sentence)
        this_word_tokens=model.encode(this_candi)[:-1]
        this_last_index=this_first_index+len(this_word_tokens)
        if layer_index!=None:
            this_encoder_outs=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][layer_index].cpu().squeeze()
            this_dynamic_outs=torch.sum(this_encoder_outs[:,1:-1],dim=0)/(len(this_encoder_outs)-2)
            

        else:
            this_encoder_outs_1=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][-1].cpu().squeeze()[this_first_index:this_last_index]
            this_encoder_outs_2=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][-2].cpu().squeeze()[this_first_index:this_last_index]
            this_encoder_outs_3=encoder(this_input_tokens.unsqueeze(0).cuda(),return_all_hiddens=True)["encoder_states"][-3].cpu().squeeze()[this_first_index:this_last_index]
            this_encoder_outs=torch.cat((this_encoder_outs_1,this_encoder_outs_2,this_encoder_outs_3),dim=-1)
            this_dynamic_outs=this_encoder_outs
            this_dynamic_outs=(torch.sum(this_dynamic_outs,dim=0)/len(this_dynamic_outs)).squeeze()
        if i==10:
            import pdb
            pdb.set_trace()
        sim_candi_complex=sim_cal(this_dynamic_outs,this_ori_outs).tolist()
        sim_matrix.append(sim_candi_complex)

    sim_matrix=torch.tensor(sim_matrix)


    return  sim_matrix








@torch.no_grad()
def cal_bart_score(sentence,complex_word,complex_index,candis):
    candis_scores=[]
    sentence_list=sentence.split()
    #prefix=" ".join(sentence.split()[max(0,complex_index-4):complex_index])
    #suffix=" ".join(sentence.split()[complex_index+1:min(complex_index+5,len(sentence_list))])

    prefix=" ".join(sentence.split()[0:complex_index]).strip()
    suffix=" ".join(sentence.split()[complex_index+1:]).strip()
        
    ori_sentence=prefix+" "+complex_word.strip()+" "+suffix
    ori_sentence=ori_sentence.strip()
    cal_oris=[]
    cal_news=[]
    for candi in candis:
        cal_oris.append(ori_sentence)
        new_sentence=prefix+" "+candi+" "+suffix
        new_sentence=new_sentence.strip()
        cal_news.append(new_sentence)

    #F1=bart_scorer.score(cal_news,cal_oris)
    with torch.no_grad():
        F1=bart_scorer.score(cal_oris,cal_news)
    #F1=np.exp(F1).tolist()
    F1=torch.tensor(F1)
    return F1

@torch.no_grad()
def cal_bleurt_score(sentence,complex_word,complex_index,candis):
    candis_scores=[]
    sentence_list=sentence.split()


    prefix=" ".join(sentence.split()[0:complex_index]).strip()
    suffix=" ".join(sentence.split()[complex_index+1:]).strip()
        
    ori_sentence=prefix+" "+complex_word.strip()+" "+suffix
    ori_sentence=ori_sentence.strip()
    cal_oris=[]
    cal_news=[]
    for candi in candis:
        cal_oris.append(ori_sentence)
        new_sentence=prefix+" "+candi+" "+suffix
        new_sentence=new_sentence.strip()
        cal_news.append(new_sentence)

    #F1=bart_scorer.score(cal_news,cal_oris)
    with torch.no_grad():
        #F1=bleurt_scorer.score(references=cal_oris, candidates=cal_news)
        input_dict=bleurt_tokenizer(cal_oris, cal_news, return_tensors='pt', padding=True)
        F1=bleurt_scorer(input_ids=input_dict["input_ids"].cuda(),\
                                token_type_ids=input_dict["token_type_ids"].cuda(),\
                                attention_mask=input_dict["attention_mask"].cuda())[0].squeeze()

    return F1.cpu()

@torch.no_grad()
def cal_bert_score(sentence,complex_word,complex_index,candis):
    candis_scores=[]
    sentence_list=sentence.split()
    prefix=" ".join(sentence.split()[max(0,complex_index-4):complex_index])
    suffix=" ".join(sentence.split()[complex_index+1:min(complex_index+5,len(sentence_list))])
    ori_sentence=prefix+" "+complex_word.strip()+" "+suffix
    ori_sentence=ori_sentence.strip()
    cal_oris=[]
    cal_news=[]
    for candi in candis:
        cal_oris.append(ori_sentence)
        new_sentence=prefix+" "+candi+" "+suffix
        new_sentence=new_sentence.strip()
        cal_news.append(new_sentence)
    P,R,F1=scorer.score(cal_news,cal_oris)
    return F1

def give_real_scores(combined_sss,prev_masks,prev_again_masks,suffix_tokens):
    if combined_sss[1]==[]:
        return combined_sss[0][0]
    else:
        prefix_score=combined_sss[0][1]-combined_sss[0][0]
        final_scores=torch.where(prev_masks,combined_sss[-2]-prefix_score,combined_sss[-3]-prefix_score)
        final_scores=torch.where(prev_again_masks*prev_masks,combined_sss[-1]-prefix_score,final_scores)
        final_scores/=(len(suffix_tokens)+1)
        final_scores[prev_masks]=final_scores[prev_masks]*((len(suffix_tokens)+1)/(len(suffix_tokens)+2))
        final_scores[prev_again_masks*prev_masks]=final_scores[prev_again_masks*prev_masks]*((len(suffix_tokens)+2)/(len(suffix_tokens)+1)*(len(suffix_tokens)+3))


        return final_scores


def give_real_scores_onlysuffix(combined_sss,prev_masks,prev_again_masks,suffix_tokens):
    if combined_sss[1]==[]:
        return combined_sss[0][0]
    else:
        # _0_suffix_score=(combined_sss[-3]-combined_sss[0][1]+combined_sss[0][0])/(len(suffix_tokens)+1)
        # _1_suffix_score=(combined_sss[-2]-combined_sss[1]+combined_sss[0][0]+0.25*combined_sss[1])/(len(suffix_tokens)+2)
        # _2_suffix_score=(combined_sss[-1]-combined_sss[2]+combined_sss[0][0]+0.25*combined_sss[1]+combined_sss[2])/(len(suffix_tokens)+3)

        _0_suffix_score=combined_sss[-3]-combined_sss[0][1]+combined_sss[0][0]
        _1_suffix_score=combined_sss[-2]-combined_sss[1]+combined_sss[0][0]
        _2_suffix_score=combined_sss[-1]-combined_sss[2]+combined_sss[0][0]

        final_scores=torch.where(prev_masks,_1_suffix_score, _0_suffix_score)
        final_scores=torch.where(prev_again_masks*prev_masks,_2_suffix_score,final_scores)
        return final_scores

def give_real_scores_ahead(tgt_dict,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=None,prefix_str=None,max_ahead=1,flag=1):


    beam_size,max_len=outputs.size()
    scores_with_suffix=scores_with_suffix[:,:max_len-1]
    scores_with_suffix_masks=scores_with_suffix_masks[:,:max_len-1]


    first_index=prefix_len
    last_index=min(prefix_len+max_ahead,max_len-1)
    # print(scores_with_suffix[:,0:5])
    for i in range(first_index,last_index):
        if first_index>0:     
            scores_with_suffix[:,i]-=scores_with_suffix[:,first_index-1]
        else:
            pass

    scores_with_suffix[scores_with_suffix_masks]=-math.inf
    for j in range(0,first_index):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)
    for j in range(last_index,max_len-1):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)      

    flat_scores_with_suffix=scores_with_suffix.reshape(1,-1).squeeze(dim=0)
    sorted_scores,sorted_indices=torch.topk(flat_scores_with_suffix,k=beam_size*(last_index-first_index))
    beam_idx=sorted_indices//(max_len-1)
    len_idx=(sorted_indices%(max_len-1))+1
    if flag!=None:
        hope_len=len(nltk.word_tokenize(prefix_str))+flag
        #hope_len=len(prefix_str.strip().split())+flag
    else:
        hope_len=-1

    hope_outputs=[]
    hope_outputs_scores=[]
    candis=[]
    for i in range(len(beam_idx)):
        if sorted_scores[i]==(-math.inf):
            continue
        tmp_str1=tgt_dict.string(outputs[beam_idx[i],:(len_idx[i]+1)]).replace("@@ ","")
        #tmp_str1=tmp_str1.replace("<unk>","")
        #if len(tmp_str1.strip().split())==hope_len:
        if "<unk>" in tmp_str1:
            print("finding a unk in prefix str")
            tmp_str1=tmp_str1.replace("<unk>","|")

        if len(nltk.word_tokenize(tmp_str1))==hope_len:
            candis.append(nltk.word_tokenize(tmp_str1)[-1].strip())
            hope_outputs.append(outputs[beam_idx[i]])
            #print(tgt_dict.string(outputs[beam_idx[i]]),sorted_scores[i])
            hope_outputs_scores.append(sorted_scores[i].tolist())
        elif hope_len==-1:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())

    return hope_outputs,hope_outputs_scores,candis



def give_real_scores_ahead_bart(tgt_dict,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=None,prefix_str=None,max_ahead=1,flag=1):

    beam_size,max_len=outputs.size()
    scores_with_suffix=scores_with_suffix[:,:max_len-1]
    scores_with_suffix_masks=scores_with_suffix_masks[:,:max_len-1]

    first_index=prefix_len
    last_index=min(prefix_len+max_ahead,max_len-1)

    for i in range(first_index,last_index):
        if first_index>0:     
            scores_with_suffix[:,i]-=scores_with_suffix[:,first_index-1]
        else:
            pass

    ahead_parts=outputs[:,1:]
    ahead_parts=ahead_parts.reshape(1,-1)[0].tolist()
    #ahead_part_tokens=list(map(lambda x:tokenizer.convert_ids_to_tokens(x),ahead_parts))
    ahead_part_tokens=list(map(lambda x:gen_gpt2_string(tgt_dict,x),ahead_parts))
    ahead_part_tokens_masks=list(map(lambda x:not x.startswith(" ") and x not in qutos,ahead_part_tokens))
    ahead_part_tokens_masks=torch.tensor(ahead_part_tokens_masks).reshape(beam_size,-1)
    scores_with_suffix[:,:-1][ahead_part_tokens_masks[:,1:]]=-math.inf


    if first_index>1:
        ids_after_prefix=outputs[:,first_index+1]
        ids_after_prefix=ids_after_prefix.reshape(1,-1)[0].tolist()
        ids_after_prefix_tokens=list(map(lambda x:gen_gpt2_string(tgt_dict,x),ids_after_prefix))
        ids_after_prefix_tokens_masks=list(map(lambda x:not x.startswith(" ") and x not in qutos,ids_after_prefix_tokens))
        ids_after_prefix_tokens_masks=torch.tensor(ids_after_prefix_tokens_masks).reshape(beam_size,-1)
        scores_with_suffix[ids_after_prefix_tokens_masks.squeeze()]=-math.inf


    scores_with_suffix[scores_with_suffix_masks]=-math.inf
    for j in range(0,first_index):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)
    for j in range(last_index,max_len-1):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)      

    flat_scores_with_suffix=scores_with_suffix.reshape(1,-1).squeeze(dim=0)
    sorted_scores,sorted_indices=torch.topk(flat_scores_with_suffix,k=beam_size*(last_index-first_index))
    beam_idx=sorted_indices//(max_len-1)
    len_idx=(sorted_indices%(max_len-1))+1
    if flag!=None:
        hope_len=len(nltk.word_tokenize(prefix_str))+flag
        #hope_len=len(prefix_str.strip().split())+flag
    else:
        hope_len=-1

    hope_outputs=[]
    hope_outputs_scores=[]
    candis=[]
    for i in range(len(beam_idx)):
        if sorted_scores[i]==(-math.inf):
            continue
        
        tmp_str1=tgt_dict.bpe.decode(tgt_dict.task.source_dictionary.string(outputs[beam_idx[i],:(len_idx[i]+1)])).strip()

        if "<unk>" in tmp_str1:
            print("finding a unk in prefix str")
            tmp_str1=tmp_str1.replace("<unk>","|")

        if len(nltk.word_tokenize(tmp_str1))==hope_len:
            #candis.append(nltk.word_tokenize(tmp_str1)[-1].strip())
            candis.append(" "+nltk.word_tokenize(tmp_str1)[-1].strip())
            hope_outputs.append(outputs[beam_idx[i]])
            #print(tgt_dict.string(outputs[beam_idx[i]]),sorted_scores[i])
            hope_outputs_scores.append(sorted_scores[i].tolist())
        elif hope_len==-1:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())

    return hope_outputs,hope_outputs_scores,candis




def give_real_scores_ahead_bart_len_regular(tgt_dict,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=None,prefix_str=None,max_ahead=1,flag=1):

    beam_size,max_len=outputs.size()
    scores_with_suffix=scores_with_suffix[:,:max_len-1]
    scores_with_suffix_masks=scores_with_suffix_masks[:,:max_len-1]

    first_index=prefix_len
    last_index=min(prefix_len+max_ahead,max_len-1)

    for i in range(first_index,last_index):
        if first_index>0:     
            scores_with_suffix[:,i]-=scores_with_suffix[:,first_index-1]
        else:
            pass

    ahead_parts=outputs[:,1:]
    ahead_parts=ahead_parts.reshape(1,-1)[0].tolist()
    #ahead_part_tokens=list(map(lambda x:tokenizer.convert_ids_to_tokens(x),ahead_parts))
    ahead_part_tokens=list(map(lambda x:gen_gpt2_string(tgt_dict,x),ahead_parts))
    ahead_part_tokens_masks=list(map(lambda x:not x.startswith(" ") and x not in qutos,ahead_part_tokens))
    ahead_part_tokens_masks=torch.tensor(ahead_part_tokens_masks).reshape(beam_size,-1)
    scores_with_suffix[:,:-1][ahead_part_tokens_masks[:,1:]]=-math.inf


    if first_index>1:
        ids_after_prefix=outputs[:,first_index+1]
        ids_after_prefix=ids_after_prefix.reshape(1,-1)[0].tolist()
        ids_after_prefix_tokens=list(map(lambda x:gen_gpt2_string(tgt_dict,x),ids_after_prefix))
        ids_after_prefix_tokens_masks=list(map(lambda x:not x.startswith(" ") and x not in qutos,ids_after_prefix_tokens))
        ids_after_prefix_tokens_masks=torch.tensor(ids_after_prefix_tokens_masks).reshape(beam_size,-1)
        scores_with_suffix[ids_after_prefix_tokens_masks.squeeze()]=-math.inf


    scores_with_suffix[scores_with_suffix_masks]=-math.inf
    for j in range(0,first_index):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)
    for j in range(last_index,max_len-1):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)      

    flat_scores_with_suffix=scores_with_suffix.reshape(1,-1).squeeze(dim=0)
    sorted_scores,sorted_indices=torch.topk(flat_scores_with_suffix,k=beam_size*(last_index-first_index))
    beam_idx=sorted_indices//(max_len-1)
    len_idx=(sorted_indices%(max_len-1))+1
    if flag!=None:
        hope_len=len(nltk.word_tokenize(prefix_str))+flag
        #hope_len=len(prefix_str.strip().split())+flag
    else:
        hope_len=-1

    hope_outputs=[]
    hope_outputs_scores=[]
    candis=[]
    for i in range(len(beam_idx)):
        if sorted_scores[i]==(-math.inf):
            continue
        
        tmp_str1=tgt_dict.bpe.decode(tgt_dict.task.source_dictionary.string(outputs[beam_idx[i],:(len_idx[i]+1)])).strip()
        tmp_str1_len=len_idx[i]-first_index+len(suffix_tokens)

        if "<unk>" in tmp_str1:
            print("finding a unk in prefix str")
            tmp_str1=tmp_str1.replace("<unk>","|")

        if len(nltk.word_tokenize(tmp_str1))==hope_len:
            #candis.append(nltk.word_tokenize(tmp_str1)[-1].strip())
            candis.append(" "+nltk.word_tokenize(tmp_str1)[-1].strip())
            hope_outputs.append(outputs[beam_idx[i]])

            hope_outputs_scores.append(sorted_scores[i].tolist()/(tmp_str1_len.tolist()))
        elif hope_len==-1:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())

    return hope_outputs,hope_outputs_scores,candis



def evaulation_SS_scores(ss,labels):
    assert len(ss)==len(labels)

    potential = 0
    instances = len(ss)
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0

    for i in range(len(ss)):

        one_prec = 0
        
        common = list(set(ss[i])&set(labels[i]))

        if len(common)>=1:
            potential +=1
        precision += len(common)
        recall += len(common)
        precision_all += len(set(ss[i]))
        recall_all += len(set(labels[i]))

    potential /=  instances
    precision /= precision_all
    recall /= recall_all
    F_score = 2*precision*recall/(precision+recall)

    return potential,precision,recall,F_score

def extract_substitute(output_sentences, original_sentence, complex_word, threshold,prev_scores=None,word_index=None,sentence_words=None,target_pos=None,target_lemma=None):

    original_words = sentence_words

    index_of_complex_word = -1


    index_of_complex_word=word_index
    if index_of_complex_word == -1:
        print("******************no found the complex word*****************")
        return [],[]

    
    len_original_words = len(original_words)
    context = original_words[max(0,index_of_complex_word-4):min(index_of_complex_word+5,len_original_words)]
    context = " ".join([word for word in context])

    if index_of_complex_word < 4:
        index_of_complex_in_context = index_of_complex_word
    else:
        index_of_complex_in_context = 4


    context = (context,index_of_complex_in_context)

    if output_sentences[0].find('<unk>'):

        for i in range(len(output_sentences)):
            tran = output_sentences[i].replace('<unk>', '|')
            output_sentences[i] = tran

    complex_stem = ps.stem(complex_word)
    # orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    not_candi = {'the', 'with', 'of', 'a', 'an', 'for', 'in', "-", "``", "*", "\"","it",""}
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    not_candi.add(target_lemma)

    if len(original_words) > 1:
        not_candi.add(original_words[index_of_complex_word - 1])
    if len(original_words) > index_of_complex_word+ 1:
        not_candi.add(original_words[index_of_complex_word + 1])

    

    #para_scores = []
    substitutes = []

    suffix_words = []

    if index_of_complex_word+1 < len_original_words:
        suffix_words.append(original_words[index_of_complex_word+1]) #suffix_words.append(original_words[index_of_complex_word+1:min(index_of_complex_word+4,len_original_words)])
    else:
        suffix_words.append("")
    
    #pdb.set_trace()

    for sentence in output_sentences:

        if len(sentence)<3:
            continue
        assert len(sentence.split()[:index_of_complex_word])==len(original_words[:index_of_complex_word])
        words=original_words[:index_of_complex_word]+nltk.word_tokenize(" ".join(sentence.split()[index_of_complex_word:]))

        if index_of_complex_word>=len(words):
            continue

        if words[index_of_complex_word] == complex_word:
            len_words = len(words)
            if index_of_complex_word+1 < len_words:
                suffix = words[index_of_complex_word+1]#words[index_of_complex_word+1:min(index_of_complex_word+4,len_words)]
                if suffix not in suffix_words:
                    suffix_words.append(suffix)
                    


    real_prev_scores=[]
    s1_count=-1

    for sentence in output_sentences:    

        s1_count+=1
        if len(sentence)<3:
            continue

        assert len(sentence.split()[:index_of_complex_word])==len(original_words[:index_of_complex_word])
        words=original_words[:index_of_complex_word]+nltk.word_tokenize(" ".join(sentence.split()[index_of_complex_word:]))

        if index_of_complex_word>=len(words):
            continue
        candi = words[index_of_complex_word].lower()
        candi_stem = ps.stem(candi)
        candi_lemma=lemma_word(candi, target_pos=target_pos)
        not_index_0 = candi.find("-")
        not_index_1 = candi.find(complex_word)
        if candi_lemma == target_lemma or candi_stem in not_candi or candi in not_candi or not_index_0 != -1 \
                or not_index_1 != -1 or candi[0] in punctuation or len(candi)==1 or candi[-1] in punctuation:
            continue


        len_words = len(words)
        sent_suffix = ""
        if index_of_complex_word + 1 < len_words:
            sent_suffix = words[index_of_complex_word+1]

        #if sent_suffix in suffix_words:
        if candi not in substitutes:
            substitutes.append(candi)
            real_prev_scores.append(prev_scores[s1_count])

    if len(substitutes)>0:

        filter_substitutes=substitutes
        rank_bert_substitutes=substitutes

        assert len(filter_substitutes)==len(real_prev_scores)
        assert len(filter_substitutes)==len(rank_bert_substitutes)

        return filter_substitutes, rank_bert_substitutes,real_prev_scores

    return [],[],[]

def substitutes_BertScore(context, target, substitutes):

    refs = []
    cands = []
    target_id = context[1]
    sent = context[0]

    words = sent.split(" ")
    for sub in substitutes:
        refs.append(sent)
        
        new_sent = ""
        
        for i in range(len(words)):
            if i==target_id:
                new_sent += sub + " "
            else:
                new_sent += words[i] + " "
        cands.append(new_sent.strip())

    P, R, F1 = scorer.score(cands, refs)

    return F1

def filterSubstitute(substitutes, bert_scores, threshold):

    max_score = np.max(bert_scores)

    if max_score - 0.1 > threshold:
        threshold = max_score - 0.1
    threshold=0

    filter_substitutes = []
    filter_bert_scores = []

    for i in range(len(substitutes)):
    # if(bert_scores[i]>threshold):
        filter_substitutes.append(substitutes[i])
        filter_bert_scores.append(bert_scores[i])

    return filter_substitutes, filter_bert_scores


def lexicalSubstitute(model, sentence, sentence_words,prefix,word_index,complex_word,target_pos,target_lemma,beam, threshold):
    
    index_complex = word_index
    
    ori_words=sentence_words
    prefix = prefix
    suffix1 = ""
    if(index_complex != -1):
        prefix = prefix
        if len(ori_words)>index_complex+1:
            suffix1=" ".join(ori_words[index_complex+1:]).strip()
            suffix1=suffix1.replace("''","\"").strip()
            suffix1=suffix1.replace("``","\"").strip()
                        
            suffix1=process_string(suffix1)
            #stored_suffix1=suffix1

            if suffix1.endswith("\""):
                suffix1=suffix1[:-1]
                suffix1=suffix1.strip()

            if suffix1.endswith("'"):
                suffix1=suffix1[:-1]
                suffix1=suffix1.strip() 

        if index_complex+1<len(ori_words):
            if ori_words[index_complex+1]=="-":
                if suffix1.startswith("-"):
                    suffix1=suffix1[1:].strip()
            if ori_words[index_complex+1]=="--":
                if suffix1.startswith("--"):
                    suffix1=suffix1[2:].strip()

            suffix1=" ".join(suffix1.split(" ")[:2])


        else:
            pass
        #print(prefix)
    else:
        print("*************cannot find the complex word")
        #print(sentence)
        #print(complex_word)
        sentence = sentence.lower()

        return lexicalSubstitute(model, sentence, complex_word,  beam, threshold)
    if ori_words[0]=="\"" and index_complex==1:
        prefix=" "+prefix
    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)
    
    if index_complex!=0:
        complex_tokens = model.encode(" "+complex_word.strip())[1:]
    else:
        complex_tokens = model.encode(" "+complex_word.strip())[1:]
    if index_complex+1<len(ori_words):
        if ori_words[index_complex+1]=="-":
            print("finding a ---------")
            sentence=" ".join(ori_words[:index_complex+1]+ori_words[index_complex+2:])

        if ori_words[index_complex+1]=="--":
            print("finding a ---------")
            sentence=" ".join(ori_words[:index_complex+1]+ori_words[index_complex+2:])
    



    if index_complex==0:
        sentence_tokens = model.encode(" "+sentence)
    elif index_complex==1 and ori_words[0]=="\"":
        sentence_tokens = model.encode(" "+sentence)
    else:
        sentence_tokens = model.encode(sentence)
    

    if suffix1!="":
        suffix_tokens=model.encode(" "+suffix1.strip())[1:-1]
    else:
        suffix_tokens=model.encode(suffix1.strip())[1:-1]
    #suffix_tokens=torch.tensor(suffix_tokens)
    suffix_tokens=suffix_tokens.tolist()
    attn_len = len(prefix_tokens[0])+len(complex_tokens)-1


    outputs,combined_sss,prev_masks,prev_masks2,scores_with_suffix,scores_with_suffix_masks,scores_with_dynamic = model.generate3([sentence_tokens.cuda()], 
                                                                                                            beam=beam, 
                                                                                                            prefix_tokens=prefix_tokens.cuda(), 
                                                                                                            attn_len=attn_len,
                                                                                                            #tgt_token=complex_tokens[:-1].tolist(),
                                                                                                            tgt_token=-1,
                                                                                                            suffix_ids=suffix_tokens,
                                                                                                            max_aheads=5)    
    outputs=outputs[0].cpu()
    

    scores_with_suffix=scores_with_suffix.cpu()
    scores_with_suffix_masks=scores_with_suffix_masks.cpu()


    outputs,outputs_scores,candis=give_real_scores_ahead_bart(
                                                        model,
                                                        outputs,
                                                        scores_with_suffix,
                                                        scores_with_suffix_masks,
                                                        suffix_tokens,
                                                        prefix_len=len(prefix_tokens[0]),
                                                        prefix_str=prefix,
                                                        max_ahead=5,
                                                        flag=1)

    
    
    new_outputs_scores=torch.tensor(outputs_scores)
    #new_outputs_scores=torch.tensor(outputs_scores)
    outputs_scores=new_outputs_scores
    new_indices=torch.topk(outputs_scores,k=len(outputs_scores),dim=0)[1]
    outputs=[outputs[index1] for index1 in new_indices]
 
    outputs_scores=[outputs_scores[index1].tolist() for index1 in new_indices]

    output_sentences=[model.bpe.decode(model.task.source_dictionary.string(x)) for x in outputs]
    if output_sentences==[]:
        print("find a missing prefix sentence!!!")
        return [],[],[],[]

    bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores = extract_substitute(output_sentences, sentence, complex_word,
                                                                              threshold,outputs_scores,word_index,sentence_words,target_pos,target_lemma)
    #print(pre_scores)

    #for sen in output_sentences:
    #    print(sen)

    bertscore_substitutes=bertscore_substitutes[:50]
    ranking_bertscore_substitutes=ranking_bertscore_substitutes[:50]
    real_prev_scores=real_prev_scores[:50]

    # if index_complex!=0:
    tmp_bertscore_substitutes=[" "+word1.strip() for word1 in bertscore_substitutes]


    glove_scores=cal_bart_score(sentence,complex_word,word_index,bertscore_substitutes)+cal_bleurt_score(sentence,complex_word,word_index,bertscore_substitutes)

    return bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores,glove_scores.tolist()


def pos_filter(pos_vocab,target_pos,candi,candi_lemma):
        PosMap={"v":"VERB", "n":"NOUN", "a":"ADJ", "r":"ADV"}
        word_form = None 
        target_pos1=PosMap[target_pos]
        if candi not in pos_vocab:
            return True
        # if candi in pos_vocab and target_pos1 in pos_vocab[candi]:
        #     word_form = candi 
        if candi_lemma in pos_vocab and target_pos1  in pos_vocab[candi_lemma]:
            word_form = candi_lemma
            
        if target_pos1=="ADJ":
            if "ADV" in pos_vocab[candi_lemma]:
                word_form = candi_lemma

        if target_pos1=="ADV":
            if "ADJ" in pos_vocab[candi_lemma]:
                word_form = candi_lemma

            
        if word_form==None:
            return False
        return True

def write_all_results(main_word, instance, target_pos, output_results, substitutes, substitutes_scores,
                      evaluation_metric):
    proposed_words = {}
    for substitute_str, score in zip(substitutes, substitutes_scores):
        substitute_lemma = lemma_word(
            substitute_str,
            target_pos
        ).lower().strip()
        max_score = proposed_words.get(substitute_lemma)
        if max_score is None or score > max_score:
            #if pos_filter(pos_vocab,target_pos,substitute_str,substitute_lemma):
            proposed_words[substitute_lemma] = score

    evaluation_metric.write_results(
        output_results + "_probabilites.txt",
        main_word, instance,
        proposed_words
    )
    evaluation_metric.write_results_p1(
        output_results + "_p1.txt",
        main_word, instance,
        proposed_words
    )

    evaluation_metric.write_results_p1(
        output_results + "_p3.txt",
        main_word, instance,
        proposed_words,limit=3
    )



    evaluation_metric.write_results_lex_oot(
        output_results + ".oot",
        main_word, instance,
        proposed_words, limit=10
    )

    # evaluation_metric.write_results_lex_oot(
    #     output_results + ".oot",
    #     main_word, instance,
    #     proposed_words, limit=10
    # )

    evaluation_metric.write_results_lex_best(
        output_results + ".best",
        main_word, instance,
        proposed_words, limit=1
    )
   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The evaluation data dir.")


    parser.add_argument("--eval_gold_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The evaluation data dir.")

    parser.add_argument("--paraphraser_path", default=None, type=str, required=True,
                        help=" the checkpoint path of paraphraser")
    parser.add_argument("--paraphraser_model", default=None, type=str, required=True,
                        help=" the checkpoint path of paraphraser")

    parser.add_argument("--bpe", default=None, type=str, required=True,
                        help=" which bpe")

    parser.add_argument("--bpe_codes", default=None, type=str, required=True,
                        help=" which bpe")

    # parser.add_argument("--paraphraser_dict", default=None, type=str, required=True,
    #                     help=" the dict path of paraphraser")

    parser.add_argument("--output_SR_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory of writing substitution selection.")

    parser.add_argument("--output_score_file",
                        type=str,
                        required=True,
                        help="path of the output file metric")


    parser.add_argument("--beam",
                        default=20,
                        type=int,
                        help="The number of beam.")
    parser.add_argument("--bertscore",
                        default=0.8,
                        type=float,
                        help="The value of bertscore.")

 
    args = parser.parse_args()

    output_sr_file = open(args.output_SR_file, "w+")
    reader = Reader_lexical()
    reader.create_feature(args.eval_dir)
    evaluation_metric = evaluation()

    en2en = BARTModel.from_pretrained('/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/checkpoints/para/bart/bart.base', checkpoint_file='checkpoint_2_105000.pt').cuda().eval()


    bert_substitutes_all=[]
    real_prev_scores_all=[]
    real_embed_scores_all=[]
    count_gen=-1

    from tqdm import tqdm
    for main_word in tqdm(reader.words_candidate):
        count_gen+=1

        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:
                text = context[1]
                original_text = text
                original_words = text.split(' ')
                index_word = int(context[2])
                target_word = text.split(' ')[index_word]

                prefix = " ".join(original_words[0:index_word]).strip()
                target_pos = main_word.split('.')[-1]
                # suffix = " ".join(original_words[index_word + 1:]).strip()

                target_lemma = lemma_word(
                    target_word,
                    target_pos=target_pos
                ).lower().strip()    

                first_target_word=target_word

                if index_word>0 and len(original_words[index_word-1])!=1 and original_words[index_word-1].endswith("."):
                    if original_words[index_word-1]=="u.s.":
                        print("change the u.s.!!!")
                        text=" ".join((original_words[:index_word-1]+original_words[index_word:])).strip()
                        original_text=text
                        original_words=text.split()
                        index_word=index_word-1
                        target_word=original_words[index_word]
                        #target_word=text.split()[index_word]
                        prefix = " ".join(original_words[0:index_word]).strip()
                        target_pos = main_word.split('.')[-1]
                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()  
                        if first_target_word!=target_word:
                            print("finding an error in the main fucntion!!!")
                            import pdb
                            pdb.set_trace()

                    elif index_word>2 and original_words[index_word-1]=="n." and original_words[index_word-2]=="u.":
                        print("changeing the u. n.!!!")
                        text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                        original_text=text
                        original_words=text.split()
                        index_word=index_word-2
                        target_word=original_words[index_word]
                        #target_word=text.split()[index_word]
                        target_pos = main_word.split('.')[-1]
                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()  
                        if first_target_word!=target_word:
                            print("finding an error in the main fucntion!!!")
                            import pdb
                            pdb.set_trace()                        
                    else:
                        original_words[index_word-1]=original_words[index_word-1][:-1]
                        text=" ".join(original_words)  
                        original_text = text
                        original_words = text.split(' ')
                        index_word = index_word
                        #target_word = text.split(' ')[index_word]
                        target_word=original_words[index_word]

                        prefix = " ".join(original_words[0:index_word]).strip()
                        target_pos = main_word.split('.')[-1]
                        # suffix = " ".join(original_words[index_word + 1:]).strip()

                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()    

                        if first_target_word!=target_word:
                            print("finding an error in the main fucntion!!!")
                            import pdb
                            pdb.set_trace()

                if index_word>0 and (original_words[index_word-1]=="t" or original_words[index_word-1]=="-"):
                    print("change the --")
                    text=" ".join((original_words[:index_word-1]+original_words[index_word:])).strip()
                    original_text=text
                    original_words=text.split()
                    index_word=index_word-1
                    #target_word=original_words[insdex_word]
                    target_word = original_words[index_word]
                    prefix = " ".join(original_words[0:index_word]).strip()
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()  
                    if first_target_word!=target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace() 
#hadn n't
                if index_word>1 and (original_words[index_word-1]=="'t" and original_words[index_word-2]=="don"):
                    print("change the don 't!!!")
                    #text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word-1]="not"
                    original_words[index_word-2]="do"
                    text==" ".join(original_words)

                    original_text=text
                    original_words=text.split()
                    target_word=original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()  
                    if first_target_word!=target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()



                if index_word>1 and (original_words[index_word-1]=="n't" and (original_words[index_word-2]=="couldn")):
                    print("change the could 't!!!")
                    #text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word-1]="not"
                    original_words[index_word-2]="could"
                    text==" ".join(original_words)

                    original_text=text
                    original_words=text.split()
                    target_word=original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()  
                    if first_target_word!=target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()



                if index_word>1 and (original_words[index_word-1]=="n't" and original_words[index_word-2]=="don"):
                    print("change the don n't!!!")
                    #text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word-1]="not"
                    original_words[index_word-2]="do"
                    text==" ".join(original_words)

                    original_text=text
                    original_words=text.split()
                    target_word=original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()  
                    if first_target_word!=target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()


                if index_word+2<len(original_words) and (original_words[index_word+1]=="hadn" and original_words[index_word+2]=="n't"):
                    print("change the hadn 't")
                    #text=" ".join((original_words[:index_word-2]+original_words[index_word:])).strip()
                    original_words[index_word+1]="had"
                    original_words[index_word+2]="not"
                    text==" ".join(original_words)

                    original_text=text
                    original_words=text.split()
                    target_word=original_words[index_word]
                    target_pos = main_word.split('.')[-1]
                    target_lemma = lemma_word(
                        target_word,
                        target_pos=target_pos
                    ).lower().strip()  
                    if first_target_word!=target_word:
                        print("finding an error in the main fucntion!!!")
                        import pdb
                        pdb.set_trace()

                bert_substitutes, bert_rank_substitutes,real_prev_scores,real_embed_scores = lexicalSubstitute(en2en, original_text,original_words,prefix,index_word,target_word,target_pos,target_lemma,args.beam, args.bertscore)
                
                bert_substitutes_all.append(bert_substitutes)
                real_prev_scores_all.append(real_prev_scores)
                real_embed_scores_all.append(real_embed_scores)
            


    import numpy as np
    import copy
    import os

    range1=np.arange(0.02,0.04,0.02)
    range2_log_softmax=np.arange(1,2,1)
    for log_quto in range2_log_softmax:
        work_dir="ls07_search_results.bart/log."+str(round(log_quto,2))+"/"
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        for embed_quto in range1:
            count_gen2=-1
            count_1=0
            tmp_bert_substitutes_all=copy.deepcopy(bert_substitutes_all)
            tmp_real_prev_scores_all=copy.deepcopy(real_prev_scores_all)
            tmp_real_embed_scores_all=copy.deepcopy(real_embed_scores_all)

            for main_word in tqdm(reader.words_candidate):
                count_gen2+=1

                for instance in reader.words_candidate[main_word]:
                    for context in reader.words_candidate[main_word][instance]:
                        text = context[1]
                        original_text = text
                        original_words = text.split(' ')
                        index_word = int(context[2])
                        target_word = text.split(' ')[index_word]

                        prefix = " ".join(original_words[0:index_word]).strip()
                        target_pos = main_word.split('.')[-1]
                        # suffix = " ".join(original_words[index_word + 1:]).strip()

                        target_lemma = lemma_word(
                            target_word,
                            target_pos=target_pos
                        ).lower().strip()      

                        tmp_log_embed_scores=torch.tensor(tmp_real_embed_scores_all[count_1])

                        tmp_log_embed_scores=tmp_log_embed_scores.tolist()

                        for k1 in range(len(tmp_real_prev_scores_all[count_1])):
                            tmp_real_prev_scores_all[count_1][k1]=embed_quto*tmp_real_prev_scores_all[count_1][k1]+tmp_log_embed_scores[k1]
                            pass
                        
                        write_all_results(main_word, instance, target_pos, work_dir+args.output_SR_file+".embed."+str(embed_quto),
                                        tmp_bert_substitutes_all[count_1], tmp_real_prev_scores_all[count_1], evaluation_metric)



                        count_1+=1  

            test_golden_file="data/LS07/test/lst_test.gold"
            output_results=work_dir+args.output_SR_file+".embed."+str(embed_quto)
            results_file=work_dir+args.output_score_file+".embed."+str(embed_quto)
            evaluation_metric.calculation_perl(
                test_golden_file,
                output_results + ".best",
                output_results + ".oot",
                results_file + ".best",
                results_file + ".oot"
            )
            evaluation_metric.calculation_p1(
                test_golden_file,
                output_results + "_p1.txt",
                results_file + "_p1.txt"
            )
            
            evaluation_metric.calculation_p3(
                test_golden_file,
                output_results + "_p3.txt",
                results_file + "_p3.txt"
            )
if __name__ == "__main__":
    main()
