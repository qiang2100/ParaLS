#!/usr/bin/python
# -*- coding: UTF-8 -*-

from operator import index
import os
from pyexpat import model
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import csv
import logging
import os
import random
import math
import sys
import re

from sklearn.metrics.pairwise import cosine_similarity as cosine
import string
punctuation = string.punctuation
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
# from reader import Reader_lexical
# from metrics.evaluation import evaluation

sys.path.append("/home/yz/liukang/liukang/fairseq-zh-en-master")
sys.path.append(os.getcwd())
from fairseq.metrics_ls07.evaluation import evaluation

import unicodedata


class Reader_lexical:
    def __init__(self):
        self.words_candidate = {}
        self.final_data = {}
        self.final_data_id = {}

    def create_feature(self, file_train):
        # side.n	303	11	if you want to find someone who can compose the biblical side , write us .
        with open(file_train, encoding='latin1') as fp:
            line = fp.readline()
            i = 0
            while line:
                context = line.split("\t")
                main_word = context[0]
                if main_word.split('.')[0] == "":
                    word = "."
                else:
                    word = main_word.split('.')[0]
                instance = context[1]
                word_index = context[2]
                sentence = self._clean_text(context[3].replace("\n", ""))
                if main_word not in self.words_candidate:
                    self.words_candidate[main_word] = {}
                if instance not in self.words_candidate[main_word]:
                    self.words_candidate[main_word][instance] = []
                self.words_candidate[main_word][instance].append([word, sentence, word_index])
                line = fp.readline()
        return

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self._is_control(char):
                continue
            if self._is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def create_candidates(self, file_path_candidate):
        with open(file_path_candidate, encoding='latin1') as fp:
            line = fp.readline()
            self.candidates = {}
            while line:
                word = line.split("::")[0]
                # if word == "..N":
                self.candidates[word] = []
                candidates_words = line.split("::")[1]
                for candidate_word in candidates_words.split(";"):
                    if ((len(candidate_word.split(' ')) > 1) or (len(candidate_word.split('-')) > 1)) or len(
                            candidate_word) < 1:
                        pass
                    else:
                        self.candidates[word].append(candidate_word.replace("\n", ""))
                line = fp.readline()
        return

    def read_eval_line(self, eval_line, ignore_mwe=True):
        eval_weights = {}
        segments = eval_line.split("\t")
        instance_id = segments[1].strip()
        for candidate_weight in segments[2:]:
            if len(candidate_weight) > 0:
                delimiter_ind = candidate_weight.rfind(' ')
                candidate = candidate_weight[:delimiter_ind]
                weight = candidate_weight[delimiter_ind:]
                if ignore_mwe and ((len(candidate.split(' ')) > 1) or (len(candidate.split('-')) > 1)):
                    continue
                try:
                    eval_weights[candidate] = float(weight)
                    # eval_weights.append((candidate, float(weight)))
                except:
                    print("Error appending: %s %s" % (candidate, weight))

        return instance_id, eval_weights

    def created_dict_proposed(self, proposed_words_gap, proposed_words_scores):
        proposed_words = {}
        for i in range(0, len(proposed_words_gap)):
            proposed_words[proposed_words_gap[i]] = proposed_words_scores[i]

        return proposed_words


# import gensim
# from gensim.test.utils import datapath,get_tmpfile
# from gensim.scripts.glove2word2vec import glove2word2vec
# from gensim.models import KeyedVectors
# wordVecPath = "/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/checkpoints/glove/glove.6B.300d.txt"
# glove_file = datapath(wordVecPath)
# tmp_file = get_tmpfile('glove_word2vec.txt')
# glove2word2vec(glove_file,tmp_file)
# glove_model = KeyedVectors.load_word2vec_format(tmp_file)


# import json
# word_pos_fp="../../Gloss/LS_infer/vocab/word_pos.json"
# with open(word_pos_fp,"r") as f:
#     pos_vocab = json.loads( f.read().strip() )

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
    if (precision+recall)==0:    
        #F_score = 2*precision*recall/(precision+recall)
        F_score=0
    else:
        F_score=(2*precision*recall)/(precision+recall)


    return potential,precision,recall,F_score




#scorer = BERTScorer(model_type="bert-base-uncased",lang="en", rescale_with_baseline=True)
#scorer = BERTScorer(lang="en", rescale_with_baseline=True)
scorer=None
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
    
    # except:
    #     print("complex word not in the glove model")

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
    # print(outputs)
    # print(scores_with_suffix[:,0:5])
    # for i in range(first_index,last_index):
    #     pass
        #scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
        #scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
    # print(scores_with_suffix[:,0:5])
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
        if "<unk>" in tmp_str1:
            tmp_str1=tmp_str1.replace("<unk>","|")

        #tmp_str1=tmp_str1.replace("<unk>","")
        #if len(tmp_str1.strip().split())==hope_len:
        if len(nltk.word_tokenize(tmp_str1))==hope_len:
            candis.append(nltk.word_tokenize(tmp_str1)[-1].strip())
            hope_outputs.append(outputs[beam_idx[i]])
            #print(tgt_dict.string(outputs[beam_idx[i]]),sorted_scores[i])
            hope_outputs_scores.append(sorted_scores[i].tolist())
        elif hope_len==-1:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())
        # if len(tmp_str1.split())==len(prefix_str.split())+1:
        #     print(tmp_str1)
    #print("&"*100)
    # import pdb
    # pdb.set_trace()
    return hope_outputs,hope_outputs_scores,candis



qutos=["<unk>",'']
def gen_gpt2_string(tgt_dict,input):
    if type(input)==int:
        input=torch.tensor([input])
    return tgt_dict.bpe.decode(tgt_dict.task.source_dictionary.string(input))

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

    # print(scores_with_suffix[:,0:5])

    # print(outputs)
    # print(scores_with_suffix[:,0:5])
    # for i in range(first_index,last_index):
    #     pass
        #scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
        #scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
    # print(scores_with_suffix[:,0:5])
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
        #tmp_str1=tgt_dict.string(outputs[beam_idx[i],:(len_idx[i]+1)]).replace("@@ ","")
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
        # if len(tmp_str1.split())==len(prefix_str.split())+1:
        #     print(tmp_str1)
    #print("&"*100)
    # import pdb
    # pdb.set_trace()
    return hope_outputs,hope_outputs_scores,candis



def give_real_scores_ahead_mbart(tgt_dict,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=None,prefix_str=None,max_ahead=1,flag=1):

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
    #ahead_part_tokens=list(map(lambda x:gen_gpt2_string(tgt_dict,x),ahead_parts))
    ahead_part_tokens=list(map(lambda x:tgt_dict.tgt_dict.string([x]),ahead_parts))
    ahead_part_tokens_masks=list(map(lambda x:not x.startswith("▁") and x not in qutos,ahead_part_tokens))
    ahead_part_tokens_masks=torch.tensor(ahead_part_tokens_masks).reshape(beam_size,-1)
    scores_with_suffix[:,:-1][ahead_part_tokens_masks[:,1:]]=-math.inf


    if first_index>1:
        ids_after_prefix=outputs[:,first_index+1]
        ids_after_prefix=ids_after_prefix.reshape(1,-1)[0].tolist()
        ids_after_prefix_tokens=list(map(lambda x:tgt_dict.tgt_dict.string([x]),ids_after_prefix))
        ids_after_prefix_tokens_masks=list(map(lambda x:not x.startswith("▁") and x not in qutos,ids_after_prefix_tokens))
        ids_after_prefix_tokens_masks=torch.tensor(ids_after_prefix_tokens_masks).reshape(beam_size,-1)
        scores_with_suffix[ids_after_prefix_tokens_masks.squeeze()]=-math.inf

    # print(scores_with_suffix[:,0:5])

    # print(outputs)
    # print(scores_with_suffix[:,0:5])
    # for i in range(first_index,last_index):
    #     pass
        #scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
        #scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
    # print(scores_with_suffix[:,0:5])

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
        hope_len=len(spanish_tok.tokenize(prefix_str))+flag
        #hope_len=len(nltk.word_tokenize(prefix_str))+flag
        #hope_len=len(prefix_str.strip().split())+flag
    else:
        hope_len=-1

    hope_outputs=[]
    hope_outputs_scores=[]
    candis=[]
    for i in range(len(beam_idx)):
        # if sorted_scores[i]==(-math.inf):
        #     continue
        
        #tmp_str1=tgt_dict.bpe.decode(tgt_dict.task.source_dictionary.string(outputs[beam_idx[i],:(len_idx[i]+1)])).strip()
        tmp_str1=tgt_dict.tgt_dict.string(outputs[beam_idx[i],:(len_idx[i]+1)]).replace(" ","").replace("▁"," ").strip()
        tmp_str1=skip_words(tmp_str1)
        #tmp_str1=tgt_dict.string(outputs[beam_idx[i],:(len_idx[i]+1)]).replace("@@ ","")
        #tmp_str1=tmp_str1.replace("<unk>","")
        #if len(tmp_str1.strip().split())==hope_len:

        # if "<unk>" in tmp_str1:
        #     print("finding a unk in prefix str")
        #     tmp_str1=tmp_str1.replace("<unk>","|")

        if len(spanish_tok.tokenize(tmp_str1))==hope_len:
            #candis.append(nltk.word_tokenize(tmp_str1)[-1].strip())
            candis.append((" "+spanish_tok.tokenize(tmp_str1)[-1]).strip())
            hope_outputs.append(outputs[beam_idx[i]])
            #print(tgt_dict.string(outputs[beam_idx[i]]),sorted_scores[i])
            hope_outputs_scores.append(sorted_scores[i].tolist())
        elif hope_len==-1:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())
            tmp_candi=nltk.word_tokenize(tmp_str1)[len(nltk.word_tokenize(prefix_str)):]
            tmp_candi=" ".join(tmp_candi).strip()
            candis.append(tmp_candi)
        # if len(tmp_str1.split())==len(prefix_str.split())+1:
        #     print(tmp_str1)
    #print("&"*100)
    # import pdb
    # pdb.set_trace()

    return hope_outputs,hope_outputs_scores,candis





def extract_substitute_bart(output_sentences, original_sentence, complex_word, threshold,prev_scores=None,word_index=None,sentence_words=None,target_pos=None,target_lemma=None):

    original_words = sentence_words

    index_of_complex_word = -1

    # if complex_word  not in original_words:
    #     i = 0
    #     for word in original_words:
    #         if complex_word == word.lower():
    #             index_of_complex_word = i
    #             break
    #         i += 1
    # else:
    #     index_of_complex_word = original_words.index(complex_word)
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
        # bert_scores = substitutes_BertScore(context, complex_word, substitutes)

        # #print(substitutes)
        # bert_scores = bert_scores.tolist()
        
        # #pdb.set_trace()


        # filter_substitutes, bert_scores = filterSubstitute(substitutes, bert_scores, threshold)

        # rank_bert = sorted(bert_scores,reverse = True)

        # rank_bert_substitutes = [filter_substitutes[bert_scores.index(v)] for v in rank_bert]
        filter_substitutes=substitutes
        rank_bert_substitutes=substitutes

        assert len(filter_substitutes)==len(real_prev_scores)
        assert len(filter_substitutes)==len(rank_bert_substitutes)

        return filter_substitutes, rank_bert_substitutes,real_prev_scores

    return [],[],[]
















def extract_substitute(output_sentences, original_sentence, complex_word, threshold,prev_scores=None,word_index=None,sentence_words=None,target_pos=None,target_lemma=None):

    original_words = sentence_words

    index_of_complex_word = -1

    # if complex_word  not in original_words:
    #     i = 0
    #     for word in original_words:
    #         if complex_word == word.lower():
    #             index_of_complex_word = i
    #             break
    #         i += 1
    # else:
    #     index_of_complex_word = original_words.index(complex_word)
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
    not_candi = {'the', 'with', 'of', 'a', 'an', 'for', 'in', "-", "``", "*", "\"","it",",",".","...","?","!",":"}
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
                or not_index_1 != -1:
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
        # bert_scores = substitutes_BertScore(context, complex_word, substitutes)

        # #print(substitutes)
        # bert_scores = bert_scores.tolist()
        
        # #pdb.set_trace()
        

        # filter_substitutes, bert_scores = filterSubstitute(substitutes, bert_scores, threshold)

        # rank_bert = sorted(bert_scores,reverse = True)

        # rank_bert_substitutes = [filter_substitutes[bert_scores.index(v)] for v in rank_bert]
        # assert len(filter_substitutes)==len(real_prev_scores)
        # assert len(filter_substitutes)==len(rank_bert_substitutes)
        # assert len(filter_substitutes)==len(substitutes)


        filter_substitutes=substitutes
        rank_bert_substitutes=substitutes

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


def lexicalSubstitute(model, sentence, sentence_words, prefix,word_index,complex_word,target_pos,target_lemma,beam, threshold):
    index_complex = word_index
    ori_words=sentence_words
    prefix = prefix
    suffix1 = ""
    if(index_complex != -1):
        prefix = prefix
        if len(ori_words)>index_complex+1:
            suffix1=" ".join(ori_words[index_complex+1:]).strip()
            # suffix1=suffix1.replace("''","\"").strip()
            # suffix1=suffix1.replace("``","\"").strip()
                        
            # suffix1=process_string(suffix1)
            # #stored_suffix1=suffix1

            # if suffix1.endswith("\""):
            #     suffix1=suffix1[:-1]
            #     suffix1=suffix1.strip()
            # if suffix1.endswith("'"):
            #     suffix1=suffix1[:-1]
            #     suffix1=suffix1.strip()    

            suffix1=" ".join(suffix1.split(" ")[:2])
            # if "," in suffix1:
            #     if suffix1.index(",")!=0:
            #         suffix1=suffix1[:suffix1.index(",")]
            #suffix1 = sentence[index_complex+:index_complex+1].strip()
            # suffix1 = " ".join(ori_words[ori_words.index(complex_word)+1:ori_words.index(complex_word)+7])
            # suffix1=process_string(suffix1)
            # medium_qutos=[",",".","!","?","\"","``",""]
            # for char1 in suffix1:

        else:
            pass
        #print(prefix)
    else:
        print("*************cannot find the complex word")
        #print(sentence)
        #print(complex_word)
        sentence = sentence.lower()

        return lexicalSubstitute(model, sentence, complex_word,  beam, threshold)
    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)

    complex_tokens = model.encode(complex_word)
    #1.make some change to the original sentence
    #=prefix.strip()+" "+process_string(complex_word.strip()+" "+stored_suffix1.strip())
    #sentence=new_sentence


    sentence_tokens = model.encode(sentence)

    suffix_tokens=model.encode(suffix1)[:-1]
    #suffix_tokens=torch.tensor(suffix_tokens)
    suffix_tokens=suffix_tokens.tolist()
    attn_len = len(prefix_tokens[0])+len(complex_tokens)-1
    if len((model.tgt_dict.string(prefix_tokens).strip().replace("@@ ","")).strip().split())!=len(prefix.strip().split()):
        print("finding prefix not good before replace mask token!!!")
        # if len((model.tgt_dict.string(prefix_tokens).strip().replace("@@ ","")).strip().replace("<unk>",""))!=len(prefix.strip().split()):
        #     print("finding prefix not good!!!")
    #outputs = model.generate2(sentence_tokens, beam=20, prefix_tokens=prefix_tokens)
    # outputs,pre_scores = model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len)
    #outputs,pre_scores = model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len,suffix_ids=suffix_tokens) 
    outputs,combined_sss,prev_masks,prev_masks2,scores_with_suffix,scores_with_suffix_masks,_= model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len,suffix_ids=suffix_tokens,max_aheads=5)   
    outputs=outputs.cpu()
    
    # for i in range(len(combined_sss)):
    #     if combined_sss[i]!=[]:
    #         if type(combined_sss[i])==list:
    #             combined_sss[i][0]=combined_sss[i][0].to("cpu")
    #             combined_sss[i][1]=combined_sss[i][1].to("cpu")
    #         else:
    #             combined_sss[i]=combined_sss[i].to("cpu")
    # prev_masks=prev_masks.cpu()
    # prev_masks2=prev_masks2.cpu()
    scores_with_suffix=scores_with_suffix.cpu()
    scores_with_suffix_masks=scores_with_suffix_masks.cpu()

    # output_final_scores=give_real_scores(combined_sss,prev_masks,prev_masks2,suffix_tokens)
    # # import pdb
    # # pdb.set_trace()

    # if combined_sss[1]!=[]:
    #     # print("123")
    #     outputs=outputs[torch.squeeze(torch.topk(output_final_scores,k=combined_sss[0][0].shape[1],dim=1)[1].view(1,-1),1)][0]
    # else:
    #     outputs=outputs[torch.squeeze(torch.topk(combined_sss[0][0],k=combined_sss[0][0].shape[1],dim=1)[1].view(1,-1),1)][0]
    #embed_scores=give_embedding_scores(outputs,model.models[0].state_dict()["decoder.embed_tokens.weight"].cpu(),complex_tokens=complex_tokens,temperature=0.2)
    #embed_scores=give_embedding_scores_v2(outputs,model.models[0].state_dict()["decoder.embed_tokens.weight"].cpu(),complex_tokens=complex_tokens,temperature=0.2)
    #assert embed_scores.size()==scores_with_suffix[:,:(outputs.size()[-1]-1)].size()
    # alkl make change the embedding scores
    #embed_scores=change_embedding_scores(outputs,embed_scores,prefix_len=len(prefix_tokens[0]),max_ahead=5)
    #scores_with_suffix[:,:(outputs.size()[-1]-1)]=scores_with_suffix[:,:(outputs.size()[-1]-1)]+embed_scores

    outputs,outputs_scores,candis=give_real_scores_ahead(model.tgt_dict,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=len(prefix_tokens[0]),prefix_str=prefix,max_ahead=5,flag=1)


    #glove_scores=get_glove_embedding(complex_word,candis,glove_model,temperature=1)
    new_outputs_scores=torch.tensor(outputs_scores)
    #new_outputs_scores=(torch.tensor(outputs_scores)+glove_scores)
    outputs_scores=new_outputs_scores
    new_indices=torch.topk(outputs_scores,k=len(outputs_scores),dim=0)[1]
    outputs=[outputs[index1] for index1 in new_indices]
    # try:
    # assert new_indices[0]==0
    # assert new_indices[-1]==len(new_indices)-1
    # except:
    #     print("&&&",sentence)
    outputs_scores=outputs_scores.tolist()

    #print(outputs)

    #outputs=outputs[torch.squeeze(torch.topk(output_final_scores,k=beam,dim=1)[-1].view(1,-1),0)][:50]

    #output_sentences = [model.decode(x['tokens']) for x in outputs]
    output_sentences=[model.decode(x) for x in outputs]
    if output_sentences==[]:
        print("find a missing prefix sentence!!!")
        return [],[],[]
    # for s1 in output_sentences:
    #     print(s1[:200])
    # for s1 in outputs:
    #     print(model.tgt_dict.string(s1)[:150])   
    #bertscore_substitutes, ranking_bertscore_substitutes = extract_substitute(output_sentences, sentence, complex_word, threshold)
    bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores = extract_substitute(output_sentences, sentence, complex_word,
                                                                              threshold,outputs_scores,word_index,sentence_words,target_pos,target_lemma)
    #print(pre_scores)

    #for sen in output_sentences:
    #    print(sen)

    #bertscore_substitutes, ranking_bertscore_substitutes = extractSubstitute_bertscore(output_sentences, sentence, complex_word, threshold)
    #suffix_substitutes = extractSubstitute_suffix(output_sentences, sentence, complex_word)

    return bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores



def lexicalSubstitute_bart(model, sentence, sentence_words, prefix,word_index,complex_word,target_pos,target_lemma,beam, threshold):
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

            suffix1=" ".join(suffix1.split(" ")[:2])
            # if "," in suffix1:
            #     if suffix1.index(",")!=0:
            #         suffix1=suffix1[:suffix1.index(",")]
            #suffix1 = sentence[index_complex+:index_complex+1].strip()
            # suffix1 = " ".join(ori_words[ori_words.index(complex_word)+1:ori_words.index(complex_word)+7])
            # suffix1=process_string(suffix1)
            # medium_qutos=[",",".","!","?","\"","``",""]
            # for char1 in suffix1:

        else:
            pass
        #print(prefix)
    else:
        print("*************cannot find the complex word")
        #print(sentence)
        #print(complex_word)
        sentence = sentence.lower()

        return lexicalSubstitute(model, sentence, complex_word,  beam, threshold)
    
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
    

    #1.make some change to the original sentence
    #=prefix.strip()+" "+process_string(complex_word.strip()+" "+stored_suffix1.strip())
    #sentence=new_sentence

    if index_complex!=0:
        sentence_tokens = model.encode(sentence)
    else:
        sentence_tokens = model.encode(" "+sentence)

    if suffix1!="":
        suffix_tokens=model.encode(" "+suffix1.strip())[1:-1]
    else:
        suffix_tokens=model.encode(suffix1.strip())[1:-1]
    #suffix_tokens=torch.tensor(suffix_tokens)
    suffix_tokens=suffix_tokens.tolist()
    attn_len = len(prefix_tokens[0])+len(complex_tokens)-1



    # if len((model.tgt_dict.string(prefix_tokens).strip().replace("@@ ","")).strip().split())!=len(prefix.strip().split()):
    #     print("finding prefix not good before replace mask token!!!")
        # if len((model.tgt_dict.string(prefix_tokens).strip().replace("@@ ","")).strip().replace("<unk>",""))!=len(prefix.strip().split()):
        #     print("finding prefix not good!!!")
    #outputs = model.generate2(sentence_tokens, beam=20, prefix_tokens=prefix_tokens)
    # outputs,pre_scores = model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len)
    #outputs,pre_scores = model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len,suffix_ids=suffix_tokens) 
    outputs,combined_sss,prev_masks,prev_masks2,scores_with_suffix,scores_with_suffix_masks,_= model.generate3(
                                                                                                [sentence_tokens.cuda()], 
                                                                                                beam=beam, 
                                                                                                prefix_tokens=prefix_tokens.cuda(),
                                                                                                attn_len=attn_len,
                                                                                                tgt_token=-1,
                                                                                                suffix_ids=suffix_tokens,
                                                                                                max_aheads=5)   
    outputs=outputs[0].cpu()
    
    # for i in range(len(combined_sss)):
    #     if combined_sss[i]!=[]:
    #         if type(combined_sss[i])==list:
    #             combined_sss[i][0]=combined_sss[i][0].to("cpu")
    #             combined_sss[i][1]=combined_sss[i][1].to("cpu")
    #         else:
    #             combined_sss[i]=combined_sss[i].to("cpu")
    # prev_masks=prev_masks.cpu()
    # prev_masks2=prev_masks2.cpu()
    scores_with_suffix=scores_with_suffix.cpu()
    scores_with_suffix_masks=scores_with_suffix_masks.cpu()

    # output_final_scores=give_real_scores(combined_sss,prev_masks,prev_masks2,suffix_tokens)
    # # import pdb
    # # pdb.set_trace()

    # if combined_sss[1]!=[]:
    #     # print("123")
    #     outputs=outputs[torch.squeeze(torch.topk(output_final_scores,k=combined_sss[0][0].shape[1],dim=1)[1].view(1,-1),1)][0]
    # else:
    #     outputs=outputs[torch.squeeze(torch.topk(combined_sss[0][0],k=combined_sss[0][0].shape[1],dim=1)[1].view(1,-1),1)][0]
    #embed_scores=give_embedding_scores(outputs,model.models[0].state_dict()["decoder.embed_tokens.weight"].cpu(),complex_tokens=complex_tokens,temperature=0.2)
    #embed_scores=give_embedding_scores_v2(outputs,model.models[0].state_dict()["decoder.embed_tokens.weight"].cpu(),complex_tokens=complex_tokens,temperature=0.2)
    #assert embed_scores.size()==scores_with_suffix[:,:(outputs.size()[-1]-1)].size()
    # alkl make change the embedding scores
    #embed_scores=change_embedding_scores(outputs,embed_scores,prefix_len=len(prefix_tokens[0]),max_ahead=5)
    #scores_with_suffix[:,:(outputs.size()[-1]-1)]=scores_with_suffix[:,:(outputs.size()[-1]-1)]+embed_scores
    
    outputs,outputs_scores,candis=give_real_scores_ahead_bart(model,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=len(prefix_tokens[0]),prefix_str=prefix,max_ahead=5,flag=1)


    #glove_scores=get_glove_embedding(complex_word,candis,glove_model,temperature=1)
    new_outputs_scores=torch.tensor(outputs_scores)
    #new_outputs_scores=(torch.tensor(outputs_scores)+glove_scores)
    outputs_scores=new_outputs_scores
    new_indices=torch.topk(outputs_scores,k=len(outputs_scores),dim=0)[1]
    outputs=[outputs[index1] for index1 in new_indices]
    # try:
    # assert new_indices[0]==0
    # assert new_indices[-1]==len(new_indices)-1
    # except:
    #     print("&&&",sentence)
    outputs_scores=outputs_scores.tolist()

    #print(outputs)

    #outputs=outputs[torch.squeeze(torch.topk(output_final_scores,k=beam,dim=1)[-1].view(1,-1),0)][:50]

    #output_sentences = [model.decode(x['tokens']) for x in outputs]
    output_sentences=[model.bpe.decode(model.task.source_dictionary.string(x)) for x in outputs]
    if output_sentences==[]:
        print("find a missing prefix sentence!!!")
        return [],[],[]
    # for s1 in output_sentences:
    #     print(s1[:200])
    # for s1 in outputs:
    #     print(model.tgt_dict.string(s1)[:150])   
    #bertscore_substitutes, ranking_bertscore_substitutes = extract_substitute(output_sentences, sentence, complex_word, threshold)
    bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores = extract_substitute_bart(output_sentences, sentence, complex_word,
                                                                              threshold,outputs_scores,word_index,sentence_words,target_pos,target_lemma)
    #print(pre_scores)

    #for sen in output_sentences:
    #    print(sen)

    #bertscore_substitutes, ranking_bertscore_substitutes = extractSubstitute_bertscore(output_sentences, sentence, complex_word, threshold)
    #suffix_substitutes = extractSubstitute_suffix(output_sentences, sentence, complex_word)

    return bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores












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
    words=[]
    scores=[]
    for substitute_str, score in zip(substitutes, substitutes_scores):
        substitute_lemma = lemma_word(
            substitute_str,
            target_pos
        ).lower().strip()
        max_score = proposed_words.get(substitute_lemma)
        if max_score is None or score > max_score:
            #if pos_filter(pos_vocab,target_pos,substitute_str,substitute_lemma):

            proposed_words[substitute_lemma] = score

    for key1 in proposed_words.keys():
        words.append(key1)
        scores.append(proposed_words[key1])

    assert len(words)==len(scores)
    new_indices=torch.topk(torch.tensor(scores),k=len(scores))[-1].tolist()
    new_words=[words[index1] for index1 in new_indices]
    new_scores=[scores[index1] for index1 in new_indices]
    return new_words,new_scores



def write_all_results_oot(main_word, instance, target_pos, output_results, substitutes, substitutes_scores,
                      evaluation_metric):
    proposed_words = {}
    for substitute_str, score in zip(substitutes, substitutes_scores):
        substitute_lemma = lemma_word(
            substitute_str,
            target_pos
        ).lower().strip()
        max_score = proposed_words.get(substitute_lemma)
        if max_score is None or score > max_score:
            proposed_words[substitute_lemma] = score

    evaluation_metric.write_results_lex_oot(
        output_results + ".oot",
        main_word, instance,
        proposed_words, limit=10
    )
    
    evaluation_metric.write_results_lex_best(
        output_results + ".best",
        main_word, instance,
        proposed_words, limit=1
    )

    evaluation_metric.write_results_p1(
        output_results + "_p1.txt",
        main_word, instance,
        proposed_words
    )

def get_ls07_labels_dict():

    labels_dict={}
    for line1 in open("/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/data/LS07/test/lst_test.gold",encoding="utf-8"):
        key1=line1.strip().split("::")[0].strip()
        #labels=line1.strip().split("::")[1].strip()[:-1].split(";")
        labels=line1.strip()
        if key1 not in labels_dict:
            labels_dict[key1]=labels
    return labels_dict
   
def gen_ls07(en2en,output_sr_file,eval_dir,beam=100,bertscore=-100,all_labels_dict=None):
    all_labels_dict=get_ls07_labels_dict()
    # test_golden_file="/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/data/LS07/test/lst_test.gold.tmp"
    # test_goldne_file_write=open(test_golden_file,"w+")
    reader = Reader_lexical()
    reader.create_feature(eval_dir)
    evaluation_metric = evaluation()
    # en2en = TransformerModel.from_pretrained(args.paraphraser_path, checkpoint_file=args.paraphraser_model,bpe=args.bpe,
    #                                          bpe_codes=args.bpe_codes).cuda().eval()

    #CS = []

    CS2 = []
    all_labels=[]
    #CS3 = []
    #complex_labels[100:130]+complex_labels[200:230]+complex_labels[300:330]+complex_labels[400:430]+complex_labels[600:630]+complex_labels[1000:1030]+complex_labels[1690:1720]
    #good_numbers=list(range(100,130))+list(range(200,230))+list(range(300,330))+list(range(400,430))+list(range(600,630))+list(range(1000,1030))+list(range(1690,1720))
    #good_numbers=list(range(100,130))+list(range(200,230))+list(range(300,330))+list(range(400,430))+list(range(600,630))+list(range(1000,1030))+list(range(1690,1720))
    good_numbers=[i for i in range(2000) if i%10==0 or i%10==9 or i%10==5 or i%10==3]
    #good_numbers=list(range(100,101))
    #output_sr_file.write("beam:", args.beam, " bertscore:", args.bertscore)
    #output_sr_file.write('\n')
    from tqdm import tqdm
    count_tmp=0
    for main_word in tqdm(reader.words_candidate):
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:
                if count_tmp not in good_numbers:
                    count_tmp+=1
                    continue
                
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
                bert_substitutes, bert_rank_substitutes,real_prev_scores = lexicalSubstitute(en2en, original_text,original_words,prefix,index_word,target_word,target_pos,target_lemma,beam, bertscore)

                write_all_results_oot(main_word, instance, target_pos, output_sr_file,
                                  bert_substitutes, real_prev_scores, evaluation_metric)
                # key1=main_word+" "+instance
                # if key1 in all_labels_dict:
                #     gold_line=all_labels_dict[key1]
                #     test_goldne_file_write.write(gold_line.strip()+"\n")
            


                count_tmp+=1
    test_golden_file="/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/data/LS07/test/lst_test.gold"
    output_results=output_sr_file+".oot"
    results_file=output_sr_file+".score"
    evaluation_metric.calculation_perl_oot(test_golden_file,output_sr_file + ".best",output_results,results_file + ".best",results_file + ".oot")    
    oot_results=open(results_file+".oot").readlines()
    oot_score=float(oot_results[-1].strip().split()[2].rstrip(","))



                # CS2.append(new_words[:10])
                # key1=main_word+" "+instance
                # complex_labels=all_labels_dict[key1]
                # all_labels.append(complex_labels)
                # # CS3.append(bert_rank_substitutes[:10])
                # #final_str=" ".join(complex_labels[i])+"|||"+" ".join(bert_rank_substitutes[:10])+"|||"+" ".join(list(set(complex_labels[i])&set(bert_rank_substitutes[:10])))+"\n"
                # final_str=";".join(complex_labels)+"|||"+" ".join(new_words[:10])+"|||"+" ".join(list(set(complex_labels)&set(new_words[:10])))+"\n"
                # count_tmp+=1
                # output_sr_file.write(final_str)
    # test_goldne_file_write.close()
    return oot_score
    # return 1


def gen_ls07_bart(en2en,output_sr_file,eval_dir,beam=100,bertscore=-100,all_labels_dict=None):
    all_labels_dict=get_ls07_labels_dict()
    # test_golden_file="/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/data/LS07/test/lst_test.gold.tmp"
    # test_goldne_file_write=open(test_golden_file,"w+")
    reader = Reader_lexical()
    reader.create_feature(eval_dir)
    evaluation_metric = evaluation()
    # en2en = TransformerModel.from_pretrained(args.paraphraser_path, checkpoint_file=args.paraphraser_model,bpe=args.bpe,
    #                                          bpe_codes=args.bpe_codes).cuda().eval()

    #CS = []

    CS2 = []
    all_labels=[]
    #CS3 = []
    #complex_labels[100:130]+complex_labels[200:230]+complex_labels[300:330]+complex_labels[400:430]+complex_labels[600:630]+complex_labels[1000:1030]+complex_labels[1690:1720]
    #good_numbers=list(range(100,130))+list(range(200,230))+list(range(300,330))+list(range(400,430))+list(range(600,630))+list(range(1000,1030))+list(range(1690,1720))
    #good_numbers=list(range(100,130))+list(range(200,230))+list(range(300,330))+list(range(400,430))+list(range(600,630))+list(range(1000,1030))+list(range(1690,1720))
    good_numbers=[i for i in range(2000) if i%10==0 or i%10==9 or i%10==5 or i%10==3]
    #good_numbers=range(2000)
    #good_numbers=list(range(100,101))
    #output_sr_file.write("beam:", args.beam, " bertscore:", args.bertscore)
    #output_sr_file.write('\n')
    from tqdm import tqdm
    count_tmp=0
    for main_word in tqdm(reader.words_candidate):
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:
                if count_tmp not in good_numbers:
                    count_tmp+=1
                    continue
                
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
                bert_substitutes, bert_rank_substitutes,real_prev_scores = lexicalSubstitute_bart(en2en, original_text,original_words,prefix,index_word,target_word,target_pos,target_lemma,beam, bertscore)

                write_all_results_oot(main_word, instance, target_pos, output_sr_file,
                                  bert_substitutes, real_prev_scores, evaluation_metric)
                # key1=main_word+" "+instance
                # if key1 in all_labels_dict:
                #     gold_line=all_labels_dict[key1]
                #     test_goldne_file_write.write(gold_line.strip()+"\n")
            


                count_tmp+=1
    test_golden_file="/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/data/LS07/test/lst_test.gold"
    output_results=output_sr_file+".oot"
    results_file=output_sr_file+".score"
    evaluation_metric.calculation_perl_oot(test_golden_file,output_sr_file + ".best",output_results,results_file + ".best",results_file + ".oot")    
    evaluation_metric.calculation_p1(
        test_golden_file,
        output_sr_file + "_p1.txt",
        results_file + "_p1.txt"
    ) 
    oot_results=open(results_file+".oot").readlines()
    oot_score=float(oot_results[-1].strip().split()[2].rstrip(","))



                # CS2.append(new_words[:10])
                # key1=main_word+" "+instance
                # complex_labels=all_labels_dict[key1]
                # all_labels.append(complex_labels)
                # # CS3.append(bert_rank_substitutes[:10])
                # #final_str=" ".join(complex_labels[i])+"|||"+" ".join(bert_rank_substitutes[:10])+"|||"+" ".join(list(set(complex_labels[i])&set(bert_rank_substitutes[:10])))+"\n"
                # final_str=";".join(complex_labels)+"|||"+" ".join(new_words[:10])+"|||"+" ".join(list(set(complex_labels)&set(new_words[:10])))+"\n"
                # count_tmp+=1
                # output_sr_file.write(final_str)
    # test_goldne_file_write.close()
    return oot_score



def gen_tsar_bart(en2en,output_sr_file,eval_dir,beam=100,bertscore=-100,all_labels_dict=None):

    import nltk
    for line in open("/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/data/tsar/tsar2022_en_test_gold.tsv"):
        count+=1
        print(count)

        original_text=line.strip().split("\t")[0]

        target_word=line.strip().split("\t")[1].strip()
        label_words=list(set(line.strip().split("\t")[2:]))



        prefix=original_text.split(target_word)[0].strip()
        prefix=" ".join(nltk.word_tokenize(prefix)).strip()
    
        suffix=original_text.split(target_word)[1].strip()
        suffix=" ".join(nltk.word_tokenize(suffix)).strip()

        tmp_original_text=prefix+" "+target_word+" "+suffix
        tmp_original_text=tmp_original_text.strip()

        bert_substitutes, bert_rank_substitutes,real_prev_scores,real_embed_scores = lexicalSubstitute(
                                                                            en2en, 
                                                                            tmp_original_text,
                                                                            prefix,
                                                                            suffix,
                                                                            target_word,
                                                                            args.beam)
        Outputs.append(bert_substitutes)
        f2.write(original_text+"\t"+target_word+"\t"+"\t".join(bert_substitutes)+"\n")

        f3.write(";".join(label_words)+"|||"+";".join(bert_substitutes)+"|||"+";".join(tp_words)+"\n")


from nltk.tokenize.toktok import ToktokTokenizer
spanish_tok = ToktokTokenizer()
def skip_words(word):
    word=word.replace("[es_XX]","")
    word=word.replace("<s>","")
    word=word.replace("</s>","")
    return word.strip()
from nltk.stem.snowball import SnowballStemmer
spanish_ps=SnowballStemmer("spanish")

def lexicalSubstitute_mbart_tsar(model, sentence,prefix,suffix,complex_word,beam):
    

    import copy

    prefix = prefix
    suffix1=" ".join(suffix.split()[:2]).strip()

    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)

    complex_tokens = model.encode(complex_word.strip())[:-1]

    sentence_tokens = model.encode(sentence)


    suffix_tokens=model.encode(suffix1.strip())[:-1]

    suffix_tokens=suffix_tokens.tolist()

    attn_len = len(prefix_tokens[0])+len(complex_tokens)

    outputs,combined_sss,prev_masks,prev_masks2,scores_with_suffix,scores_with_suffix_masks,scores_with_dynamic = model.generate3([sentence_tokens.cuda()], 
                                                                                                            beam=beam, 
                                                                                                            prefix_tokens=prefix_tokens.cuda(), 
                                                                                                            attn_len=attn_len,
                                                                                                            #tgt_token=complex_tokens[:-1].tolist(),
                                                                                                            tgt_token=-1,
                                                                                                            suffix_ids=suffix_tokens,
                                                                                                            max_aheads=5)   

    #outputs=outputs[0].cpu()
    outputs=outputs[0].cpu()
    scores_with_suffix=scores_with_suffix.cpu()
    scores_with_suffix_masks=scores_with_suffix_masks.cpu()

    outputs,outputs_scores,candis=give_real_scores_ahead_mbart(
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
    outputs_scores=new_outputs_scores
    new_indices=torch.topk(outputs_scores,k=len(outputs_scores),dim=0)[1]
    outputs=[outputs[index1] for index1 in new_indices]
    outputs_scores=[outputs_scores[index1].tolist() for index1 in new_indices]
    candis=[candis[index1] for index1 in new_indices]

    output_sentences=[model.tgt_dict.string(x).replace(" ","").replace("▁"," ").strip() for x in outputs]

    for i3 in range(len(output_sentences)):
        output_sentences[i3]=skip_words(output_sentences[i3])
    if output_sentences==[]:
        print("find a missing prefix sentence!!!")
        return [],[],[],[]



    bertscore_substitutes=[]
    ranking_bertscore_substitutes=[]
    real_prev_scores=[]
    complex_stem = spanish_ps.stem(complex_word)
    not_candi = {"de","que","no","a","la","el","es","en","y","lo","un","por","me","con","tú",\
                    "los","para","como","pero","su","si","al","le","se","del","este","como","esta",".","¡","¿",",","'",'"'}
    not_candi.add(complex_stem)
    not_candi.add(complex_word)

    try:
        not_candi.add(prefix.split()[-1])
    except:
        pass

    try:
        not_candi.add(suffix.split()[0])
    except:
        pass

    all_stored_cand=[]
    for i3 in range(len(candis)):
        candi=candis[i3].lower()
        if "," in candi or "." in candi or "'" in candi or '"' in candi or "," in candi:
            continue
        candi_list=spanish_tok.tokenize(candi)
        max_several=2
        now_several=0
        pointer_several=False
        all_stored_cand.append(candi)


        if len(candi_list)==1:
            candi_stem = spanish_ps.stem(candi)
            not_index_0 = candi.find("-")
            not_index_1 = candi.find(complex_word)
            if candi_stem in not_candi or candi in not_candi or not_index_0 != -1 \
                    or not_index_1 != -1 or candi[0] in punctuation or len(candi)==1 or candi[-1] in punctuation:
                continue

        if candi not in bertscore_substitutes:
            if pointer_several==False:
                bertscore_substitutes.append(candi)
                real_prev_scores.append(outputs_scores[i3])

            elif now_several<max_several:
                bertscore_substitutes.append(candi)
                real_prev_scores.append(outputs_scores[i3])
                now_several+=1
            else:
                pass   
        ranking_bertscore_substitutes=bertscore_substitutes 



    new_real_prev_scores=torch.tensor(real_prev_scores)
    new_indices=torch.topk(new_real_prev_scores,k=len(real_prev_scores),dim=0)[1]    
    bertscore_substitutes=[bertscore_substitutes[index1] for index1 in new_indices]
    ranking_bertscore_substitutes=bertscore_substitutes
    real_prev_scores=[real_prev_scores[index1] for index1 in new_indices]


    bertscore_substitutes=bertscore_substitutes[:10]
    ranking_bertscore_substitutes=ranking_bertscore_substitutes[:10]
    real_prev_scores=real_prev_scores[:10]

    return bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores,1



def gen_mbart_tsar(en2en,output_sr_file,eval_dir,beam=100,bertscore=-100,all_labels_dict=None):

    import nltk
    potential_1=0
    potential_2=0
    potential_3=0
    potential_5=0
    potential_10=0
    count1=0
    rec_all=0
    from tqdm import tqdm
    tsar_lines=open("/home/yz/liukang/liukang/fairseq-main_prefix/fairseq-main_prefix/data/tsar/tsar2022_es_test_gold.tsv").readlines()
    # new_tsar_lines=tsar_lines[:20]+tsar_lines[40:60]+tsar_lines[80:100]
    # new_tsar_lines+=(tsar_lines[110:130]+tsar_lines[150:170]+tsar_lines[180:200])
    # new_tsar_lines+=(tsar_lines[210:240]+tsar_lines[250:280])
    # new_tsar_lines+=tsar_lines[330:350]
    new_tsar_lines=[]
    bad_number=[0,2,3,4,5,7,8,12,13,15]
    for number1 in bad_number:
        new_tsar_lines+=tsar_lines[number1*20:number1*20+20]
    

    # new_tsar_lines=tsar_lines[:20]

    for line in tqdm(new_tsar_lines):
    
        count1+=1
        original_text=line.strip().split("\t")[0].strip()

        target_word=line.strip().split("\t")[1].strip()
        label_words=list(set(line.strip().split("\t")[2:]))



        prefix=original_text.split(target_word)[0].strip()
        prefix=" ".join(spanish_tok.tokenize(prefix)).strip()
    
        suffix=original_text.split(target_word)[1].strip()
        suffix=" ".join(spanish_tok.tokenize(suffix)).strip()

        tmp_original_text=prefix+" "+target_word+" "+suffix
        tmp_original_text=tmp_original_text.strip()

        bert_substitutes, bert_rank_substitutes,real_prev_scores,real_embed_scores = lexicalSubstitute_mbart_tsar(
                                                                            en2en, 
                                                                            tmp_original_text,
                                                                            prefix,
                                                                            suffix,
                                                                            target_word,
                                                                            beam)
        tp_words=list(set(label_words)&set(bert_substitutes))
        rec_all+=len(list(set(label_words)))
        potential_1+=len(tp_words)
        tp_words_one=list(set(label_words)&set(bert_substitutes[0:1]))
        tp_words_three=list(set(label_words)&set(bert_substitutes[0:3]))
        tp_words_five=list(set(label_words)&set(bert_substitutes[0:5]))
        tp_words_ten=list(set(label_words)&set(bert_substitutes[0:10]))

        if len(tp_words_one)>0:
            potential_2+=1

        if len(tp_words_three)>0:
            potential_3+=1
        
        if len(tp_words_five)>0:
            potential_5+=1

        if len(tp_words_ten)>0:
            potential_10+=1
        

        output_sr_file.write(";".join(label_words)+"|||"+";".join(bert_substitutes)+"|||"+";".join(tp_words)+"\n")
        output_sr_file.flush()
    print("[THE POTENTIAL@1 SCORE IS]",potential_2/count1)
    print("[THE POTENTIAL@3 SCORE IS]",potential_3/count1)
    print("[THE POTENTIAL@5 SCORE IS]",potential_5/count1)
    print("[THE POTENTIAL@10 SCORE IS]",potential_10/count1)

    return potential_1/rec_all

    





