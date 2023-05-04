#!/usr/bin/python
# -*- coding: UTF-8 -*-

from operator import index
import os
from pyexpat import model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import csv
import logging
import os
import random
import math
import sys
import re

from sklearn.metrics.pairwise import cosine_similarity as cosine
from torch import nn
import torch.nn.functional as F

import numpy as np
import torch
import nltk

import pdb

from pathlib import Path
import openpyxl
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from fairseq.models.transformer import TransformerModel

#from transformers import AutoTokenizer, AutoModelWithLMHead

from nltk.stem import PorterStemmer

from bert_score.scorer import BERTScorer


ps = PorterStemmer()

#scorer = BERTScorer(model_type="bert-base-uncased",lang="en", rescale_with_baseline=True)
scorer = BERTScorer(lang="en", rescale_with_baseline=True)
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
    #     scores_with_suffix[:,i]/=(len(suffix_tokens)+i-prefix_len+1)
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
        #hope_len=len(nltk.word_tokenize(prefix_str))+flag
        hope_len=len(prefix_str.strip().split())+flag
    else:
        hope_len=-1

    hope_outputs=[]
    hope_outputs_scores=[]
    
    for i in range(len(beam_idx)):
        if sorted_scores[i]==(-math.inf):
            continue
        tmp_str1=tgt_dict.string(outputs[beam_idx[i],:(len_idx[i]+1)]).replace("@@ ","")
        #if len(nltk.word_tokenize(tmp_str1))==hope_len:
        if len(tmp_str1.strip().split())==hope_len:
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
    return hope_outputs,hope_outputs_scores







def read_eval_index_dataset(data_path, is_label=True):
    sentences=[]
    mask_words = []
    mask_labels = []

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            
            if not line:
                break
            
            sentence,words = line.strip().split('\t',1)
                #print(sentence)
            mask_word,labels = words.strip().split('\t',1)
            label = labels.split('\t')
                
            sentences.append(sentence)
            mask_words.append(mask_word)
                
            one_labels = []
            for la in label[1:]:
                la = la.strip()
                if la not in one_labels:
                    la_id,la_word = la.split(':')
                    one_labels.append(la_word)
                
                #print(mask_word, " ---",one_labels)
            mask_labels.append(one_labels)
            
    return sentences,mask_words,mask_labels

def read_eval_dataset(data_path, is_label=True):
    sentences=[]
    mask_words = []
    mask_labels = []
    id = 0

    with open(data_path, "r", encoding='ISO-8859-1') as reader:
        while True:
            line = reader.readline()
            if is_label:
                id += 1
                if id==1:
                    continue
                if not line:
                    break
                sentence,words = line.strip().split('\t',1)
                #print(sentence)
                mask_word,labels = words.strip().split('\t',1)
                label = labels.split('\t')
                
                sentences.append(sentence)
                mask_words.append(mask_word)
                
                one_labels = []
                for la in label:
                    la = la.strip()
                    if la not in one_labels:
                        one_labels.append(la)
                
                #print(mask_word, " ---",one_labels)
                    
                mask_labels.append(one_labels)
            else:
                if not line:
                    break
                #print(line)
                sentence,mask_word = line.strip().split('\t')
                sentences.append(sentence)
                mask_words.append(mask_word)
    return sentences,mask_words,mask_labels
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

def extract_substitute(output_sentences, original_sentence, complex_word, threshold):

    original_words = nltk.word_tokenize(original_sentence)

    index_of_complex_word = -1

    if complex_word  not in original_words:
        i = 0
        for word in original_words:
            if complex_word == word.lower():
                index_of_complex_word = i
                break
            i += 1
    else:
        index_of_complex_word = original_words.index(complex_word)
    
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
            tran = output_sentences[i].replace('<unk> ', '')
            output_sentences[i] = tran

    complex_stem = ps.stem(complex_word)
    #orig_pos = nltk.pos_tag(original_words)[index_of_complex_word][1]
    not_candi = set(['the', 'with', 'of', 'a', 'an' , 'for' , 'in'])
    not_candi.add(complex_stem)
    not_candi.add(complex_word)
    

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
        words = nltk.word_tokenize(sentence)

        if index_of_complex_word>=len(words):
            continue

        if words[index_of_complex_word] == complex_word:
            len_words = len(words)
            if index_of_complex_word+1 < len_words:
                suffix = words[index_of_complex_word+1]#words[index_of_complex_word+1:min(index_of_complex_word+4,len_words)]
                if suffix not in suffix_words:
                    suffix_words.append(suffix)

    for sentence in output_sentences:


        if len(sentence)<3:
            continue
      
        words = nltk.word_tokenize(sentence)
        if index_of_complex_word>=len(words):
            continue
        candi = words[index_of_complex_word].lower()
        candi_stem = ps.stem(candi)
        if candi_stem in not_candi or candi in not_candi:
            continue

        len_words = len(words)
        sent_suffix = ""
        if index_of_complex_word + 1 < len_words:
            sent_suffix = words[index_of_complex_word+1]

        #if sent_suffix in suffix_words:
        if candi not in substitutes:
            substitutes.append(candi)
   
    if len(substitutes)>0:
        bert_scores = substitutes_BertScore(context, complex_word, substitutes)

        #print(substitutes)
        bert_scores = bert_scores.tolist()
        
        #pdb.set_trace()


        filter_substitutes, bert_scores = filterSubstitute(substitutes, bert_scores, threshold)

        rank_bert = sorted(bert_scores,reverse = True)

        rank_bert_substitutes = [filter_substitutes[bert_scores.index(v)] for v in rank_bert]
        #filter_substitutes=substitutes
        #rank_bert_substitutes=substitutes

        return filter_substitutes, rank_bert_substitutes

    return [],[]

def lexicalSubstitute(model, sentence, complex_word, beam, threshold):
    index_complex = sentence.find(complex_word)
    ori_words=sentence.strip().split()
    prefix = ""
    suffix1 = ""
    if(index_complex != -1):
        prefix = sentence[0:index_complex]
        if len(ori_words)>ori_words.index(complex_word)+1:
            suffix1=sentence[index_complex+len(complex_word):].strip()
            suffix1=suffix1.replace("''","\"").strip()
            suffix1=suffix1.replace("``","\"").strip()
            if suffix1.endswith("\""):
                suffix1=suffix1[:-1]
                suffix1=suffix1.strip()
            if suffix1.endswith("'"):
                suffix1=suffix1[:-1]
                suffix1=suffix1.strip()                            
            #suffix1=process_string(suffix1)
            suffix1=" ".join(suffix1.split(" ")[:6])
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
        #print("*************cannot find the complex word")
        #print(sentence)
        #print(complex_word)
        sentence = sentence.lower()

        return lexicalSubstitute(model, sentence, complex_word,  beam, threshold)
    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)

    complex_tokens = model.encode(complex_word)

    sentence_tokens = model.encode(sentence)

    suffix_tokens=model.encode(suffix1)[:-1]
    suffix_tokens=torch.tensor(suffix_tokens)
    suffix_tokens=suffix_tokens.tolist()
    attn_len = len(prefix_tokens[0])+len(complex_tokens)-1
    if model.tgt_dict.string(prefix_tokens).strip().replace("@@ ","")!=prefix.strip():
        print("finding prefix not good")

    #outputs = model.generate2(sentence_tokens, beam=20, prefix_tokens=prefix_tokens)
    # outputs,pre_scores = model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len)
    #outputs,pre_scores = model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len,suffix_ids=suffix_tokens) 
    outputs,combined_sss,prev_masks,prev_masks2,scores_with_suffix,scores_with_suffix_masks = model.generate2(sentence_tokens.cuda(), beam=beam, prefix_tokens=prefix_tokens.cuda(), attn_len=attn_len,suffix_ids=suffix_tokens,max_aheads=5)   
    outputs=outputs.cpu()
    
    for i in range(len(combined_sss)):
        if combined_sss[i]!=[]:
            if type(combined_sss[i])==list:
                combined_sss[i][0]=combined_sss[i][0].to("cpu")
                combined_sss[i][1]=combined_sss[i][1].to("cpu")
            else:
                combined_sss[i]=combined_sss[i].to("cpu")
    prev_masks=prev_masks.cpu()
    prev_masks2=prev_masks2.cpu()
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
    #embed_scores=give_embedding_scores(outputs,model.models[0].state_dict()["decoder.embed_tokens.weight"].cpu(),complex_tokens=complex_tokens,temperature=0.15)
    #assert embed_scores.size()==scores_with_suffix[:,:(outputs.size()[-1]-1)].size()
    #scores_with_suffix[:,:(outputs.size()[-1]-1)]=scores_with_suffix[:,:(outputs.size()[-1]-1)]+embed_scores
    # if sentence=='women usually notice little change in their breasts , but if you are a man , your breasts may become slightly larger and may be tender .':
    #     import pdb
    #     pdb.set_trace()

    outputs,outputs_scores=give_real_scores_ahead(model.tgt_dict,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=len(prefix_tokens[0]),prefix_str=prefix,max_ahead=5,flag=1)


    # if outputs==[]:
    #     import pdb
    #     pdb.set_trace()

    #print(outputs)

    #outputs=outputs[torch.squeeze(torch.topk(output_final_scores,k=beam,dim=1)[-1].view(1,-1),0)][:50]

    #output_sentences = [model.decode(x['tokens']) for x in outputs]
    output_sentences=[model.decode(x) for x in outputs]
    # for s1 in output_sentences:
    #     print(s1[:200])
    # for s1 in outputs:
    #     print(model.tgt_dict.string(s1)[:150])   
    bertscore_substitutes, ranking_bertscore_substitutes = extract_substitute(output_sentences, sentence, complex_word, threshold)

    #print(pre_scores)

    #for sen in output_sentences:
    #    print(sen)

    #bertscore_substitutes, ranking_bertscore_substitutes = extractSubstitute_bertscore(output_sentences, sentence, complex_word, threshold)
    #suffix_substitutes = extractSubstitute_suffix(output_sentences, sentence, complex_word)

    return bertscore_substitutes, ranking_bertscore_substitutes


   

def gen_nnseval(en2en,output_sr_file,eval_dir,beam=100,bertscore=-100):
    # output_SR_file="results/parabank/nnseval/recall.score"
    # eval_dir="datasets/NNSeval.txt"
    # output_sr_file = open(output_SR_file,"w+")
    # paraphraser_path="../fairseq-main_prefix/fairseq-main_prefix/checkpoints/para/transformer"
    # paraphraser_model="checkpoint_best.pt"
    # bpe="subword_nmt"
    # bpe_codes="../fairseq-main_prefix/fairseq-main_prefix/checkpoints/para/transformer/codes.40000.bpe.en"
    # beam=100
    # bertscore=0

    eval_examples, complex_words, complex_labels = read_eval_index_dataset(eval_dir)
    eval_examples=eval_examples
    complex_words=complex_words
    complex_labels=complex_labels
    eval_size = len(eval_examples)

    CS = []
    CS2 = []
    CS3 = []

    from tqdm import tqdm
    for i in tqdm(range(eval_size)):

        bert_substitutes, bert_rank_substitutes = lexicalSubstitute(en2en,eval_examples[i],complex_words[i],beam, bertscore)    
        CS2.append(bert_substitutes[:20])
        CS3.append(bert_rank_substitutes[:20])
        #final_str=" ".join(complex_labels[i])+"|||"+" ".join(bert_rank_substitutes[:10])+"|||"+" ".join(list(set(complex_labels[i])&set(bert_rank_substitutes[:10])))+"\n"
        final_str="&".join(complex_labels[i])+"|||"+" ".join(bert_substitutes[:10])+"|||"+" ".join(list(set(complex_labels[i])&set(bert_substitutes[:10])))+"\n"
        output_sr_file.write(final_str)
        #bro
        #if i==5:
            #break
        output_sr_file.flush()

    potential,precision,recall,F_score=evaulation_SS_scores(CS2, complex_labels)
    print("The score of evaluation for candidate selection")
    output_sr_file.write(str(potential))
    output_sr_file.write('\t')
    output_sr_file.write(str(precision))
    output_sr_file.write('\t')
    output_sr_file.write(str(recall))
    output_sr_file.write('\t')
    output_sr_file.write(str(F_score))
    output_sr_file.write('\n')
    potential_bert,precision_bert,recall_bert,F_score_bert=evaulation_SS_scores(CS3, complex_labels)
    print("The score of evaluation for candidate selection")
    output_sr_file.write(str(potential_bert))
    output_sr_file.write('\t')
    output_sr_file.write(str(precision_bert))
    output_sr_file.write('\t')
    output_sr_file.write(str(recall_bert))
    output_sr_file.write('\t')
    output_sr_file.write(str(F_score_bert))
    output_sr_file.write('\n')
    output_sr_file.close()
    return recall

