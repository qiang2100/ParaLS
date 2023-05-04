import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import numpy as np
import random
import time
from tqdm import tqdm
from reader import Reader_lexical
from metrics.evaluation import evaluation

from bart_score import BARTScorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bert_score.scorer import BERTScorer
from fairseq.models.transformer import TransformerModel

# bart_scorer=BARTScorer(device="cuda",checkpoint="/home/yz/liukang/liukang/huggingface/facebook/bart-large-cnn")
# bart_scorer.load(path="/home/yz/liukang/liukang/huggingface/facebook/bart-large-cnn/bart.pth")
bart_scorer=None
# bleurt_tokenizer = AutoTokenizer.from_pretrained("bleurt-large-512")
# bleurt_model = AutoModelForSequenceClassification.from_pretrained("bleurt-large-512")
# bleurt_model.eval()
bleurt_model=None

# bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
bertscorer=None
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

def substitutes_scores(original_sentence, substitutes, index_of_word):
    sss = []
    ccs = []

    words = original_sentence.split(' ')

    for sub in substitutes:
        sss.append(original_sentence)

        new_sent = ""

        for i in range(len(words)):
            if i == index_of_word:
                new_sent += sub + " "
            else:
                new_sent += words[i] + " "
        ccs.append(new_sent.strip())

    P, R, F1 = bertscorer.score(ccs, sss)
    score_bart = bart_scorer.score(sss, ccs, batch_size=64)
    with torch.no_grad():
        scores_bleurt = bleurt_model(**bleurt_tokenizer(sss, ccs, return_tensors='pt', padding=True))[0].squeeze()

    return score_bart,scores_bleurt.tolist(),F1

def lookahead_scores(model,original_sentence,substitutes,word_index):
    sentence_words=original_sentence.split()
    prefix=" ".join(original_sentence.strip().split()[:word_index]).strip()
    complex_word=sentence_words[word_index]
    sentence=original_sentence
    beam=1
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
        print("finding a error I should do something")
        import pdb
        pdb.set_trace()

    prefix_tokens = model.encode(prefix)
    prefix_tokens = prefix_tokens[:-1].view(1,-1)

    complex_tokens = model.encode(complex_word)
    #1.make some change to the original sentence
    #=prefix.strip()+" "+process_string(complex_word.strip()+" "+stored_suffix1.strip())
    #sentence=new_sentence


    sentence_tokens = model.encode(sentence)

    suffix_tokens=model.encode(suffix1)[:-1]
    suffix_tokens=torch.tensor(suffix_tokens)
    suffix_tokens=suffix_tokens.tolist()

    final_gap_score=[]

    for i in range(len(substitutes)):
        #make some change to the suffix tokens
        candi_tokens=model.encode(substitutes[i])
        tmp_suffix_tokens=candi_tokens[:-1].tolist()+suffix_tokens

        
        attn_len = len(prefix_tokens[0])+len(candi_tokens)-1
        import time
        first_time=time.time()
        outputs,combined_sss,prev_masks,prev_masks2,scores_with_suffix,scores_with_suffix_masks,scores_with_dynamic = model.generate2(sentence_tokens.cuda(), 
                                                                                                                beam=beam, 
                                                                                                                prefix_tokens=prefix_tokens.cuda(), 
                                                                                                                attn_len=attn_len,
                                                                                                                #tgt_token=complex_tokens[:-1].tolist(),
                                                                                                                tgt_token=-1,
                                                                                                                suffix_ids=tmp_suffix_tokens,
                                                                                                                max_aheads=0) 
        second_time=time.time()
        #print("sepend this times",second_time-first_time)
        outputs=outputs.cpu()
        
        for i in range(len(combined_sss)):
            if combined_sss[i]!=[]:
                if type(combined_sss[i])==list:
                    combined_sss[i][0]=combined_sss[i][0].to("cpu")
                    combined_sss[i][1]=combined_sss[i][1].to("cpu")
                else:
                    combined_sss[i]=combined_sss[i].to("cpu")
        try:
            prev_masks=prev_masks.cpu()
            prev_masks2=prev_masks2.cpu()
        except:
            pass
        scores_with_suffix=scores_with_suffix.cpu()
        scores_with_suffix_masks=scores_with_suffix_masks.cpu()
        
        if len(prefix_tokens[0])!=0:
            if len(prefix_tokens[0])>1:
                tmp_final_gap_score=scores_with_suffix[0,len(prefix_tokens[0])-1].tolist()-scores_with_suffix[0,len(prefix_tokens[0])-2].tolist()
            else:
                tmp_final_gap_score=scores_with_suffix[0,len(prefix_tokens[0])-1].tolist()
        else:
            tmp_final_gap_score=scores_with_dynamic[0,0].tolist()
        final_gap_score.append(tmp_final_gap_score)


    return final_gap_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #


    # --------------- test dataset
    parser.add_argument("-tt", "--test_file", type=str, help="path of the test file dataset",
                        default='data/LS14/test/coinco_test.preprocessed')
    parser.add_argument("-tgf", "--test_golden_file", type=str, help="path of the golden file dataset",
                        default='data/LS14/test/coinco_test.gold')

    # --------------- output results
    parser.add_argument("-outp", "--output_results", type=str, help="path of the output file with results",
                        default='results/ls14.gap/coinco_result_bart_bluert')
    parser.add_argument("-eval", "--results_file", type=str, help="path of the output file with gap metric",
                        default='results/ls14.gap/coinco_score_bart_bluert')

    parser.add_argument("-bart", "--bartscore", type=bool, help="whether we use bartscore",
                        default=True)
    parser.add_argument("--weight_bart",
                        default=1,
                        type=float,
                        help="The weight of bartscore.")
    parser.add_argument("--weight_bert",
                        default=0,
                        type=float,
                        help="The weight of bartscore.")
    parser.add_argument("--weight_bleurt",
                        default=1,
                        type=float,
                        help="The weight of bartscore.")
   
    # ----------gap flags
    parser.add_argument("-g", "--gap", type=bool, help="whether we use the gap ranking (candidate ranking)",
                        default=True)
    parser.add_argument("-gfc", "--golden_file_cadidates", type=str, help="path of the golden file dataset for gap",
                        default='data/LS14/coinco.gold.candidates')
    args = parser.parse_args()
    
    """
    reader of features/labels and candidates if gap
    """
    reader = Reader_lexical()
    reader.create_feature(args.test_file)

   
    reader.create_candidates(args.golden_file_cadidates)
    
    evaluation_metric = evaluation()

    paraphraser_path="checkpoints/para/transformer/"
    paraphraser_model="checkpoint_best.pt"
    bpe="subword_nmt"
    bpe_codes="checkpoints/para/transformer/codes.40000.bpe.en"
    
    en2en = TransformerModel.from_pretrained(paraphraser_path, checkpoint_file=paraphraser_model,bpe=bpe,
                                             bpe_codes=bpe_codes).cuda().eval()

    for main_word in tqdm(reader.words_candidate):
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:

                change_word = context[0]
                text = context[1]
                original_text = text
                index_word = context[2]
                change_word = text.split(' ')[int(index_word)]
                synonyms = []

                #print(text)
                #print(change_word)

                if args.gap:
                    try:
                        proposed_words_list = reader.candidates[main_word]
                    except:
                        # for ..N in LS14
                        proposed_words_list = ["..", ".", ",", "!"]         
                    proposed_words = reader.created_dict_proposed(proposed_words_list)
                #bart_scores,bleurt_scores,bert_scores = substitutes_scores(original_text, proposed_words_list, int(index_word))
                ahead_scores=lookahead_scores(en2en,original_text, proposed_words_list, int(index_word))

                final_scores = []
                #print(len(proposed_words_list),len(bart_scores))
                for i in range(len(proposed_words_list)):
                    #score = args.weight_bart*float(bart_scores[i])+ args.weight_bleurt*float(bleurt_scores[i]) +args.weight_bert*float(bert_scores[i])
                    #score = args.weight_bart*float(1)+ args.weight_bleurt*float(1) +args.weight_bert*float(1)
                    score=ahead_scores[i]
                    final_scores.append(score)

                for substitute_str, score in zip(proposed_words_list, final_scores):
                    proposed_words[substitute_str] = score

                #print(proposed_words)
                
                if args.gap:
                    evaluation_metric.write_results(args.output_results + "_gap.txt",
                                                    main_word, instance,
                                                    proposed_words)

        #break

               
    
    if args.gap:
        evaluation_metric.gap_calculation(args.test_golden_file,
                                          args.output_results  + "_gap.txt",
                                          args.results_file  + "_gap.txt")
    