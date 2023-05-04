
##NNSeval
#python3 Substitute_Para.py \
#  --eval_dir "data/swords-v1.1_test.json" \
#  --paraphraser_path "checkpoints/para/transformer/" \
#  --paraphraser_model "checkpoint_best.pt" \
#  --bpe "subword_nmt" \
#  --bpe_codes "data/para/codes.40000.bpe.en" \
#  --paraphraser_dict 'data-bin/para/' \
#  --output_SR_file "results/swords-v1.1_test_mygenerator.lsr.json"  
TEST_FILE=data/LS14/test/coinco_test.preprocessed
TEST_GOLDEN_FILE=data/LS14/test/coinco_test.gold
OUTPUT=ls14.out
OUTPUT_SCORE=ls14.scores.out


python3 LSPara.multi.ls14.bart.py \
  --eval_dir $TEST_FILE \
  --eval_gold_dir $TEST_GOLDEN_FILE \
  --paraphraser_path "checkpoints/para/transformer/" \
  --paraphraser_model "checkpoint_best.pt" \
  --bpe "subword_nmt" \
  --bpe_codes "checkpoints/para/transformer/codes.40000.bpe.en" \
  --beam 100 \
  --bertscore -100  \
  --output_SR_file $OUTPUT \
  --output_score_file $OUTPUT_SCORE



#--paraphraser_dict None \
