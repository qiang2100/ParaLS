# f2=open("tsar2022_en_test_gold.tsv.nosen","w+")

# for line in open("tsar2022_en_test_gold.tsv"):
#     tmp1="|||".join(list(set(line.strip().split("\t")[2:])))
#     f2.write(tmp1.strip()+"\n")
# f2.close()
with open("en.gold","w+") as f1 :
    for line in open("tsar2022_en_test_gold.tsv"):
        dict1={}
        sentence="\t".join(line.strip().split("\t")[:2]).strip()
        gold_words=line.strip().split("\t")[2:]
        for word1 in gold_words:
            if word1 not in dict1:
                dict1[word1]=1
            else:
                dict1[word1]+=1
        gold_tmp_line=""
        for word2 in dict1:
            gold_tmp_line+=word2+" "+str(dict1[word2])+" "
        gold_tmp_line=gold_tmp_line.strip()
        sentence=sentence+"\t"+gold_tmp_line
        sentence=sentence.strip()+"\n"
        f1.write(sentence)
    

        
        