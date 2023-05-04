ori=0
sys=0
tp=0
potential=0
#for line in open("results/ls07/ls07.subs.txt",encoding="utf-8"):
for line in open("results/swords/swords.subs.txt",encoding="utf-8"):
#for line in open("results_for_qiang/gen_datasets_4/embed.sorted.50.txt",encoding="utf-8"):
#for line in open("results.transzh.29w.max50.attn+2.txt",encoding="utf-8"):
    ori+=len(list(set(line.split("|||")[2].strip().split("&"))))
    tmp_sys=line.split("|||")[3].strip()
    if tmp_sys!="":
        sys+=len(list(set(tmp_sys.split(" "))))
        tmp_tp=line.split("|||")[4].strip()
        
        if tmp_tp!="":
            #print(len(list(set(tmp_tp.split()))))
            tp+=len(list(set(tmp_tp.split(" "))))
            
            
            potential+=1
rc1=tp/ori
pre1=tp/sys
print("recall",tp/ori)
print("precisiion",tp/sys)
print("potential",potential/762)
print("f1",(2*rc1*pre1)/(rc1+pre1))

# print(ori)
# print(sys)
# print(tp)
# with open("rubbish/bad_words","w+") as f1:
#     read_lines=open("results.transzh.1.suffix.txt").readlines()[:524]
#     for i in range(len(read_lines)):
#         good_word=read_lines[i].strip().split("|||")[-1].split()
#         label_word=read_lines[i].strip().split("|||")[1].split()
#         if len(good_word)<=1:
#             for word in label_word:
#                 if len(word)>1:
#                     f1.write(word.strip()+"\n")

    


       