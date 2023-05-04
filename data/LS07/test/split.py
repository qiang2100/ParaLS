# with open("ls07.two_word.txt","w+") as f1:
#     for line in open("lst_test.gold"):
#         labels=line.strip().split("::")[1].split(";")[:-1]
#         str1=""
#         for word1 in labels:
#             if len(word1.strip().split(" "))>=3:
#                 str1+=word1.strip()
#                 str1+=";"
#         if str1=="":
#             continue
#         f1.write(line.strip().split("::")[0]+"::"+str1+"\n")

# from nltk import sent_tokenize
# sens=open("lst_test.preprocessed").readlines()
# for i1 in range(len(sens)):
#     context1=sens[i1].strip().split("\t")[3]
#     sen1=sent_tokenize(context1)
#     if len(sen1)>1:
#         print(i1+1)
# sens=[line.strip() for line in open("lst_test.preprocessed")]
# for i1 in range(len(sens)):
#     index1=int(sens[i1].strip().split("\t")[2])
#     sen1=sens[i1].strip().split("\t")[3].split()
#     if index1-1!=0:
#         if sen1[index1-1].strip()=="an":
#             print(i1+1)

