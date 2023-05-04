datasets=[line.strip() for line in open("swords.txt")]
labels_count=0
for i in range(len(datasets)):
    tmp_count=len(list(set(datasets[i].split("|||")[-1].split("&"))))
    labels_count+=tmp_count
print(labels_count)
