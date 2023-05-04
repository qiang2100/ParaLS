from bert_score.scorer import BERTScorer
import torch
from bert_score import score



scorer = BERTScorer(lang="en", rescale_with_baseline=True)

sen  = "been tendered under its offer."

candis = ['bid', 'bids', 'tender', 'proposal', 'quote', 'tendering', 'tenders']

complex = 'offer'

refs = []

cands = []

for cand in candis:
	refs.append(sen)

	pre = sen.replace(complex, cand)
	cands.append(pre)

Ff1 = []

P, R, F1 = scorer.score(cands, refs)
print(F1)
P, R, F1 = scorer.score(cands, refs)
print(F1)
P, R, F1 = scorer.score(cands, refs)
print(F1)
P, R, F1 = scorer.score(cands, refs)
P, R, F1 = scorer.score(cands, refs)


print(F1)





cands = [["The books were later adapted to the TV series and the movie ."],
["The books were later adjusted to the TV series and the movie ."],
["The books were later tailored to the TV series and the movie ."],
["The books were later converted to a TV series and a film of the show."],
["The books were later transformed into a TV series and a film of the show."],
["The books were later modified into a TV series and a movie with a characteristic feature ."],
["The books were later set up for the TV series and the movie ."],
["The books were later put into the TV series and the movie ."],
["The books were later turned into a TV series and a movie with a feature ."],
["The books were later tuned to the TV series and the movie ."],
["The books were later changed to a TV series and a movie with a feature ."],
["The books were later made into a TV series and a movie with a feature on it."],
["The books were later re-adapted to the TV series and the film of the show."]]



#for pre in predictions:
	#sis, r_tokens, h_tokens = scorer.comput_sis(pre,references[0])



	#max = torch.topk(sis[4], 1)

	#print(h_tokens[4], "---", max)