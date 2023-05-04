# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lib2to3.pgen2 import token
import math
from socket import TIPC_ADDR_NAME
from typing import Dict, List, Optional
import sys

import pdb

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock


class SequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        max_len=0,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len or self.model.max_decoder_positions()

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
            hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """

        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """

        return self._generate(sample, **kwargs)

    @torch.no_grad()
    def generate2(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate2(sample, **kwargs)

    @torch.no_grad()
    def generate3(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate3(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):

       
        #print("generate------------------")
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input. input keys: " + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)

        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

       

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        #print(encoder_outs[0])
        #print(len(encoder_outs[0]['encoder_out'][0][0][0]))
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        #print("src_tokens: ", src_tokens, " scores:",scores)
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask


        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams

            #pdb.set_trace()

            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            # if step==1:
            #     import pdb
            #     pdb.set_trace()
            with torch.autograd.profiler.record_function("EnsembleModel: forward_decoder"):
                lprobs, avg_attn_scores = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )


            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )
            #print("***************")

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )


            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]

            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    @torch.no_grad()
    def look_ahead(
        self,
        tokens=None,
        suffix_ids=None,
        step=None,
        incremental_states=None,
        encoder_outs=None,
        scores_with_suffix=None,
        scores_with_suffix_masks=None,
        finished_beams=None,
        attn_len=-1

    ):  
        theta=1
        lambda1=0

        ahead_step=0
        max_aheads=len(suffix_ids)

        now_tokens=tokens

        import copy      
        now_incremental_states=copy.deepcopy(incremental_states)
        now_encoder_outs=encoder_outs
        #now_attn_len=attn_len+len(suffix_ids) if attn_len!=-1 else -1
        now_attn_len=-1
        #now_attn_len=attn_len
        # now_scores_with_suffix=scores_with_suffix
        # now_scores_with_suffix_masks=scores_with_suffix_masks

        beam_size,_=tokens.size()


        tmp_scores=torch.zeros(beam_size,1)
        while ahead_step<max_aheads:

            # lprobs, avg_attn_scores = self.model.forward_decoder(now_tokens,
            #     now_encoder_outs,
            #     now_incremental_states,
            #     self.temperature         
            # )  
            if ahead_step==0 and tokens.size(0)==1:
                #print("I am facing gap score calcuation if you are doing genereation,stop!!!")
                lprobs, avg_attn_scores= self.model.forward_decoder(now_tokens,
                    now_encoder_outs,
                    now_incremental_states,
                    self.temperature,
                    attn_len,
                    tgt_token=-1
                )

            else:
                lprobs, avg_attn_scores = self.model.forward_decoder(now_tokens,
                    now_encoder_outs,
                    now_incremental_states,
                    self.temperature,
                    now_attn_len     
                )      
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
            suffix_id=suffix_ids[ahead_step]
            tmp_scores=tmp_scores+(1-ahead_step*lambda1)*torch.index_select(lprobs,1,torch.tensor(suffix_id).cuda()).cpu()
            now_tokens=torch.cat((now_tokens,torch.tensor([[suffix_id]]*beam_size).cuda()),dim=-1)
            ahead_step+=1

        this_tokens=self.tgt_dict.string(tokens[:, -1:]).split("\n")
        this_tokens_labels=torch.tensor(list(map(lambda x:x.endswith("@@"),this_tokens))).cpu()
        
        this_tokens_labels=this_tokens_labels+(tokens[:,-1]==torch.tensor(2)).cpu()
        this_tokens_labels=this_tokens_labels+finished_beams.cpu().squeeze(dim=0)

        if suffix_ids==None:
            this_tokens_labels=(this_tokens_labels!=this_tokens_labels)
        elif len(suffix_ids)==0:
            this_tokens_labels=(this_tokens_labels!=this_tokens_labels)
        else:
            pass

        
        scores_with_suffix_masks[:,step]=this_tokens_labels.cuda()
        # print(scores_with_suffix_masks[:,:step+1])
        scores_with_suffix[:,step]=scores_with_suffix[:,step]+tmp_scores.squeeze(1).cuda()
        return{
            "scores_with_suffix":scores_with_suffix,
            "scores_with_suffix_masks":scores_with_suffix_masks
        }



    @torch.no_grad()
    def look_ahead_bart(
        self,
        tokens=None,
        suffix_ids=None,
        step=None,
        incremental_states=None,
        encoder_outs=None,
        scores_with_suffix=None,
        scores_with_suffix_masks=None,
        finished_beams=None,
        attn_len=-1

    ):  
        theta=1
        lambda1=0

        ahead_step=0
        max_aheads=len(suffix_ids)

        now_tokens=tokens

        import copy      
        now_incremental_states=copy.deepcopy(incremental_states)
        now_encoder_outs=encoder_outs
        #now_attn_len=attn_len+len(suffix_ids) if attn_len!=-1 else -1
        now_attn_len=-1
        #now_attn_len=attn_len
        # now_scores_with_suffix=scores_with_suffix
        # now_scores_with_suffix_masks=scores_with_suffix_masks

        beam_size,_=tokens.size()


        tmp_scores=torch.zeros(beam_size,1)
        while ahead_step<max_aheads:

            # lprobs, avg_attn_scores = self.model.forward_decoder(now_tokens,
            #     now_encoder_outs,
            #     now_incremental_states,
            #     self.temperature         
            # )  
            if ahead_step==0 and tokens.size(0)==1:
                #print("I am facing gap score calcuation if you are doing genereation,stop!!!")
                lprobs, avg_attn_scores= self.model.forward_decoder(now_tokens,
                    now_encoder_outs,
                    now_incremental_states,
                    self.temperature,
                    attn_len,
                    tgt_token=-1
                )

            else:
                lprobs, avg_attn_scores = self.model.forward_decoder(now_tokens,
                    now_encoder_outs,
                    now_incremental_states,
                    self.temperature,
                    now_attn_len     
                )      
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty
            suffix_id=suffix_ids[ahead_step]
            tmp_scores=tmp_scores+(1-ahead_step*lambda1)*torch.index_select(lprobs,1,torch.tensor(suffix_id).cuda()).cpu()
            now_tokens=torch.cat((now_tokens,torch.tensor([[suffix_id]]*beam_size).cuda()),dim=-1)
            ahead_step+=1

        this_tokens=self.tgt_dict.string(tokens[:, -1:]).split("\n")
        this_tokens_labels=torch.tensor(list(map(lambda x:x.endswith("@@"),this_tokens))).cpu()
        
        #this_tokens_labels=this_tokens_labels+(tokens[:,-1]==torch.tensor(2)).cpu()
        this_tokens_labels=(this_tokens_labels!=this_tokens_labels)
        
        this_tokens_labels=this_tokens_labels+finished_beams.cpu().squeeze(dim=0)

        if suffix_ids==None:
            this_tokens_labels=(this_tokens_labels!=this_tokens_labels)
        elif len(suffix_ids)==0:
            this_tokens_labels=(this_tokens_labels!=this_tokens_labels)
        else:
            pass

        
        scores_with_suffix_masks[:,step]=this_tokens_labels.cuda()
        # print(scores_with_suffix_masks[:,:step+1])
        scores_with_suffix[:,step]=scores_with_suffix[:,step]+tmp_scores.squeeze(1).cuda()
        return{
            "scores_with_suffix":scores_with_suffix,
            "scores_with_suffix_masks":scores_with_suffix_masks
        }
    
    def _generate2(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        attn_len: int = -1,
        tgt_token: int = -1,
        suffix_ids=None,
        max_aheads=None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        #print("generate2：------------------", attn_len)
        #pdb.set_trace()
        combined_predicted_scores=[[]for i in range(3+len(suffix_ids))]
        # predict_scores = []
        # prefix_predict_scores=[]
        # prev_predict_scores = []
        # prev_scores=[]
        # prev_again_predict_scores=[]

        prefix_len = len(prefix_tokens[0])

        all_suffix_scores=[[] for i in range(1)]
        all_suffix_scores_masks=[[] for i in range(1)]

        indices_next = []
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input. input keys: " + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)

        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        #print("bsz: ", bsz,  " src_len:", src_len,  " beam_size:", beam_size)
        #print("src_tokens: ", src_tokens)
        
        #print("prefix_tokens: ", prefix_tokens, ' . len: ', len(prefix_tokens[0]))

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
        
            encoder_outs = self.model.forward_encoder(net_input)
        #print(encoder_outs[0])
        #print(len(encoder_outs[0]['encoder_out'][0][0][0]))
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring

        scores_with_suffix=(
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        ) 
        scores_with_suffix_masks=(
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        ) 
        scores_with_suffix_masks=(scores_with_suffix_masks!=scores_with_suffix_masks)

        first_position_ahead_scores=None
        #dynamic scors by aklk
        # try:
        #     dynamic_scores=(
        #         torch.zeros(max_len+1,bsz * beam_size, 512).to(src_tokens).float()
        #     )
        # except:
        embed_dim_tmp=self.model.models[0].encoder.cfg.encoder.embed_dim
        dynamic_scores=(
            torch.zeros(max_len+1,bsz * beam_size,embed_dim_tmp).to(src_tokens).float()
        )            
        
        #print("src_tokens: ", src_tokens, " scores:",scores)
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        finished_beams = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask        

        # list of completed sentences 
        # import pdb
        # pdb.set_trace()
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        #print("maxlen",max_len+1)
        if suffix_ids==[] or suffix_ids==None:
            len_suffix=0
        else:
            len_suffix=len(suffix_ids)
        #for step in range(prefix_len+1+len_suffix+1):  # one extra step for EOS marker
        for step in range(prefix_len+1+100):
        #for step in range(max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            import time
            first_time=time.time()

            with torch.autograd.profiler.record_function("EnsembleModel: forward_decoder_inner"):

                #pdb.set_trace()
                if(step>prefix_len):
                    lprobs, avg_attn_scores,dynamic_states = self.model.forward_decoder_inner(tokens[:, : step + 1],
                        encoder_outs,
                        incremental_states,
                        self.temperature,
                    )   
                    
   
                elif step<prefix_len:
                    # import time
                    # first_time=time.time()
                    lprobs, avg_attn_scores,dynamic_states = self.model.forward_decoder_inner(tokens[:, : step + 1],
                        encoder_outs,
                        incremental_states,
                        self.temperature,
                    )

                    # second_time=time.time()
                    # print("spend timeing",second_time-first_time)
                    # if step==3:
                    #     import pdb
                    #     pdb.set_trace()

                    # lprobs=(-torch.ones(tokens.shape[0],len(self.tgt_dict))*0.1)# the is the last token
                    # lprobs[:,4]=-1.1
                    # lprobs=lprobs.cuda()
                    # avg_attn_scores=None



                elif step==prefix_len:
                    #pdb.set_trace()
                    #tgt_token = int(src_tokens[0][step])
                    # import time
                    # first_time=time.time()
                    #
                    # if tgt_token!=-1:
                    # tgt_token=net_input["src_tokens"].cpu().tolist()[0][len(prefix_tokens):attn_len]
                    import copy
                    tgt_incremental_states=copy.deepcopy(incremental_states)
                    lprobs, avg_attn_scores,dynamic_states = self.model.forward_decoder_inner(tokens[:, : step + 1],
                        encoder_outs,
                        tgt_incremental_states,
                        self.temperature,
                        attn_len,
                        tgt_token
                    )
                    if tgt_token==-1:
                        incremental_states=copy.deepcopy(tgt_incremental_states)
                    else:
                        _, _,_ = self.model.forward_decoder_inner(tokens[:, : step + 1],
                            encoder_outs,
                            incremental_states,
                            self.temperature,
                            attn_len, 
                            tgt_token=-1
                        )
                        incremental_states=copy.deepcopy(tgt_incremental_states)
                    # second_time=time.time()
                    # print("spended time",second_time-first_time)


                    

            #print("!!!!!!!!!!!!", len(lprobs), " ---", len(lprobs[0]))

            #pdb.set_trace()
            
            #print(lprobs)
            #print("tokens: ", tokens[:, : step + 1])
            #if(step>len_prefix):
            #    pdb.set_trace()

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                # import time
                # time.time()
                # first_time=time.time()
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )

                # second_time=time.time()
                # print("spend time",second_time-first_time)
                #print("----", lprobs)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf
            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            if(step==prefix_len):
                #print("0000000000000000000")
                cand_scores, cand_indices, cand_beams = self.search.step2(
                    step,lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                    tokens[:, : step + 1],
                    original_batch_idxs,
                    )
                #print("@@@@@@@@@@@@@@",tokens[:, : step + 1])
                #print("%%%%%%%%%%%%%%",cand_scores)
                #print("$$$$$$$$$$$$$$",cand_indices)
                #print("&&&&&&&&&&&&&&",cand_beams)
                # if prefix_len!=0:                    
                #     predict_scores = cand_scores-prev_scores
                # else:
                # predict_scores=cand_scores
                # prefix_predict_scores=predict_scores
                if prefix_len!=0:
                    combined_predicted_scores[0]=[cand_scores-prev_scores,cand_scores]
                else:
                    combined_predicted_scores[0]=[cand_scores,cand_scores]

                
            elif(step<prefix_len):
                # import time
                # first_time=time.time()
                #print("11111111111111111111")
                cand_scores, cand_indices, cand_beams = self.search.step1(
                    step,lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                    tokens[:, : step + 1],
                    original_batch_idxs,
                    )
                if step==prefix_len-1:
                    #print(step)
                    prev_scores=cand_scores
                # second_time=time.time()
                # print("spending time",second_time-first_time)

            else:
                # import time
                # time.time()
                # first_time=time.time()
                if step==prefix_len+1:

                    this_tokens=self.tgt_dict.string(tokens[:, step: step + 1]).split("\n")
                    this_tokens_labels=list(map(lambda x:x.endswith("@@"),this_tokens))
                    this_tokens_labels=torch.tensor([this_tokens_labels]).cuda()
                    this_tokens_labels_again=(this_tokens_labels!=this_tokens_labels)
                    if "onlt one @@"!= "onlt one @@":
                        this_tokens_labels=(this_tokens_labels!=this_tokens_labels)
                    else:
                        pass
                elif step==prefix_len+2:
                    if "onlt one @@"!= "onlt one @@":
                        this_tokens_labels_again=(this_tokens_labels!=this_tokens_labels)
                    else:
                        this_tokens_again=self.tgt_dict.string(tokens[:, step: step + 1]).split("\n")
                        this_tokens_labels_again=list(map(lambda x:x.endswith("@@"),this_tokens_again)) 
                        this_tokens_labels_again=torch.tensor([this_tokens_labels_again]).cuda()
                    
                else:
                    pass

                suffix_index=step-prefix_len-1
        
                if suffix_ids==None or suffix_ids==[]:
                    suffix_id=[]
                    prev_suffix_id=[]
                    prev_again_suffix_id=[]
                else:
                    suffix_id=[]
                    prev_suffix_id=[]
                    prev_again_suffix_id=[]   
                    if -1<suffix_index<len(suffix_ids):
                        suffix_id=[suffix_ids[suffix_index]]
                    if -1<suffix_index-1<len(suffix_ids):
                        prev_suffix_id=[suffix_ids[suffix_index-1]]
                    if -1<suffix_index-2<len(suffix_ids):
                        prev_again_suffix_id=[suffix_ids[suffix_index-2]]



                    # suffix_id=[suffix_ids[suffix_index]] if 0<=suffix_index<len(suffix_ids) else []
                    # if suffix_index-1<0:
                    #     prev_suffix_id=[]
                    #     prev_again_suffix_id=[]
                    # elif suffix_index-1<1:
                    #     prev_suffix_id=[suffix_ids[suffix_index-1]]
                    #     prev_again_suffix_id=[]
                    # elif 1<=suffix_index-1<len(suffix_ids)+1:
                    #     prev_suffix_id=[suffix_ids[suffix_index-1]]
                    #     prev_again_suffix_id=[]
                    # elif suffix_index-1                        
                    # elif suffix_index-1<len(suffix_ids):

                    #     prev_suffix_id=[suffix_ids[suffix_index-1]]
                    # else:
                    #     prev_suffix_id=[]
                if step==prefix_len+1:

                    cand_scores, cand_indices, cand_beams = self.search.step3(
                        step,lprobs.view(bsz, -1, self.vocab_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                        tokens[:, : step + 1],
                        original_batch_idxs,
                        suffix_id=[],
                        prev_suffix_id=[],
                        prev_again_suffix_id=[],
                        this_tokens_labels=this_tokens_labels,
                        this_again_tokens_labels=this_tokens_labels_again
                        
                    )

                    # cand_scores, cand_indices, cand_beams = self.search.step4(
                    #     step,lprobs.view(bsz, -1, self.vocab_size),
                    #     scores.view(bsz, beam_size, -1)[:, :, :step],
                    #     tokens[:, : step + 1],
                    #     original_batch_idxs,
                    #     id1=1                      
                    # )

                else:
                    cand_scores, cand_indices, cand_beams = self.search.step3(
                        step,lprobs.view(bsz, -1, self.vocab_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                        tokens[:, : step + 1],
                        original_batch_idxs,
                        suffix_id=[],
                        prev_suffix_id=[],
                        prev_again_suffix_id=[],
                        this_tokens_labels=this_tokens_labels,
                        this_again_tokens_labels=this_tokens_labels_again                      
                    )                    

                # cand_scores, cand_indices, cand_beams = self.search.step(
                #     step,
                #     lprobs.view(bsz, -1, self.vocab_size),
                #     scores.view(bsz, beam_size, -1)[:, :, :step],
                #     tokens[:, : step + 1],
                #     original_batch_idxs,
                # )                


                # if suffix_ids!=None and suffix_index<len(suffix_ids):
                #     predict_scores=cand_scores
                # elif suffix_ids!=None and ((suffix_index==len(suffix_ids)) and suffix_index!=0):
                #     prev_predict_scores=cand_scores
                #     pass

                # if suffix_ids==None or suffix_ids==[]:
                #     pass
                # elif -1<suffix_index<len(suffix_ids):
                #     predict_scores=cand_scores
                # elif suffix_index-1==len(suffix_ids)-1:
                #     prev_predict_scores=cand_scores
                # elif suffix_index-2==len(suffix_ids)-1:
                #     prev_again_predict_scores=cand_scores
                if suffix_ids==None or suffix_ids==[]:
                    pass
                elif suffix_index<=len(suffix_ids)+1:
                    # print(len(combined_predicted_scores),suffix_index,len(suffix_ids)+1)
                    combined_predicted_scores[suffix_index+1]=cand_scores
                    
                

                #print("@@@@@@@@@@@@@@",tokens[:, : step + 1])
                #print(step)

            #print(tokens[:, : step + 1])
            # cand_bbsz_idx contains beam indices for the top can8didate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]


            cand_bbsz_idx = cand_beams.cuda().add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)

            #d=eos_mask[:, :beam_size][cands_to_ignore]
            if step<prefix_len:
                pass               
            else:
                eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            # second_time=time.time()
            # print(second_time-first_time)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []

            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                if combined_predicted_scores[-1]==[]:
                    combined_predicted_scores[-1]=combined_predicted_scores[-2]
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.


                
            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))

            #print(eos_mask)
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # if cands_to_ignore[0][-1]==True:
            #     print(step<prefix_len)
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
            
            # if predict_scores!=[]:
            #     active_predict_scores=torch.gather(predict_scores, dim=1, index=active_hypos)
            #     predict_scores=active_predict_scores
            # if prefix_predict_scores!=[]:
            #     active_prefix_predict_scores=torch.gather(prefix_predict_scores, dim=1, index=active_hypos)
            #     predict_prefix_scores=active_prefix_predict_scores                

            # if prev_predict_scores!=[]:
            #     active_prev_predict_scores=torch.gather(prev_predict_scores, dim=1, index=active_hypos)
            #     prev_predict_scores=active_prev_predict_scores

            # if prev_again_predict_scores!=[]:
            #     active_prev_again_predict_scores=torch.gather(prev_again_predict_scores, dim=1, index=active_hypos)
            #     prev_again_predict_scores=active_prev_again_predict_scores

            #******************************the first version of look_ahead***************************************
            #
            
            for i in range(len(combined_predicted_scores)):
                if combined_predicted_scores[i]!=[]:
                    if type(combined_predicted_scores[i])==list:
                        combined_predicted_scores[i][0]=torch.gather(combined_predicted_scores[i][0], dim=1, index=active_hypos)
                        combined_predicted_scores[i][1]=torch.gather(combined_predicted_scores[i][1], dim=1, index=active_hypos)
                    else:
                        combined_predicted_scores[i]=torch.gather(combined_predicted_scores[i], dim=1, index=active_hypos)

            for i in range(len(all_suffix_scores)):
                if all_suffix_scores[i]!=[]:
                    all_suffix_scores[i]=torch.gather(all_suffix_scores[i], dim=1, index=active_hypos.cpu())
            
            for i in range(len(all_suffix_scores_masks)):
                if all_suffix_scores_masks[i]!=[]:
                    all_suffix_scores_masks[i]=torch.gather(all_suffix_scores_masks[i], dim=1, index=active_hypos.cpu())

            # if step==prefix_len+1:
            #     import step==prefix_len
            #     pdb.set_trace()
            if step>=prefix_len+1:
                active_this_tokens_labels=torch.gather(this_tokens_labels, dim=1, index=active_hypos)
                this_tokens_labels=active_this_tokens_labels

            if step>=prefix_len+2:
                active_this_tokens_labels_again=torch.gather(this_tokens_labels_again, dim=1, index=active_hypos)
                this_tokens_labels_again=active_this_tokens_labels_again

            #*****************************end of the first version look_ahead*****************************

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)
            # if step==4:
            #     import pdb
            #     pdb.set_trace()

            next_tokens=tokens[:,:step+2]
            # print(next_tokens[0])

            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
                scores_with_suffix[:, :step] = torch.index_select(
                    scores_with_suffix[:, :step], dim=0, index=active_bbsz_idx
                )
                scores_with_suffix_masks[:, :step] = torch.index_select(
                    scores_with_suffix_masks[:, :step], dim=0, index=active_bbsz_idx
                )
                dynamic_scores[:step,:,:]=torch.index_select(dynamic_scores[:step,:,:],dim=1,index=active_bbsz_idx)
                # print(scores[:, :step])
                # print(scores_with_suffix[:, :step])
                # #print(scores_with_suffix_masks[:, :step])
                # print(self.tgt_dict.string(tokens[:,:step+1]))


            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )
            
            dynamic_scores[step,:,:]=torch.index_select(dynamic_states,dim=1,index=active_bbsz_idx).squeeze()

            finished_beams=torch.index_select(finished_beams, dim=1, index=active_bbsz_idx)
            finished_beams+=cands_to_ignore
            # print(finished_beams)
            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

            if reorder_state is not None:
                # print(reorder_state,batch_idxs)
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            all_suffix_scores_index=step-prefix_len

                

            if prefix_len==0 and beam_size==1 and max_aheads==0 and all_suffix_scores_index==0:
                
                first_position_ahead_scores=copy.deepcopy(scores_with_suffix)
                first_scores_with_suffix_masks=copy.deepcopy(scores_with_suffix_masks)
                first_finished_beams=copy.deepcopy(finished_beams)
                tmp_dict1=self.look_ahead(next_tokens[:,:-1],suffix_ids,step,incremental_states,encoder_outs,
                    first_position_ahead_scores,first_scores_with_suffix_masks,finished_beams=first_finished_beams,attn_len=attn_len)
                # all_suffix_scores[all_suffix_scores_index]=tmp_dict1["suffix_scores"]
                # all_suffix_scores_masks[all_suffix_scores_index]=tmp_dict1["suffix_scores_masks"]
                first_position_ahead_scores=tmp_dict1["scores_with_suffix"]
                #scores_with_suffix_masks=tmp_dict1["scores_with_suffix_masks"]   
            
            if max_aheads==None:
                scores_with_suffix[:,step]=scores[:,step]
            elif all_suffix_scores_index<0:
                # if step==2:
                #     import pdb
                #     pdb.set_trace()
                
                scores_with_suffix[:,step]=scores[:,step]
                if prefix_len!=0 and beam_size==1 and max_aheads==0 and all_suffix_scores_index==-1:

                    tmp_dict1=self.look_ahead(next_tokens,suffix_ids,step,incremental_states,encoder_outs,
                        scores_with_suffix,scores_with_suffix_masks,finished_beams=finished_beams,attn_len=attn_len)
                    # all_suffix_scores[all_suffix_scores_index]=tmp_dict1["suffix_scores"]
                    # all_suffix_scores_masks[all_suffix_scores_index]=tmp_dict1["suffix_scores_masks"]
                    scores_with_suffix=tmp_dict1["scores_with_suffix"]
                    scores_with_suffix_masks=tmp_dict1["scores_with_suffix_masks"]   

                    
            elif all_suffix_scores_index<max_aheads and all_suffix_scores_index>=0:

                scores_with_suffix[:,step]=scores[:,step]
                #scores_with_suffix[:,step]=scores[:,prefix_len]
                tmp_dict1=self.look_ahead(next_tokens,suffix_ids,step,incremental_states,encoder_outs,
                    scores_with_suffix,scores_with_suffix_masks,finished_beams=finished_beams,attn_len=attn_len)
                # all_suffix_scores[all_suffix_scores_index]=tmp_dict1["suffix_scores"]
                # all_suffix_scores_masks[all_suffix_scores_index]=tmp_dict1["suffix_scores_masks"]
                scores_with_suffix=tmp_dict1["scores_with_suffix"]
                scores_with_suffix_masks=tmp_dict1["scores_with_suffix_masks"]
            elif all_suffix_scores_index>=max_aheads:
                scores_with_suffix[:,step]=scores[:,step]
            # print(scores_with_suffix_masks[:,:step+1])

        #print(next_tokens)
        #print(next_tokens.shape)
        if first_position_ahead_scores!=None:
            print("facing the prefix length is zero,you should use the dynamic_scores!!!")
            dynamic_scores=first_position_ahead_scores
        this_tokens_labels=1
        this_tokens_labels_again=1
        return [next_tokens], combined_predicted_scores,this_tokens_labels,this_tokens_labels_again,scores_with_suffix,scores_with_suffix_masks,dynamic_scores
        # sort by score descending
        
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized,predict_scores


    def _generate3(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        attn_len: int = -1,
        tgt_token: int = -1,
        suffix_ids=None,
        max_aheads=None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        #print("generate2：------------------", attn_len)
        #pdb.set_trace()
        combined_predicted_scores=[[]for i in range(3+len(suffix_ids))]
        # predict_scores = []
        # prefix_predict_scores=[]
        # prev_predict_scores = []
        # prev_scores=[]
        # prev_again_predict_scores=[]

        prefix_len = len(prefix_tokens[0])

        all_suffix_scores=[[] for i in range(1)]
        all_suffix_scores_masks=[[] for i in range(1)]

        indices_next = []
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input. input keys: " + str(net_input.keys()))

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)

        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        #print("bsz: ", bsz,  " src_len:", src_len,  " beam_size:", beam_size)
        #print("src_tokens: ", src_tokens)
        
        #print("prefix_tokens: ", prefix_tokens, ' . len: ', len(prefix_tokens[0]))

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
        
            encoder_outs = self.model.forward_encoder(net_input)
        #print(encoder_outs[0])
        #print(len(encoder_outs[0]['encoder_out'][0][0][0]))
        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring

        scores_with_suffix=(
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        ) 
        scores_with_suffix_masks=(
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        ) 
        scores_with_suffix_masks=(scores_with_suffix_masks!=scores_with_suffix_masks)

        first_position_ahead_scores=None
        #dynamic scors by aklk
        # try:
        #     dynamic_scores=(
        #         torch.zeros(max_len+1,bsz * beam_size, 512).to(src_tokens).float()
        #     )
        # except:
        embed_dim_tmp=self.model.models[0].encoder.cfg.encoder.embed_dim
        dynamic_scores=(
            torch.zeros(max_len+1,bsz * beam_size,embed_dim_tmp).to(src_tokens).float()
        )            
        
        #print("src_tokens: ", src_tokens, " scores:",scores)
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        finished_beams = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask        

        # list of completed sentences 
        # import pdb
        # pdb.set_trace()
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        #print("maxlen",max_len+1)
        if suffix_ids==[] or suffix_ids==None:
            len_suffix=0
        else:
            len_suffix=len(suffix_ids)
        #for step in range(prefix_len+1+len_suffix+1):  # one extra step for EOS marker
        for step in range(prefix_len+1+100):
        #for step in range(max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            import time
            first_time=time.time()

            with torch.autograd.profiler.record_function("EnsembleModel: forward_decoder_inner"):

                #pdb.set_trace()
                if(step>prefix_len):
                    lprobs, avg_attn_scores,dynamic_states = self.model.forward_decoder_inner(tokens[:, : step + 1],
                        encoder_outs,
                        incremental_states,
                        self.temperature,
                    )   
                    
   
                elif step<prefix_len:
                    # import time
                    # first_time=time.time()
                    lprobs, avg_attn_scores,dynamic_states = self.model.forward_decoder_inner(tokens[:, : step + 1],
                        encoder_outs,
                        incremental_states,
                        self.temperature,
                    )

                    # second_time=time.time()
                    # print("spend timeing",second_time-first_time)
                    # if step==3:
                    #     import pdb
                    #     pdb.set_trace()

                    # lprobs=(-torch.ones(tokens.shape[0],len(self.tgt_dict))*0.1)# the is the last token
                    # lprobs[:,4]=-1.1
                    # lprobs=lprobs.cuda()
                    # avg_attn_scores=None



                elif step==prefix_len:
                    #pdb.set_trace()
                    #tgt_token = int(src_tokens[0][step])
                    # import time
                    # first_time=time.time()
                    #
                    # if tgt_token!=-1:
                    # tgt_token=net_input["src_tokens"].cpu().tolist()[0][len(prefix_tokens):attn_len]
                    import copy
                    tgt_incremental_states=copy.deepcopy(incremental_states)
                    lprobs, avg_attn_scores,dynamic_states = self.model.forward_decoder_inner(tokens[:, : step + 1],
                        encoder_outs,
                        tgt_incremental_states,
                        self.temperature,
                        attn_len,
                        tgt_token
                    )
                    if tgt_token==-1:
                        incremental_states=copy.deepcopy(tgt_incremental_states)
                    else:
                        _, _,_ = self.model.forward_decoder_inner(tokens[:, : step + 1],
                            encoder_outs,
                            incremental_states,
                            self.temperature,
                            attn_len, 
                            tgt_token=-1
                        )
                        incremental_states=copy.deepcopy(tgt_incremental_states)
                    # second_time=time.time()
                    # print("spended time",second_time-first_time)


                    

            #print("!!!!!!!!!!!!", len(lprobs), " ---", len(lprobs[0]))

            #pdb.set_trace()
            
            #print(lprobs)
            #print("tokens: ", tokens[:, : step + 1])
            #if(step>len_prefix):
            #    pdb.set_trace()

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs
            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                # import time
                # time.time()
                # first_time=time.time()
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )

                # second_time=time.time()
                # print("spend time",second_time-first_time)
                #print("----", lprobs)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf
            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            if(step==prefix_len):
                #print("0000000000000000000")
                cand_scores, cand_indices, cand_beams = self.search.step2(
                    step,lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                    tokens[:, : step + 1],
                    original_batch_idxs,
                    )
                #print("@@@@@@@@@@@@@@",tokens[:, : step + 1])
                #print("%%%%%%%%%%%%%%",cand_scores)
                #print("$$$$$$$$$$$$$$",cand_indices)
                #print("&&&&&&&&&&&&&&",cand_beams)
                # if prefix_len!=0:                    
                #     predict_scores = cand_scores-prev_scores
                # else:
                # predict_scores=cand_scores
                # prefix_predict_scores=predict_scores
                if prefix_len!=0:
                    combined_predicted_scores[0]=[cand_scores-prev_scores,cand_scores]
                else:
                    combined_predicted_scores[0]=[cand_scores,cand_scores]

                
            elif(step<prefix_len):
                # import time
                # first_time=time.time()
                #print("11111111111111111111")
                cand_scores, cand_indices, cand_beams = self.search.step1(
                    step,lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                    tokens[:, : step + 1],
                    original_batch_idxs,
                    )
                if step==prefix_len-1:
                    #print(step)
                    prev_scores=cand_scores
                # second_time=time.time()
                # print("spending time",second_time-first_time)

            else:
                # import time
                # time.time()
                # first_time=time.time()
                if step==prefix_len+1:

                    this_tokens=self.tgt_dict.string(tokens[:, step: step + 1]).split("\n")
                    this_tokens_labels=list(map(lambda x:x.endswith("@@"),this_tokens))
                    this_tokens_labels=torch.tensor([this_tokens_labels]).cuda()
                    this_tokens_labels_again=(this_tokens_labels!=this_tokens_labels)
                    if "onlt one @@"!= "onlt one @@":
                        this_tokens_labels=(this_tokens_labels!=this_tokens_labels)
                    else:
                        pass
                elif step==prefix_len+2:
                    if "onlt one @@"!= "onlt one @@":
                        this_tokens_labels_again=(this_tokens_labels!=this_tokens_labels)
                    else:
                        this_tokens_again=self.tgt_dict.string(tokens[:, step: step + 1]).split("\n")
                        this_tokens_labels_again=list(map(lambda x:x.endswith("@@"),this_tokens_again)) 
                        this_tokens_labels_again=torch.tensor([this_tokens_labels_again]).cuda()
                    
                else:
                    pass

                suffix_index=step-prefix_len-1
        
                if suffix_ids==None or suffix_ids==[]:
                    suffix_id=[]
                    prev_suffix_id=[]
                    prev_again_suffix_id=[]
                else:
                    suffix_id=[]
                    prev_suffix_id=[]
                    prev_again_suffix_id=[]   
                    if -1<suffix_index<len(suffix_ids):
                        suffix_id=[suffix_ids[suffix_index]]
                    if -1<suffix_index-1<len(suffix_ids):
                        prev_suffix_id=[suffix_ids[suffix_index-1]]
                    if -1<suffix_index-2<len(suffix_ids):
                        prev_again_suffix_id=[suffix_ids[suffix_index-2]]



                    # suffix_id=[suffix_ids[suffix_index]] if 0<=suffix_index<len(suffix_ids) else []
                    # if suffix_index-1<0:
                    #     prev_suffix_id=[]
                    #     prev_again_suffix_id=[]
                    # elif suffix_index-1<1:
                    #     prev_suffix_id=[suffix_ids[suffix_index-1]]
                    #     prev_again_suffix_id=[]
                    # elif 1<=suffix_index-1<len(suffix_ids)+1:
                    #     prev_suffix_id=[suffix_ids[suffix_index-1]]
                    #     prev_again_suffix_id=[]
                    # elif suffix_index-1                        
                    # elif suffix_index-1<len(suffix_ids):

                    #     prev_suffix_id=[suffix_ids[suffix_index-1]]
                    # else:
                    #     prev_suffix_id=[]
                if step==prefix_len+1:

                    cand_scores, cand_indices, cand_beams = self.search.step3(
                        step,lprobs.view(bsz, -1, self.vocab_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                        tokens[:, : step + 1],
                        original_batch_idxs,
                        suffix_id=[],
                        prev_suffix_id=[],
                        prev_again_suffix_id=[],
                        this_tokens_labels=this_tokens_labels,
                        this_again_tokens_labels=this_tokens_labels_again
                        
                    )

                    # cand_scores, cand_indices, cand_beams = self.search.step4(
                    #     step,lprobs.view(bsz, -1, self.vocab_size),
                    #     scores.view(bsz, beam_size, -1)[:, :, :step],
                    #     tokens[:, : step + 1],
                    #     original_batch_idxs,
                    #     id1=1                      
                    # )

                else:
                    cand_scores, cand_indices, cand_beams = self.search.step3(
                        step,lprobs.view(bsz, -1, self.vocab_size),
                        scores.view(bsz, beam_size, -1)[:, :, :step],
                        tokens[:, : step + 1],
                        original_batch_idxs,
                        suffix_id=[],
                        prev_suffix_id=[],
                        prev_again_suffix_id=[],
                        this_tokens_labels=this_tokens_labels,
                        this_again_tokens_labels=this_tokens_labels_again                      
                    )              

                    # cand_scores, cand_indices, cand_beams = self.search.step1(
                    #     step,lprobs.view(bsz, -1, self.vocab_size),
                    #     scores.view(bsz, beam_size, -1)[:, :, :step],
                    #     tokens[:, : step + 1],
                    #     original_batch_idxs,
                    # )      

                # cand_scores, cand_indices, cand_beams = self.search.step(
                #     step,
                #     lprobs.view(bsz, -1, self.vocab_size),
                #     scores.view(bsz, beam_size, -1)[:, :, :step],
                #     tokens[:, : step + 1],
                #     original_batch_idxs,
                # )                


                # if suffix_ids!=None and suffix_index<len(suffix_ids):
                #     predict_scores=cand_scores
                # elif suffix_ids!=None and ((suffix_index==len(suffix_ids)) and suffix_index!=0):
                #     prev_predict_scores=cand_scores
                #     pass

                # if suffix_ids==None or suffix_ids==[]:
                #     pass
                # elif -1<suffix_index<len(suffix_ids):
                #     predict_scores=cand_scores
                # elif suffix_index-1==len(suffix_ids)-1:
                #     prev_predict_scores=cand_scores
                # elif suffix_index-2==len(suffix_ids)-1:
                #     prev_again_predict_scores=cand_scores
                if suffix_ids==None or suffix_ids==[]:
                    pass
                elif suffix_index<=len(suffix_ids)+1:
                    # print(len(combined_predicted_scores),suffix_index,len(suffix_ids)+1)
                    combined_predicted_scores[suffix_index+1]=cand_scores
                    
                

                #print("@@@@@@@@@@@@@@",tokens[:, : step + 1])
                #print(step)

            #print(tokens[:, : step + 1])
            # cand_bbsz_idx contains beam indices for the top can8didate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]


            cand_bbsz_idx = cand_beams.cuda().add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)

            #d=eos_mask[:, :beam_size][cands_to_ignore]
            if step<prefix_len:
                pass               
            else:
                eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            # second_time=time.time()
            # print(second_time-first_time)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []

            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                if combined_predicted_scores[-1]==[]:
                    combined_predicted_scores[-1]=combined_predicted_scores[-2]
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.


                
            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))

            #print(eos_mask)
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # if cands_to_ignore[0][-1]==True:
            #     print(step<prefix_len)
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
            
            # if predict_scores!=[]:
            #     active_predict_scores=torch.gather(predict_scores, dim=1, index=active_hypos)
            #     predict_scores=active_predict_scores
            # if prefix_predict_scores!=[]:
            #     active_prefix_predict_scores=torch.gather(prefix_predict_scores, dim=1, index=active_hypos)
            #     predict_prefix_scores=active_prefix_predict_scores                

            # if prev_predict_scores!=[]:
            #     active_prev_predict_scores=torch.gather(prev_predict_scores, dim=1, index=active_hypos)
            #     prev_predict_scores=active_prev_predict_scores

            # if prev_again_predict_scores!=[]:
            #     active_prev_again_predict_scores=torch.gather(prev_again_predict_scores, dim=1, index=active_hypos)
            #     prev_again_predict_scores=active_prev_again_predict_scores

            #******************************the first version of look_ahead***************************************
            #
            
            for i in range(len(combined_predicted_scores)):
                if combined_predicted_scores[i]!=[]:
                    if type(combined_predicted_scores[i])==list:
                        combined_predicted_scores[i][0]=torch.gather(combined_predicted_scores[i][0], dim=1, index=active_hypos)
                        combined_predicted_scores[i][1]=torch.gather(combined_predicted_scores[i][1], dim=1, index=active_hypos)
                    else:
                        combined_predicted_scores[i]=torch.gather(combined_predicted_scores[i], dim=1, index=active_hypos)

            for i in range(len(all_suffix_scores)):
                if all_suffix_scores[i]!=[]:
                    all_suffix_scores[i]=torch.gather(all_suffix_scores[i], dim=1, index=active_hypos.cpu())
            
            for i in range(len(all_suffix_scores_masks)):
                if all_suffix_scores_masks[i]!=[]:
                    all_suffix_scores_masks[i]=torch.gather(all_suffix_scores_masks[i], dim=1, index=active_hypos.cpu())

            # if step==prefix_len+1:
            #     import step==prefix_len
            #     pdb.set_trace()
            if step>=prefix_len+1:
                active_this_tokens_labels=torch.gather(this_tokens_labels, dim=1, index=active_hypos)
                this_tokens_labels=active_this_tokens_labels

            if step>=prefix_len+2:
                active_this_tokens_labels_again=torch.gather(this_tokens_labels_again, dim=1, index=active_hypos)
                this_tokens_labels_again=active_this_tokens_labels_again

            #*****************************end of the first version look_ahead*****************************

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)
            # if step==4:
            #     import pdb
            #     pdb.set_trace()

            next_tokens=tokens[:,:step+2]
            # print(next_tokens[0])

            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
                scores_with_suffix[:, :step] = torch.index_select(
                    scores_with_suffix[:, :step], dim=0, index=active_bbsz_idx
                )
                scores_with_suffix_masks[:, :step] = torch.index_select(
                    scores_with_suffix_masks[:, :step], dim=0, index=active_bbsz_idx
                )
                dynamic_scores[:step,:,:]=torch.index_select(dynamic_scores[:step,:,:],dim=1,index=active_bbsz_idx)
                # print(scores[:, :step])
                # print(scores_with_suffix[:, :step])
                # #print(scores_with_suffix_masks[:, :step])
                # print(self.tgt_dict.string(tokens[:,:step+1]))


            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )
            
            dynamic_scores[step,:,:]=torch.index_select(dynamic_states,dim=1,index=active_bbsz_idx).squeeze()

            finished_beams=torch.index_select(finished_beams, dim=1, index=active_bbsz_idx)
            finished_beams+=cands_to_ignore
            # print(finished_beams)
            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

            if reorder_state is not None:
                # print(reorder_state,batch_idxs)
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            all_suffix_scores_index=step-prefix_len

                

            if prefix_len==0 and beam_size==1 and max_aheads==0 and all_suffix_scores_index==0:
                
                first_position_ahead_scores=copy.deepcopy(scores_with_suffix)
                first_scores_with_suffix_masks=copy.deepcopy(scores_with_suffix_masks)
                first_finished_beams=copy.deepcopy(finished_beams)
                tmp_dict1=self.look_ahead(next_tokens[:,:-1],suffix_ids,step,incremental_states,encoder_outs,
                    first_position_ahead_scores,first_scores_with_suffix_masks,finished_beams=first_finished_beams,attn_len=attn_len)
                # all_suffix_scores[all_suffix_scores_index]=tmp_dict1["suffix_scores"]
                # all_suffix_scores_masks[all_suffix_scores_index]=tmp_dict1["suffix_scores_masks"]
                first_position_ahead_scores=tmp_dict1["scores_with_suffix"]
                #scores_with_suffix_masks=tmp_dict1["scores_with_suffix_masks"]   
            
            if max_aheads==None:
                scores_with_suffix[:,step]=scores[:,step]
            elif all_suffix_scores_index<0:
                # if step==2:
                #     import pdb
                #     pdb.set_trace()
                
                scores_with_suffix[:,step]=scores[:,step]
                if prefix_len!=0 and beam_size==1 and max_aheads==0 and all_suffix_scores_index==-1:

                    tmp_dict1=self.look_ahead(next_tokens,suffix_ids,step,incremental_states,encoder_outs,
                        scores_with_suffix,scores_with_suffix_masks,finished_beams=finished_beams,attn_len=attn_len)
                    # all_suffix_scores[all_suffix_scores_index]=tmp_dict1["suffix_scores"]
                    # all_suffix_scores_masks[all_suffix_scores_index]=tmp_dict1["suffix_scores_masks"]
                    scores_with_suffix=tmp_dict1["scores_with_suffix"]
                    scores_with_suffix_masks=tmp_dict1["scores_with_suffix_masks"]   

                    
            elif all_suffix_scores_index<max_aheads and all_suffix_scores_index>=0:

                scores_with_suffix[:,step]=scores[:,step]
                #scores_with_suffix[:,step]=scores[:,prefix_len]
                tmp_dict1=self.look_ahead_bart(next_tokens,suffix_ids,step,incremental_states,encoder_outs,
                    scores_with_suffix,scores_with_suffix_masks,finished_beams=finished_beams,attn_len=attn_len)
                # all_suffix_scores[all_suffix_scores_index]=tmp_dict1["suffix_scores"]
                # all_suffix_scores_masks[all_suffix_scores_index]=tmp_dict1["suffix_scores_masks"]
                scores_with_suffix=tmp_dict1["scores_with_suffix"]
                scores_with_suffix_masks=tmp_dict1["scores_with_suffix_masks"]
            elif all_suffix_scores_index>=max_aheads:
                scores_with_suffix[:,step]=scores[:,step]
            # print(scores_with_suffix_masks[:,:step+1])

        #print(next_tokens)
        #print(next_tokens.shape)
        if first_position_ahead_scores!=None:
            print("facing the prefix length is zero,you should use the dynamic_scores!!!")
            dynamic_scores=first_position_ahead_scores
        this_tokens_labels=1
        this_tokens_labels_again=1
        combined_predicted_scores=1
        return [next_tokens], combined_predicted_scores,this_tokens_labels,this_tokens_labels_again,scores_with_suffix,scores_with_suffix_masks,dynamic_scores
        # sort by score descending
        
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized,predict_scores





    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.min(prefix_lprobs) - 1
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix

            print("I am facing a eos in prefix_tokens,maybe error")
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )
        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = bbsz_idx // beam_size
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()
        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, len(finalized[unique_sent]), beam_size
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        unfin_idx: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
            for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models if hasattr(m, "max_decoder_positions")] + [sys.maxsize])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
        attn_len: int = -1,
        tgt_token: int = -1,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None

        
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():

                #pdb.set_trace()
                if attn_len == -1:
                    decoder_out = model.decoder.forward(
                        tokens,
                        encoder_out=encoder_out,
                        incremental_state=incremental_states[i],
                    )
                else:
                    #pdb.set_trace()
                    decoder_out = model.decoder.forward(
                        tokens,
                        encoder_out=encoder_out,
                        incremental_state=incremental_states[i],
                        attn_len=attn_len,
                        tgt_token=tgt_token,
                    )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)
            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)

            #print("$$$$$$$$$$$$$$$4")
            #print(decoder_out)
            #print("$$$$$$$$$$$$$$$4")
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            #pdb.set_trace()
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn




    @torch.jit.export
    def forward_decoder_inner(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
        attn_len: int = -1,
        tgt_token: int = -1,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None

        
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():

                #pdb.set_trace()
                if attn_len == -1:
                    decoder_out = model.decoder.forward(
                        tokens,
                        encoder_out=encoder_out,
                        incremental_state=incremental_states[i],
                    )
                else:
                    #pdb.set_trace()
                    decoder_out = model.decoder.forward(
                        tokens,
                        encoder_out=encoder_out,
                        incremental_state=incremental_states[i],
                        attn_len=attn_len,
                        tgt_token=tgt_token,
                    )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)
            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)

            #print("$$$$$$$$$$$$$$$4")
            #print(decoder_out)
            #print("$$$$$$$$$$$$$$$4")
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            #pdb.set_trace()
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn,decoder_out[1]["inner_states"][-1]

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn, decoder_out[1]["inner_states"][-1]


    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


class SequenceGeneratorWithAlignment(SequenceGenerator):
    def __init__(
        self, models, tgt_dict, left_pad_target=False, print_alignment="hard", **kwargs
    ):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

        if print_alignment == "hard":
            self.extract_alignment = utils.extract_hard_alignment
        elif print_alignment == "soft":
            self.extract_alignment = utils.extract_soft_alignment

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        (
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
        ) = self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, "full_context_alignment", False) for m in self.model.models):
            attn = self.model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]["attention"].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        if src_tokens.device != "cpu":
            src_tokens = src_tokens.to("cpu")
            tgt_tokens = tgt_tokens.to("cpu")
            attn = [i.to("cpu") for i in attn]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = self.extract_alignment(
                attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos
            )
            finalized[i // beam_size][i % beam_size]["alignment"] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :]
            .expand(-1, self.beam_size, -1)
            .contiguous()
            .view(bsz * self.beam_size, -1)
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
            .expand(-1, self.beam_size)
            .contiguous()
            .view(bsz * self.beam_size)
        )
        prev_output_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]["attn"][0]
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn
