# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import h5py
import numpy as np
from collections import Counter
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def hinge(x):
    return torch.clamp(x, min=0.)


def voken_paired_hinge_loss(lang_output, voken_embedding, margin, lang_mask, voken_label):
    '''
    :param lang_output: [batch_size, max_len, dim]
    :param voken_embedding: [batch_size, max_len-1, dim] without EOS token in the end
    :param margin: margin parameter
    :param lang_mask: [batch_size, max_len]
    :return: a scalar value loss
    '''
    # lang_output = net_output[-1]['voken']
    # lang_output = lang_output[:, :-1, :]  # delete last one, which is an extra EOS token </s>
    # lang_mask = lang_mask[:, :-1]  # do the same
    if lang_output.shape[0] == 1:
        #print(lang_output.shape)
        lang_output = lang_output.squeeze()
        voken_embedding = voken_embedding.squeeze()
        voken_label = voken_label.squeeze()
        #print(lang_output.shape, voken_embedding.shape)
        voken_label_np = voken_label.detach().cpu().numpy()
        duplicate = [item for item, count in Counter(voken_label_np).items() if count > 1]
        mask = torch.zeros(lang_output.shape[0], dtype=torch.bool)
        for i in range(lang_output.shape[0]):
            if voken_label_np[i] in duplicate:
                mask[i] = True

        dot_product = torch.matmul(lang_output, voken_embedding.t())
        #print(dot_product.shape)
        position = torch.eye(lang_output.shape[0]).to(dot_product.device)
        loss = torch.nn.functional.log_softmax(dot_product, dim=-1) * position
        loss = (-loss.sum(dim=1)).masked_fill_(mask.to(loss.device), 0.0).mean()
        #print(loss)
        return loss

    if lang_output.shape[0] % 2 != 0:
        # print("drop last sentence", lang_output.shape, voken_embedding.shape)
        lang_output = lang_output[:-1, :, :]
        lang_mask = lang_mask[:-1, :]
        voken_embedding = voken_embedding[:-1, :, :]
        voken_label = voken_label[:-1, :]
    batch_size, lang_len, dim = lang_output.shape
    assert batch_size % 2 == 0
    assert batch_size == voken_embedding.shape[0]
    assert lang_len == voken_embedding.shape[1]
    assert margin > 0.

    # voken_embedding [b, t, dim]
    half_batch_size = batch_size // 2
    pos_lang, neg_lang = torch.split(lang_output, half_batch_size, dim=0)
    pos_know, neg_know = torch.split(voken_embedding, half_batch_size, dim=0)
    pos_label, neg_label = torch.split(voken_label, half_batch_size, dim=0)
    mask_pos_equal_neg  = pos_label.eq(neg_label)

    true_pos_score = (pos_lang * pos_know).sum(-1)  # [b / 2, t]
    true_neg_score = (neg_lang * neg_know).sum(-1)
    false_pos_score = (pos_lang * neg_know).sum(-1)
    false_neg_score = (neg_lang * pos_know).sum(-1)
    # false_neg_score = neg_lang * pos_know
    # false_neg_score.masked_fill_(mask_pos_equal_neg, 0.0)
    # false_neg_score.sum_(-1)
    # token-level hinge loss
    pos_mask, neg_mask = torch.split(lang_mask, half_batch_size, dim=0)
    pos_loss = hinge(margin - true_pos_score + false_pos_score) * pos_mask
    neg_loss = hinge(margin - true_neg_score + false_neg_score) * neg_mask

    if mask_pos_equal_neg.any():
        pos_loss.masked_fill_(mask_pos_equal_neg, 0.0)
        neg_loss.masked_fill_(mask_pos_equal_neg, 0.0)
    # averaging
    cnt = lang_mask.sum()
    # loss = (pos_loss.sum() + neg_loss.sum()) / cnt
    loss = pos_loss.sum() + neg_loss.sum()

    return loss


def generate_voken_mask(voken_label):
    pass


@register_criterion("voken_label_smoothed_cross_entropy")
class VokenLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        margin,
        knowledge_embedding_file,
        voken_weight,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.margin = margin
        self.voken_weight = voken_weight
        print('loding extracted knowledge embedding/keys from:', knowledge_embedding_file)
        h5_file = h5py.File(knowledge_embedding_file, 'r')
        dset = h5_file["keys"]

        voken_embed = dset[:]
        assert len(voken_embed) == len(dset)
        h5_file.close()

        know_vocab, self.know_dim = voken_embed.shape
        embeddings_matrix = np.zeros((know_vocab + 2, self.know_dim))
        embeddings_matrix[2:] = voken_embed
        self.know_embeddings = torch.FloatTensor(embeddings_matrix)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on
        parser.add_argument('--knowledge-embedding-file', type=str, metavar='STR',
                            help='path of extracted knowledge embedding/keys')
        parser.add_argument('--margin', default=0.5, type=float)
        parser.add_argument('--voken-weight', default=1.0, type=float)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss, voken_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            'voken_loss': voken_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        voken_label = sample['voken']
        voken_embedding = self.know_embeddings[voken_label].to(net_output[-1]['voken'].device)  # [b, t, d]

        target = model.get_targets(sample, net_output)
        #mask = target.ne(self.padding_idx).long()  # padding on the decoder side
        mask = sample['net_input']['src_tokens'].ne(self.padding_idx).long()  # padding on encoder tokens
        voken_loss = voken_paired_hinge_loss(net_output[-1]['voken'], voken_embedding, self.margin, mask, voken_label)
        loss += self.voken_weight * voken_loss
        return loss, nll_loss, voken_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        voken_loss_sum = sum(log.get("voken_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "voken_loss", voken_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
