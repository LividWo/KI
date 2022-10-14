# copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BLEU scoring of generated translations against reference translations.
"""

import argparse
import os
import sys

from fairseq.data import dictionary
from fairseq.scoring import bleu


def read_file(path):
    sys_toks = {}
    ref_toks = {}

    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            head = line[0][:2]
            if head == 'H-':
                idx = int(line[0][2:])
                sys_toks[idx] = line[2]

            if head == 'T-':
                idx = int(line[0][2:])
                ref_toks[idx] = line[1]
    sys, ref = [], []
    for k,v in sys_toks.items():
        sys.append(v)
        ref.append(ref_toks[k])
    return sys, ref


def read_diffks(path):
    sys_toks, ref_toks = {}, {}
    cnt = 0
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            #if line.startswith('post:'):
            #    sys[cnt] = line.split('\t')[1]
            if line.startswith('resp:'):
                ref_toks[cnt] = line.split('\t')[1]
            if line.startswith('gen:'):
                sys_toks[cnt] = line.split('\t')[1]
                cnt += 1
    sys, ref = [], []
    for k,v in sys_toks.items():
        sys.append(v)
        ref.append(ref_toks[k])
    return sys, ref


def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for BLEU scoring."
    )
    # fmt: off
    parser.add_argument('-i', '--input', default='trans-generate-test.txt', help='generate.py output')
    parser.add_argument('-o', '--order', default=4, metavar='N',
                        type=int, help='consider ngrams up to this order')
    parser.add_argument('--ignore-case', action='store_true',
                        help='case-insensitive scoring')
    parser.add_argument('--sacrebleu', action='store_true',
                        help='score with sacrebleu')
    parser.add_argument('--sentence-bleu', action='store_true',
                        help='report sentence-level BLEUs (i.e., with +1 smoothing)')
    parser.add_argument('--fname', type=str, default='fairseq')    # fmt: on
    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    # print(args)

    dict = dictionary.Dictionary()

    sys_toks, ref_toks = read_file(args.input)
    if args.fname == 'diffks':
        sys_toks, ref_toks = read_diffks(args.input)

    if args.sacrebleu:
        import sacrebleu

        def score(fdsys):
            with open(args.ref) as fdref:
                print(sacrebleu.corpus_bleu(fdsys, [fdref]))

    elif args.sentence_bleu:

        scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
        for i, (sys_tok, ref_tok) in enumerate(
            zip(sys_toks, ref_toks)
        ):
            scorer.reset(one_init=True)
            sys_tok = dict.encode_line(sys_tok)
            ref_tok = dict.encode_line(ref_tok)
            scorer.add(ref_tok, sys_tok)
            print(i, scorer.result_string(args.order))

    else:
        scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
        for sys_tok, ref_tok in zip(sys_toks, ref_toks):
            sys_tok = dict.encode_line(sys_tok)
            ref_tok = dict.encode_line(ref_tok)
            scorer.add(ref_tok, sys_tok)
        for i in range(1, args.order + 1):
            print(scorer.result_string(i))



if __name__ == "__main__":
    cli_main()
