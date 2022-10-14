
from collections import Counter
import re
import math
import argparse


def read_file(path):
    sys_toks = {}
    ref_toks = {}

    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            head = line[0][:2]
            if head == 'H-':
                idx = int(line[0][2:])
                sys_toks[idx] = line[2]

            if head == 'T-':
                idx = int(line[0][2:])
                ref_toks[idx] = line[1]

    return sys_toks, ref_toks


def perplexity(path):
    cnt = 0
    nll_score = 0.
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('P-'):
                 line = line.split('\t')[-1]
                 position_score = [float(x) for x in line.split()]
                 nll_score += sum(position_score)
                 cnt += len(position_score)
            #if line.startswith('H-'):
            #    line = line.split('\t')[1]
            #    nll_score += float(line) / math.log(2)
            #    cnt += 1
    print("nll loss:", nll_score, -nll_score/cnt)
    print("perplexity: ", math.pow(2, -nll_score/cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default='data/trans/generate-test1.txt', type=str)

    args = parser.parse_args()
    perplexity(args.file)


