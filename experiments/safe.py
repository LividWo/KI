import os
import argparse


def read_fairseq(path):
    sys_toks = {}
    ref_toks = {}
    post_toks = {}

    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            head = line[0][:2]
            if head == 'S-':
                idx = int(line[0][2:])
                post_toks[idx] = line[1]

            if head == 'H-':
                idx = int(line[0][2:])
                sys_toks[idx] = line[2]

            if head == 'T-':
                idx = int(line[0][2:])
                ref_toks[idx] = line[1]
    return sys_toks


def get_safe_response_rate(response):
    total, safe = 0, 0
    for k,v in response.items():
        total += 1
        if "i ' m not sure" in v or "i do know" in v:
            safe += 1
    print("safe response rate:", safe/total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default='data/trans/voken-generate-test.txt', type=str)

    args = parser.parse_args()
    get_safe_response_rate(read_fairseq(args.file))

