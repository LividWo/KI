#!/bin/bash

usage(){
        echo "
        Usage:
         -p, --path             checkpoint dir
         -s, --split            dataset split
        "
}
while getopts ":p:g:s:" arg; do
        case "${arg}" in
                p)
                        CKPT_DIR=${OPTARG}
                        echo "dir:              $CKPT_DIR"
                        ;;
                g)
                        GPUS=${OPTARG}
                        echo "using gpus:       $GPUS"
                        ;;
                s)
                        SPLIT=${OPTARG}
                        echo "evaluating on the split: $SPLIT"
                        ;;
                *)
                        usage
                        ;;
        esac
done

generate="${CKPT_DIR}/generate-${SPLIT}.txt"
score="${CKPT_DIR}/score-${SPLIT}.txt"

python experiments/bleu_score.py --input $generate

python experiments/ppl.py --file $score

python experiments/metrics.py --file $generate

python experiments/safe.py --file $generate
