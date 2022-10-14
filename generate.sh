#!/bin/bash

usage(){
        echo "
        Usage:
         -p, --path             checkpoint dir
         -s, --split            dataset split
         -b, --beam             beam size
         -d, --data             data set
         -c, --checkpoint     
        "
}
while getopts ":p:g:s:b:d:c:" arg; do
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
                b)
                        BEAM=${OPTARG}
                        echo "beam size: $BEAM"
                        ;;
                d)
                        DATA=${OPTARG}
                        echo "data set: $DATA"
                        ;;
                c)
                        CKP=${OPTARG}
                        echo "checkpoint: $CKP"
                        ;;
                *)
                        echo "unexpected parameter"
                        usage
                        ;;
        esac
done

if [ ! -f "$CKPT_DIR/checkpoint_last10_avg.pt" ]; then
python scripts/average_checkpoints.py \
--inputs $CKPT_DIR \
--output $CKPT_DIR/checkpoint_last10_avg.pt \
--num-epoch-checkpoints  10 #   --checkpoint-upper-bound 20
fi

fairseq-generate $DATA --beam $BEAM --batch-size 128 --remove-bpe ' ##' --gen-subset  $SPLIT --path $CKPT_DIR/$CKP --results-path $CKPT_DIR

fairseq-generate $DATA --beam $BEAM --batch-size 128 --remove-bpe ' ##' --gen-subset $SPLIT --path $CKPT_DIR/$CKP --results-path $CKPT_DIR --gen-type score --score-reference
