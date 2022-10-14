#!/bin/bash


TEXT=data/wow
DATA=data-bin/wow/
#fairseq-preprocess --source-lang src --target-lang tgt \
#  --trainpref $TEXT/train \
#  --validpref $TEXT/valid \
#  --testpref $TEXT/test.seen,$TEXT/test.unseen  \
#  --destdir $DATA \
#  --workers 8 --joined-dictionary

ARCH=transformer_iwslt_de_en
CKPT_DIR=checkpoints/wow_transformer/
mkdir -p $CKPT_DIR

python train.py $DATA --task translation --arch $ARCH \
--share-all-embeddings --dropout 0.3 --warmup-updates 2000 \
--lr 0.0005 --min-lr 1e-09 --max-tokens 4096  --max-update 50000 \
--target-lang tgt --source-lang src --save-dir $CKPT_DIR \
--find-unused-parameters --patience 10 \
--optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0  \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 