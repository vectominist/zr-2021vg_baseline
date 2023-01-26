#!/bin/bash

stage=4
end_stage=5

lab_dir=~/scratch/exp/pnmi_exp/km_lab/speech_dino_local_5_500
lex_lab_dir=~/scratch/exp/pnmi_exp/km_lab/speech_dino_local_5_500/lexical
syn_lab_dir=~/scratch/exp/pnmi_exp/km_lab/speech_dino_local_5_500/syntactic
exp_dir=~/scratch/exp/slm_exp/speech_dino_local_5_500_merged
corpus_dir=~/scratch/dataset/zs21dataset

mkdir -p $exp_dir
mkdir -p $exp_dir/txt
mkdir -p $exp_dir/data
mkdir -p $exp_dir/submission
mkdir -p $exp_dir/result

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    echo "==== Stage 1: Convert to fairseq ===="
    for split in train dev test
    do
        python3 scripts/convert_for_fairseq.py \
            $lab_dir/zs21_${split}_labels_merged.txt \
            $exp_dir/txt/fairseq_${split}.txt
    done
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    echo "==== Stage 2: Fairseq pre-process ===="
    fairseq-preprocess --only-source \
        --trainpref $exp_dir/txt/fairseq_train.txt \
        --validpref $exp_dir/txt/fairseq_dev.txt   \
        --testpref  $exp_dir/txt/fairseq_test.txt  \
        --destdir $exp_dir/data \
        --workers 8
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
    echo "==== Stage 3: BERT-small Training ===="
    SPAN_SIZE=5 # equivalent to 100 ms
    MAX_TOKENS=4096
    fairseq-train --fp16 \
        $exp_dir/data \
        --save-dir $exp_dir/lm-bert-small-ls960 \
        --task masked_lm \
        --keep-last-epochs 1 \
        --tensorboard-logdir tensorboard \
        --train-subset train \
        --num-workers 4 \
        --criterion masked_lm \
        --arch roberta_base \
        --sample-break-mode eos --tokens-per-sample 3072 --max-positions 6144 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr 0.0005 --total-num-update 250000 --warmup-updates 10000 \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --mask-multiple-length $SPAN_SIZE --mask-prob 0.5 --mask-stdev $SPAN_SIZE \
        --max-tokens $MAX_TOKENS --max-update 250000 \
        --encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 8 --encoder-layers 8 \
        --seed 5 --log-format simple --log-interval 100 --skip-invalid-size-inputs-valid-test \
        | tee $exp_dir/train.log
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
    echo "==== Stage 4: Extract LM pseudo-probabilities ===="
    for task in lexical syntactic
    do
        for split in dev # test
        do
            mkdir -p $exp_dir/submission/$task
            
            python3 scripts/compute_proba_BERT.py \
                ~/scratch/exp/pnmi_exp/code_lab_$task/speech_dino_0/zs21_$split.txt \
                $exp_dir/submission/$task/$split.txt \
                $exp_dir/lm-bert-small-ls960/checkpoint_best.pt \
                --dict $exp_dir/data/dict.txt \
                --decoding_span_size 10 --temporal_sliding_size 1
        done
    done
fi

if [ $stage -le 5 ] && [ $end_stage -ge 5 ]; then
    echo "==== Stage 5: Evaluation ===="
    pwd=`pwd`
    cd $exp_dir/result
    cat > meta.yaml <<'EOF'
author: Heng-Jui Chang
affiliation: MIT CSAIL
description: High-budget
open_source: True
train_set: Librispeech
gpu_budget: 72.0
parameters:
  phonetic:
    metric: cosine
    frame_shift: 0.02
  semantic:
    metric: cosine
    pooling: max
EOF
    mkdir -p code
    echo "https://github.com/bhigy/zr-2021vg_baseline" > code/README
    cd $pwd

    zerospeech2021-evaluate $corpus_dir $exp_dir/submission -o $exp_dir/result \
        --no-phonetic --no-semantic
    
    echo "==== Lexical ===="
    cat $exp_dir/result/score_lexical_dev_by_frequency.csv

    echo "==== Syntactic ===="
    cat $exp_dir/result/score_syntactic_dev_by_type.csv
fi
