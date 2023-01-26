#!/bin/bash

stage=4
end_stage=5

n_cluster=50
lab_dir=~/scratch/exp/pnmi_exp/km_lab/speech_dino_local_5_$n_cluster
exp_dir=~/scratch/exp/slm_exp/speech_dino_local_5_$n_cluster
corpus_dir=~/scratch/dataset/zs21dataset

echo "==== Clusters: $n_cluster ===="

mkdir -p $exp_dir
mkdir -p $exp_dir/txt
mkdir -p $exp_dir/data
mkdir -p $exp_dir/submission_lstm
mkdir -p $exp_dir/result_lstm

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    echo "==== Stage 1: Convert to fairseq ===="
    for split in train dev test
    do
        python3 scripts/convert_for_fairseq.py \
            $lab_dir/zs21_${split}_labels.txt \
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
    echo "==== Stage 3: LSTM Training ===="
    fairseq-train --fp16 \
        $exp_dir/data \
        --task language_modeling \
        --save-dir $exp_dir/lm-lstm-ls960 \
        --keep-last-epochs 2 \
        --tensorboard-logdir tensorboard \
        --arch lstm_lm \
        --decoder-embed-dim 200 \
        --decoder-hidden-size 1024 \
        --decoder-layers 3 \
        --decoder-out-embed-dim 200 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --lr 0.0005 \
        --warmup-updates 1000 \
        --warmup-init-lr 1e-07 \
        --dropout 0.1 \
        --weight-decay 0.01 \
        --sample-break-mode none \
        --tokens-per-sample 2048 \
        --max-tokens 131072 \
        --update-freq 1 \
        --max-update 100000 \
        | tee $exp_dir/train.log
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
    echo "==== Stage 4: Extract LM pseudo-probabilities ===="
    for task in lexical syntactic
    do
        for split in dev # test
        do
            mkdir -p $exp_dir/submission_lstm/$task
            
            python3 scripts/compute_proba_LSTM.py \
                $lab_dir/$task/zs21_$split.txt \
                $exp_dir/submission_lstm/$task/$split.txt \
                $exp_dir/lm-lstm-ls960/checkpoint_best.pt \
                --dict $exp_dir/data/dict.txt
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

    zerospeech2021-evaluate $corpus_dir $exp_dir/submission_lstm -o $exp_dir/result_lstm \
        --no-phonetic --no-semantic
    
    echo "==== Lexical ===="
    cat $exp_dir/result_lstm/score_lexical_dev_by_frequency.csv

    echo "==== Syntactic ===="
    cat $exp_dir/result_lstm/score_syntactic_dev_by_type.csv
fi
