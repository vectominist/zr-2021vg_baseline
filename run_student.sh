#!/bin/bash

stage=4
end_stage=6
run_lex=$2  # run lexical if 1 else skip lexical
run_syn=$3  # run syntactic if 1 else skip syntactic
run_sem=$4  # run semantic if 1 else skip semantic

layer=$1

n_cluster=500
lab_dir=~/scratch/exp/pnmi_exp/km_lab/speech_dino_local_5_${n_cluster}
exp_dir=~/scratch/exp/slm_exp/speech_dino_local_5_${n_cluster}_3
corpus_dir=~/scratch/dataset/zs21dataset
lm_name=lm-bert-small-ls960

echo "==== Clusters: $n_cluster ===="

M=10
Dt=1
level=-1

mkdir -p $exp_dir
mkdir -p $exp_dir/txt
mkdir -p $exp_dir/data
mkdir -p $exp_dir/submission_${M}_${Dt}
mkdir -p $exp_dir/result_${M}_${Dt}

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
    echo "==== Stage 3: BERT-small Training ===="
    SPAN_SIZE=10
    MAX_TOKENS=16384
    fairseq-train --fp16 \
        $exp_dir/data \
        --save-dir $exp_dir/$lm_name \
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
        | tee $exp_dir/$lm_name/train.log
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ] && [ $run_lex -ge 1 ]; then
    echo "==== Stage 4: Extract LM pseudo-probabilities & Evaluate (Lexical) ===="
    for split in dev # test
    do
        mkdir -p $exp_dir/submission/lexical
        
        python3 scripts/compute_proba_BERT.py \
            $lab_dir/lexical/zs21_$split.txt \
            $exp_dir/submission_${M}_${Dt}/lexical/$split.txt \
            $exp_dir/$lm_name/checkpoint_best.pt \
            --dict $exp_dir/data/dict.txt \
            --decoding_span_size $M --temporal_sliding_size $Dt \
            --batchsen_size 32
    done

    zerospeech2021-evaluate $corpus_dir $exp_dir/submission_${M}_${Dt} \
        -o $exp_dir/result_${M}_${Dt} \
        --no-phonetic --no-semantic --no-syntactic
    
    echo "==== Lexical ===="
    echo "File: $exp_dir/result_${M}_${Dt}/score_lexical_dev_by_frequency.csv"
    cat $exp_dir/result_${M}_${Dt}/score_lexical_dev_by_frequency.csv
fi

if [ $stage -le 5 ] && [ $end_stage -ge 5 ] && [ $run_syn -ge 1 ]; then
    echo "==== Stage 5: Extract LM pseudo-probabilities & Evaluate (Syntactic) ===="
    for split in dev # test
    do
        mkdir -p $exp_dir/submission/syntactic
        
        python3 scripts/compute_proba_BERT.py \
            $lab_dir/syntactic/zs21_$split.txt \
            $exp_dir/submission_${M}_${Dt}/syntactic/$split.txt \
            $exp_dir/$lm_name/checkpoint_best.pt \
            --dict $exp_dir/data/dict.txt \
            --decoding_span_size $M --temporal_sliding_size $Dt \
            --batchsen_size 32
    done

    zerospeech2021-evaluate $corpus_dir $exp_dir/submission_${M}_${Dt} \
        -o $exp_dir/result_${M}_${Dt} \
        --no-phonetic --no-semantic --no-lexical
    
    echo "==== Syntactic ===="
    echo "File: $exp_dir/result_${M}_${Dt}/score_syntactic_dev_by_type.csv"
    cat $exp_dir/result_${M}_${Dt}/score_syntactic_dev_by_type.csv
fi

if [ $stage -le 6 ] && [ $end_stage -ge 6 ] && [ $run_sem -ge 1 ]; then
    echo "==== Stage 6: Dump BERT features for semantic task & Evaluate (level = $level) ===="
    
    for corpus in librispeech synthetic
    do
        mkdir -p $exp_dir/submission_${M}_${Dt}/semantic/dev/$corpus
        python3 scripts/build_BERT_features.py \
            $lab_dir/semantic/$corpus/zs21_dev.txt \
            $exp_dir/submission_${M}_${Dt}/semantic/dev/$corpus \
            $exp_dir/$lm_name/checkpoint_best.pt \
            --dict $exp_dir/data/dict.txt \
            --hidden_level $level
    done

    pwd=`pwd`
    cd $exp_dir/submission_${M}_${Dt}
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

    zerospeech2021-evaluate $corpus_dir $exp_dir/submission_${M}_${Dt} \
        -o $exp_dir/result_${M}_${Dt} \
        --no-phonetic --no-lexical --no-syntactic

    echo "==== Semantic ===="
    echo "File: $exp_dir/result_${M}_${Dt}/score_semantic_dev_correlation.csv"
    cat $exp_dir/result_${M}_${Dt}/score_semantic_dev_correlation.csv
fi

# if [ $stage -le 6 ] && [ $end_stage -ge 6 ]; then
#     echo "==== Stage 6: Evaluation ===="
#     pwd=`pwd`
#     cd $exp_dir/submission
#     cat > meta.yaml <<'EOF'
# author: Heng-Jui Chang
# affiliation: MIT CSAIL
# description: High-budget
# open_source: True
# train_set: Librispeech
# gpu_budget: 72.0
# parameters:
#   phonetic:
#     metric: cosine
#     frame_shift: 0.02
#   semantic:
#     metric: cosine
#     pooling: max
# EOF
#     mkdir -p code
#     echo "https://github.com/bhigy/zr-2021vg_baseline" > code/README
#     cd $pwd

#     zerospeech2021-evaluate $corpus_dir $exp_dir/submission -o $exp_dir/result --no-phonetic
    
#     echo "==== Lexical ===="
#     cat $exp_dir/result/score_lexical_dev_by_frequency.csv

#     echo "==== Syntactic ===="
#     cat $exp_dir/result/score_syntactic_dev_by_type.csv

#     echo "==== Semantic ===="
#     cat $exp_dir/result/score_semantic_dev_correlation.csv
# fi
