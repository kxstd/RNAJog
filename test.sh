#!/bin/bash
python run.py --model RNAJog \
    --input_type rna \
    --data_path data/test/rna.txt \
    --mfe_weight 0.15 \
    --sample_method sample \
    --sample_size 4 \
    --sample_temperature 0.01 \
    --save_path result \
    --ban_seqs "CUCGAG;GCUCUUC"\
    --ban_m6a \
    --gc_repress 0.3
    