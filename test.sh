#!/bin/bash
python run.py --model RNAJog --input_type rna --data_path data/test/rna.txt --codon_usage_freq_table_path "codon_usage_freq_table_human.csv" --mfe_weight 0.15 --sample_method sample --sample_size 1 --sample_temperature 0.01 --save_path result
