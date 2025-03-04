import RNA
import concurrent.futures
import torch
import time
import math
import pandas as pd
from utils.constant import trans_dict, MASK, id2pro


def calculate_mfe(sequence):
    mfe = RNA.fold(sequence)[1]
    return mfe


def calculate_mfe_mul(sequences, max_processes):
    mfe_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        future_to_sequence = {executor.submit(calculate_mfe, seq): seq for seq in sequences}
        

        concurrent.futures.wait(future_to_sequence)
        

        for sequence in sequences:
            future = [f for f in future_to_sequence if future_to_sequence[f] == sequence][0]
            try:
                mfe = future.result()
                mfe_results.append(mfe)  
            except Exception as e:
                print(f"Error calculating MFE for sequence {sequence}: {str(e)}")
    
    return mfe_results


def calculate_cai(rna_seqs, cai_table):
    cais = []
    for seq in rna_seqs:
        if len(seq) % 3 != 0:
            raise ValueError("The length of the sequence must be a multiple of 3, got {}".format(len(seq)))
        log_cai = 0
        for i in range(0, len(seq), 3):
            codon = seq[i:i+3]
            log_cai += calculate_cai_single(codon, cai_table)
        log_cai = log_cai/(len(seq)/3)
        cais.append(math.exp(log_cai))
    return cais

def calculate_cai_single(codon, cai_table):
    # cai_table = pd.read_csv("./codon_usage_freq_table_ecoli.csv")
    aa = cai_table[cai_table.iloc[:, 0] == codon].iloc[:, 1].values[0]
    rate = cai_table[cai_table.iloc[:, 0] == codon].iloc[:, 2].values[0]
    max_rate = cai_table[cai_table.iloc[:, 1] == aa].iloc[:, 2].values.max()
    log_cai = math.log(rate/max_rate)
    return log_cai

def get_cai_prob(cai_table):   
    cai_prob_table = torch.zeros_like(MASK, dtype=torch.float32)
    for i in range(21):
        total = 0
        for j in range(6):
            if MASK[i, j]:
                cai = math.exp(calculate_cai_single(trans_dict[id2pro[i]][j], cai_table))
                cai_prob_table[i, j] = cai
                total += cai
        cai_prob_table[i] /= total

    return cai_prob_table

