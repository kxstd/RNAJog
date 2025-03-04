from utils.constant import id2pro, trans_dict, codon2pro
import numpy as np
import copy
from tqdm import tqdm
from multiprocessing import Pool

def amino_gen(seed,size,length):
    np.random.seed(seed)
    id2pro_n = copy.deepcopy(id2pro)
    id2pro_n.remove("*")
    id2pro_n.remove("M")
    amino = np.random.choice(id2pro_n,(length-2)*size) 
    amino = amino.reshape(size,length-2)
    stops = np.array(["*" for _ in range(size)]).reshape(size,1)
    amino = np.append(amino,stops,axis = 1)
    starts = np.array(["M" for _ in range(size)]).reshape(size,1)
    amino = np.append(starts, amino, axis = 1)
    return amino.tolist()

def amino2codon_helper(args):
    seed, sequence = args
    np.random.seed(seed)
    return [np.random.choice(trans_dict[ami]) for ami in sequence]

def amino2codon(seed, amino):
    if len(amino) >= 2000:
        with Pool() as pool:
            args_list = [(seed, sequence) for sequence in amino]
            results = list(tqdm(pool.imap(amino2codon_helper, args_list), total=len(args_list)))
        return results
    else:
        np.random.seed(seed)
        codon_seqs = []
        for sequence in amino:
            codon_seq = [np.random.choice(trans_dict[ami]) for ami in sequence]
            codon_seqs.append(codon_seq)
        return codon_seqs

def codon_gen(seed,size,length):
    amino_seqs = amino_gen(seed,size,length)
    codon_seqs = amino2codon(seed,amino_seqs)
    return codon_seqs, amino_seqs

def pro2case(seed, aminos):
    amino_seqs = [list(amino) for amino in aminos]
    codon_seqs = amino2codon(seed ,amino_seqs)
    length = [len(amino) for amino in amino_seqs]
    return codon_seqs, amino_seqs, length

def cod2case(seed, codons):
    codon_seqs = [[codon[i:i+3].replace("T", "U") for i in range(0, len(codon), 3)] for codon in codons]
    length = [len(codon_seq) - codon_seq.count("???") for codon_seq in codon_seqs]
    codon_seqs = [[codon if codon != "???" else "[PAD]" for codon in codon_seq] for codon_seq in codon_seqs]
    amino_seqs = [[codon2pro[codon] if codon != "[PAD]" else "[PAD]" for codon in codon_seq] for codon_seq in codon_seqs]
    return codon_seqs, amino_seqs, length

def cod2case_mul(seed, codons, num):
    '''
    Only for batch size == 1 (len(codons) == 1)
    '''
    codon_seqs_ori = [[codon[i:i+3].replace("T", "U") for i in range(0, len(codon), 3)] for codon in codons]
    amino_seq = [[codon2pro[codon] for codon in codon_seq] for codon_seq in codon_seqs_ori]
    amino_seqs = [amino_seq[0] for _ in range(num)]
    codon_seqs = amino2codon(seed, amino_seqs)

    return codon_seqs, amino_seqs

def check(amnio, codons):
    trans_amnio = ""
    flag = True
    if len(amnio) != len(codons)/3:
        return False
    for i in range(len(amnio)):
        if codon2pro[codons[i*3:i*3+3]] != amnio[i]:
            print(i)
            print(codon2pro[codons[i*3:i*3+3]])
            print(amnio[i])
            flag = False
    return flag


    