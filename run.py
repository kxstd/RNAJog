from omegaconf import OmegaConf
from os import path
import os
import torch
import time
import pandas as pd
import argparse
import sys
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from model.actor_critic import Actor_Critic
from model.cai_model import Cai_model
from utils.constant import codon2pro, ban_table_gen
from utils.random_gen import amino2codon
from utils.random_gen import pro2case, cod2case

from utils.evaluate import calculate_mfe_mul, calculate_cai, calculate_motif_rate 
from utils.dataset import load_data

MFE_MODEL_PATH = {"RNAJog":"./save/rnajog/model.pt",
                  "RNAJog_zero":"./save/rnajog_zero/model500.pt"}

def parse_args():
    parser = argparse.ArgumentParser(description="RNAJog")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="RNAJog", choices=["RNAJog", "RNAJog_zero"])
    parser.add_argument("--input_type", type=str, default="rna", choices=["rna", "protein"])
    parser.add_argument("--data_path", type=str, default="upload/")
    parser.add_argument("--codon_usage_freq_table_path", type=str, default="./codon_usage_freq_table_human_20250902.csv")
    parser.add_argument("--mfe_weight", type=float, default=0.15)
    parser.add_argument("--sample_method", type=str, default="sample", choices=["greedy", "sample"])
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default="result/")
    parser.add_argument("--ban_seqs", type=str, default="", help="User defined banned sequences, separated by ;")
    parser.add_argument("--ban_m6a", action="store_true", help="Whether to enable m6A motif elimination")
    parser.add_argument("--m6a_motif_path", type=str, default="motif.json", help="Path to m6A motif json file")
    parser.add_argument("--gc_repress", type=float, default=None, help="GC repression coefficient (0.0-1.0)")
    parser.add_argument("--pareto", action="store_true", help="Whether to output Pareto Front sequences")
    return parser.parse_args()

def codons2aa(codons):
    codons = codons.replace("T", "U")
    return "".join([codon2pro[codons[i:i+3]] for i in range(0, len(codons), 3)])

def transfer2codons(sequence, input_type):
    if input_type == "rna":
        seq = sequence.upper().replace("T", "U")
        if len(seq) % 3 != 0:
            raise ValueError("RNA length must be multiple of 3")
        return seq
    elif input_type == "protein":
        seq = sequence.upper()
        if not seq.startswith("M") or not seq.endswith("*"):
             raise ValueError("Protein must start with M and end with *")
        codon_seqs = amino2codon(0, [list(seq)])[0]
        return "".join(codon_seqs)
    return sequence

def optimize(mfe_model, cai_model, cai_table, eval_datas, weights, sample_size, sample_method, sample_temperature, max_threads, device, seed, user_ban=[], m6a_ban=[], gc_repress=None):
    device = torch.device(device)
    torch.manual_seed(seed)
    
    eval_cases_codon, eval_cases_pro, length = cod2case(seed, eval_datas)
    mfe_model.eval()
    start = time.time()

    cai_score = cai_model.calculate_score(eval_cases_pro, length).to(device)
    
    for i in range(len(m6a_ban) + 1):
        try:
            current_m6a = m6a_ban[:len(m6a_ban)-i] if i > 0 else m6a_ban
            current_total_ban = user_ban + current_m6a
            
            ban_codon_table, ban_pro_seqs = ban_table_gen(current_total_ban)

            rna = []
            if sample_method == "greedy":
                output, _ = mfe_model.optimize(eval_cases_codon, eval_cases_pro, length, cai_score, weights[0], "greedy", sample_temperature, ban_codon_table, ban_pro_seqs, gc_repress)
                rna.extend(output)
            elif sample_method == "sample":
                output, _ = mfe_model.optimize(eval_cases_codon*sample_size, eval_cases_pro*sample_size, length*sample_size, cai_score.repeat(sample_size,1,1), weights[0], "sample", sample_temperature, ban_codon_table, ban_pro_seqs, gc_repress)
                rna.extend(output)
            
            time_cost = time.time()-start
            mfe = calculate_mfe_mul(rna, max_threads)
            cai = calculate_cai(rna, cai_table)
            
            if i > 0:
                print(f"Optimization succeeded after removing {i} m6A motifs.")
            
            return rna, mfe, cai, time_cost

        except Exception as e:
            if "banned" in str(e).lower() and i < len(m6a_ban):
                # print(f"Retrying: reduced m6A motifs to {len(m6a_ban)-i-1}")
                continue 
            else:
                raise e
                
    return [], [], [], 0

if __name__ == "__main__":
    args = parse_args()

    with open(args.data_path, "r") as f:
        line = f.readline()
        name = line.replace(">", "").strip() if line.startswith(">") else "unknown_gene"
        raw_seq = f.read().strip()
    
    input_rna = transfer2codons(raw_seq, args.input_type)

    train_cfg = OmegaConf.load(path.join("config", "train.yaml"))
    mfe_model = Actor_Critic(train_cfg, device=args.device)
    mfe_model.load_state_dict(torch.load(MFE_MODEL_PATH[args.model], map_location=torch.device(args.device)), strict=False)
    
    cai_table = pd.read_csv(args.codon_usage_freq_table_path)
    cai_model = Cai_model(cai_table)
    max_threads = 32

    user_ban = []
    if args.ban_seqs and args.ban_seqs.lower() != 'none':
        user_ban = [s.upper().replace("T", "U") for s in args.ban_seqs.split(";")]
    
    m6a_ban = []
    if args.ban_m6a:
        if path.exists(args.m6a_motif_path):
            m6a_ban = pd.read_json(args.m6a_motif_path)["motif"].tolist()
            m6a_ban = [s.upper().replace("T", "U") for s in m6a_ban]
        else:
            print(f"Warning: m6A motif path {args.m6a_motif_path} not found.")

    rna, mfe, cai, time_cost = optimize(
        mfe_model, cai_model, cai_table, [input_rna], 
        [args.mfe_weight, 1-args.mfe_weight], args.sample_size, 
        args.sample_method, args.sample_temperature, max_threads, 
        args.device, args.seed, 
        user_ban=user_ban, m6a_ban=m6a_ban, 
        gc_repress=args.gc_repress
    )

    def get_gc(seqs): return [(s.count("G") + s.count("C")) / len(s) for s in seqs]
    
    motif_rates = calculate_motif_rate(rna, m6a_ban) if rna else []
    gc_contents = get_gc(rna) if rna else []

    input_rna_mfe = calculate_mfe_mul([input_rna], max_threads)
    input_rna_cai = calculate_cai([input_rna], cai_table)
    input_motif_rate = calculate_motif_rate([input_rna], m6a_ban)
    input_gc = get_gc([input_rna])[0]

    # 保存结果
    output_file = path.join(args.save_path, "output.csv")
    data_list = []
    
    data_list.append({
        "gene": name, "sample_method": "origin", "length": len(input_rna), 
        "alpha": -1, "gc_repress": -1, "mfe": input_rna_mfe[0], "cai": input_rna_cai[0], 
        "gc_content": input_gc, "motif_rate": input_motif_rate[0], "seq": input_rna
    })

    for idx in range(len(rna)):
        data_list.append({
            "gene": name, "sample_method": args.sample_method, "length": len(rna[idx]), 
            "alpha": args.mfe_weight, "gc_repress": args.gc_repress if args.gc_repress else 0, 
            "mfe": mfe[idx], "cai": cai[idx], "gc_content": gc_contents[idx],
            "motif_rate": motif_rates[idx], "seq": rna[idx]
        })

    df = pd.DataFrame(data_list)
    df.drop_duplicates(subset=["seq"], inplace=True)
    df.to_csv(output_file, index_label="id")

    if args.pareto and NonDominatedSorting is not None and len(df) > 1:
        print("Calculating Pareto Front...")
        nds = NonDominatedSorting()
        objectives = df[["mfe", "cai", "gc_content"]].copy()
        objectives["mfe"] = -objectives["mfe"] 
        objectives["cai"] = -objectives["cai"]
        
        front_indices = nds.do(objectives.values, return_rank=False)[0]
        pareto_df = df.iloc[front_indices]
        pareto_df.to_csv(path.join(args.save_path, "pareto.csv"), index=False)
        print(f"Pareto Front saved to {path.join(args.save_path, 'pareto.csv')}")

    print("Time cost per sequence: {:.2f}s".format(time_cost / len(rna) if len(rna) > 0 else 0))
    print("done")