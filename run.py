from omegaconf import OmegaConf
from os import path
import torch
import time
import pandas as pd
import argparse

from model.actor_critic import Actor_Critic
from model.cai_model import Cai_model
from utils.constant import codon2pro,ban_table_gen
from utils.random_gen import pro2case, cod2case

from utils.evaluate import calculate_mfe_mul, calculate_cai
from utils.dataset import load_data


MFE_MODEL_PATH = {"RNAJog":"./save/rnajog/model.pt",
                  "RNAJog_zero":"./save/rnajog_zero/model.pt"}

def parse_args():
    parser = argparse.ArgumentParser(description="RNAJog")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="RNAJog", choices=["RNAJog", "RNAJog_zero"])
    parser.add_argument("--input_type", type=str, default="rna", choices=["rna", "protein"])
    parser.add_argument("--data_path", type=str, default="upload/")
    parser.add_argument("--codon_usage_freq_table_path", type=str, default="./codon_usage_freq_table_human.csv")
    parser.add_argument("--mfe_weight", type=float, default=0.15)
    parser.add_argument("--sample_method", type=str, default="sample", choices=["greedy", "sample"])
    parser.add_argument("--sample_size", type=int, default=1)
    parser.add_argument("--sample_temperature", type=float, default=0.01)
    parser.add_argument("--save_path", type=str, default="result/")
    parser.add_argument("--ban_seqs", type=str, default="")
    return parser.parse_args()


def codons2aa(codons):
    codons = codons.replace("T", "U")
    return "".join([codon2pro[codons[i:i+3]] for i in range(0, len(codons), 3)])


def optimize(mfe_model, cai_model, cai_table, eval_datas, weights, sample_size, sample_method, sample_temperature, max_threads, device, seed, ban_seqs=[]):
    device = torch.device(device)
    torch.manual_seed(seed)
    
    eval_cases_codon, eval_cases_pro, length = cod2case(seed, eval_datas)

    mfe_model.eval()
    start = time.time()

    cai_score = cai_model.calculate_score(eval_cases_pro, length).to(device)
    ban_codon_table, ban_pro_seqs = ban_table_gen(ban_seqs)

    rna = []
    if sample_method == "greedy":
        output, _ = mfe_model.optimize(eval_cases_codon, eval_cases_pro, length, cai_score, weights[0], "greedy", sample_temperature,ban_codon_table, ban_pro_seqs)
        rna.extend(output)
    elif sample_method == "sample":
        output, _ = mfe_model.optimize(eval_cases_codon*sample_size, eval_cases_pro*sample_size, length*sample_size, cai_score.repeat(sample_size,1,1), weights[0], "sample", sample_temperature, ban_codon_table, ban_pro_seqs)
        rna.extend(output)
    else:
        raise ValueError("Invalid sample method: {}".format(sample_method))

    time_cost = time.time()-start
    mfe = calculate_mfe_mul(rna, max_threads)
    cai = calculate_cai(rna, cai_table)
    return rna, mfe, cai, time_cost



if __name__ == "__main__":
    args = parse_args()

    if args.input_type == "rna":
        with open(args.data_path, "r") as f:
            name = f.readline().replace(">", "").strip()
            input_rna = f.read().strip().replace("T", "U")
    elif args.input_type == "protein":
        with open(args.data_path, "r") as f:
            name = f.readline().replace(">", "").strip()
            input_protein = f.read().strip()
        input_rna, _, _ = pro2case(args.seed,[input_protein])
        input_rna = "".join(input_rna[0])
        
    else:
        raise ValueError("Invalid input type: {}".format(args.input_type))

    print(name)

    config = OmegaConf.load("config.yaml")
    config.device = args.device
    mfe_model = Actor_Critic(config)
    mfe_model.load_state_dict(torch.load(MFE_MODEL_PATH[args.model], map_location=torch.device(args.device)), strict=False)
    cai_table = pd.read_csv(args.codon_usage_freq_table_path)
    cai_model = Cai_model(cai_table)
    max_threads = 1

    if args.ban_seqs == 'none' or args.ban_seqs == '':
        ban_seqs = []
    else:
        ban_seqs = args.ban_seqs.split(";")
    
    rna, mfe, cai, time_cost = optimize(mfe_model, cai_model, cai_table, [input_rna], [args.mfe_weight, 1-args.mfe_weight], args.sample_size, args.sample_method, args.sample_temperature, max_threads, args.device, args.seed, ban_seqs)

    print("Time cost: {:.2f}s".format(time_cost))

    if args.input_type == "rna":
        input_rna_mfe = calculate_mfe_mul([input_rna], max_threads)
        input_rna_cai = calculate_cai([input_rna], cai_table)
    
    with open(path.join(args.save_path,"output.csv"), "w") as f:
        f.write("id,sample_method,length,mfe_cai_weight,mfe,cai,seq\n")
        if args.input_type == "rna":
            f.write("{},{},{},{},{},{},{}\n".format(0, "input", len(input_rna), -1, input_rna_mfe[0], input_rna_cai[0], input_rna))
        for idx, r in enumerate(rna):
            f.write("{},{},{},{},{},{},{}\n".format(idx+1, args.sample_method, len(r), args.mfe_weight, mfe[idx], cai[idx], r))
    
    print("done")
