# RNAJog：Fast Multi-objective RNA Optimization with Autoregressive Reinforcement Learning
RNAJog is a tool designed for optimizing the coding sequence (CDS) of mRNA to achieve high protein expression levels. RNAJog can generate codon sequences with high codon adaptation index (CAI) and low minimum free energy (MFE), ensuring enhanced translational efficiency and mRNA stability. This tool enables users to optimize mRNA sequences for their target proteins or existing mRNA sequences. 

Access RNAJog online: [RNAJog Web Application](http://www.csbio.sjtu.edu.cn/bioinf2/RNAJog/)

<!-- [Check the RNAJog paper](). -->

## Prerequisites
Before installing RNAJog, ensure that you have Conda installed. You can download either:
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)
- [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install)

## Installation
To install RNAJog, follow these steps:
```bash
git clone https://github.com/kxstd/RNAJog.git
cd RNAJog
conda env create -f environment.yml
conda activate rnajog
```
We recommend you to run RNAJog on Linux.

## Download Model Parameters
Download the required model parameters from [this link](http://www.csbio.sjtu.edu.cn/bioinf2/RNAJog/data/save.zip) and extract them into the RNAJog project directory.

## Usage
### Command-Line Arguments
RNAJog provides multiple options to customize the optimization process. Below are the available arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | `cuda` | Device to use (`cpu` or `cuda`). |
| `--seed` | int | `0` | Random seed for reproducibility. |
| `--model` | str | `RNAJog` | Optimization model (`RNAJog` or `RNAJog_zero`). |
| `--input_type` | str | `rna` | Input type (`rna` or `protein`). |
| `--data_path` | str | `data/test/rna.txt` | Path to input data file. |
| `--codon_usage_freq_table_path` | str | `./codon_usage_freq_table_human_20250902.csv` | Path to the codon usage frequency table. |
| `--mfe_weight` | float | `0.15` | Weight of MFE in optimization (MFE-CAI balance). |
| `--sample_method` | str | `sample` | Sampling method (`greedy` or `sample`). |
| `--sample_size` | int | `1` | Number of generated samples. |
| `--sample_temperature` | float | `0.01` | Temperature parameter for sampling. |
| `--save_path` | str | `result/` | Directory to save output. |
| `--ban_seqs` | str | `""` | Forbidden subsequences in output. |
| `--ban_m6a`  | flag |	`False` | Whether to enable m6A motif elimination. |
| `--m6a_motif_path` | str | `motif.json` |	Path to m6A motif JSON file. |
| `--gc_repress` | float | `None` |	GC repression coefficient (0.0–1.0). |
| `--pareto` | flag | `False` |	Whether to output Pareto Front sequences. |


### Running RNA Optimization
To optimize an RNA sequence, use:
```bash
python run.py --input_type rna --data_path data/test/rna.txt --mfe_weight 0.15 --device cuda --model RNAJog --ban_seqs "CUCGAG;GCUCUUC"
```
For protein sequence optimization:
```bash
python run.py --input_type protein --data_path data/test/protein.txt --mfe_weight 0.15 --device cuda --model RNAJog
```

## Output
The optimized RNA sequences are saved in a CSV file located in the specified `result`. The output file contains the following columns:

- `id`: Sample identifier
- `gene`: Gene name
- `sample_method`: Sampling method used
- `length`: Sequence length
- `alpha`: MFE-CAI weight used in optimization
- `gc_repress`: GC repression coefficient
- `mfe`: Minimum Free Energy of the sequence
- `cai`: Codon Adaptation Index
- `gc_content`: GC content
- `motif_rate`: Proportion of m6A motifs occurring in the sequence
- `seq`: Optimized RNA sequence

## Citation
If you find the tool useful in your research, please cite our paper:
```
1. Huang, J. et al. Fast Multi-objective RNA Optimization with Autoregressive Reinforcement Learning. 2025.08.26.672486 Preprint at https://doi.org/10.1101/2025.08.26.672486 (2025).
```