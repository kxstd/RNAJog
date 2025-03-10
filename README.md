# RNAJog

## Introduction
RNAJog is a tool designed for optimizing the coding sequence (CDS) of mRNA to achieve high protein expression levels. RNAJog can generate codon sequences with high codon adaptation index (CAI) and low minimum free energy (MFE), ensuring enhanced translational efficiency and mRNA stability. This tool enables users to optimize mRNA sequences for their target proteins or existing mRNA sequences. You can access and use the tool directly on our online platform. [RNAJog web-application](http://www.csbio.sjtu.edu.cn/bioinf2/RNAJog/)

<!-- [Check the RNAJog paper](). -->

## Prerequisites
If you haven't installed Conda yet, you can install either [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation) or [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install).

## Installation
``` r
git clone https://github.com/kxstd/RNAJog.git
cd RNAJog
conda env create -f environment.yml
conda activate rnajog
```
## Download the model parameters
Download the model parameters from http://www.csbio.sjtu.edu.cn/bioinf2/RNAJog/data/save.zip, and extract it to the RNAJog project directory.

## Usage
### Command-Line Arguments
The script provides various options to customize the optimization process. Below are the available arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | `cuda` | Device to use (`cpu` or `cuda`). |
| `--seed` | int | `0` | Random seed. |
| `--model` | str | `RNAJog` | The optimization model (`RNAJog`, `RNAJog_zero`). |
| `--input_type` | str | `rna` | The type of input (`rna` or `protein`). |
| `--data_path` | str | `data/test/rna.txt` | Path to the input data file. |
| `--codon_usage_freq_table_path` | str | `./codon_usage_freq_table_human.csv` | Path to the codon usage frequency table. |
| `--mfe_weight` | float | `0.15` | Weight for MFE in optimization (i.e. the MFE-CAI weight). |
| `--sample_method` | str | `sample` | Sampling method (`greedy` or `sample`). |
| `--sample_size` | int | `1` | Number of samples generated. |
| `--sample_temperature` | float | `0.01` | Temperature parameter for sampling. |
| `--save_path` | str | `result/` | Directory to save the output. |
| `--ban_seqs` | str | `""` | Subsequences to be banned from the output. |

### Running the Program
To run the RNA optimization script, use the following command:

```bash
python run.py --input_type rna --data_path data/test/rna.txt --device cuda --model RNAJog
```

If you want to optimize a protein sequence:

```bash
python run.py --input_type protein --data_path data/test/protein.txt --device cuda --model RNAJog
```

### Output
The program outputs optimized RNA sequences to a CSV file located in the specified `save_path`. The file contains:

- `id`: Sample identifier
- `sample_method`: Sampling method used
- `length`: Sequence length
- `mfe_cai_weight`: MFE-CAI weight in optimization
- `mfe`: Minimum Free Energy of the sequence
- `cai`: Codon Adaptation Index of the sequence
- `seq`: Optimized RNA sequence

The output file is saved as `output.csv` in the specified directory.

## Example
An example output in `output.csv`:

```csv
id,sample_method,length,mfe_cai_weight,mfe,cai,seq
0,input,300,-1,-12.5,0.89,UAUGCGUAGC...
1,sample,300,0.15,-14.2,0.91,UAUGCGUAGC...
```

