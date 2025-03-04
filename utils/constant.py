import torch
import torch.nn as nn

protein_dict={"Phe":'F',
              "Leu":'L',
              "Ser":'S',
              "Tyr":'Y',
              "STOP":'*',
              "Cys":'C', 
              "Trp":'W',
              "Pro": 'P',
              "His": 'H',
              "Gln": 'Q',
              "Arg": 'R',
              "Ile": 'I',
              "Met": 'M',
              "Thr": 'T',
              "Asn": 'N',
              "Lys": 'K',
              "Val":'V',
              "Asp": 'D',
              "Glu":'E',
              "Gly": 'G',
              "Ala": 'A'} # 21 in total
# inverse_protein_dict = dict((v,k) for k,v in protein_dict.items())
# 一个氨基酸对应Codon的字典，key为氨基酸，value为Codon的list
decode_dict={"Phe":["UUU","UUC"],
                "Leu":["UUA","UUG","CUU","CUC","CUA","CUG"],
                "Ser":["UCU","UCC","UCA","UCG","AGU","AGC"],
                "Tyr":["UAU","UAC"],
                "STOP":["UAA","UAG","UGA"],
                "Cys":["UGU","UGC"],
                "Trp":["UGG"],
                "Pro":["CCU","CCC","CCA","CCG"],
                "His":["CAU","CAC"],
                "Gln":["CAA","CAG"],
                "Arg":["CGU","CGC","CGA","CGG","AGA","AGG"],
                "Ile":["AUU","AUC","AUA"],
                "Met":["AUG"],
                "Thr":["ACU","ACC","ACA","ACG"],
                "Asn":["AAU","AAC"],
                "Lys":["AAA","AAG"],
                "Val":["GUU","GUC","GUA","GUG"],
                "Asp":["GAU","GAC"],
                "Glu":["GAA","GAG"],
                "Gly":["GGU","GGC","GGA","GGG"],
                "Ala":["GCU","GCC","GCA","GCG"]} # 21
trans_dict = dict((protein_dict[k],v) for k,v in decode_dict.items())
codonlist = [c for c_l in decode_dict.values() for c in c_l]
id2pro = list(protein_dict.values())
pro2id = dict((v,i) for i,v in enumerate(id2pro))
codon2pro = dict()
for a, c in trans_dict.items():
    for cc in c:
        codon2pro[cc] = a
# decode_list = list(decode_dict.values())
# protein2rna = dict((v,decode_dict[k]) for k,v in protein_dict.items())
MASK = torch.zeros(21, 6) # 每个氨基酸有多少个Codon,就将对应行的前几个元素置为1
for i in range(21):
    MASK[i, :len(trans_dict[id2pro[i]])] = 1
MASK = (MASK==1)

def codon2idx(codons):
    """
    codons: [[codon1, codon2, codon3], [codon1, codon2, codon3], ...]
    """
    idx_list = []
    for codon in codons:
        idxs = []
        for c in codon:
            if c =="[PAD]": break
            pro = codon2pro[c]
            idx = trans_dict[pro].index(c)
            idxs.append(idx)
        idx_list.append(idxs)
    return idx_list

def get_mask(inputs):
    mask_list = []
    for i in range(len(inputs)):
        mask = torch.stack([MASK[pro2id[inputs[i][step]]] if inputs[i][step] != "[PAD]" else torch.tensor([True,True,True,True,True,True]) for step in range(len(inputs[i]))])
        # mask = torch.stack([MASK[pro2id[inputs[i][step]]] if inputs[i][step] != "[PAD]" else torch.tensor([False,False,False,False,False,False]) for step in range(len(inputs[i]))])
        mask_list.append(mask)
    mask = torch.stack(mask_list)
    return mask

def get_mask_f():
    return get_mask

def trans_idx2codon(idxs, inputs):
    codons_list = []
    for i in range(len(idxs)):
        idx = idxs[i]
        input = inputs[i]
        # codons = [trans_dict[input[k]][idx[k]] for k in range(len(idx)) if input[k] != "[PAD]" else "&&&"]
        # codons = []
        # for k in range(len(idx)):
        #     print(input[k])
        #     if input[k] == "[":
        #         codons.append("&&&")
        #     else:
        #         codons.append(trans_dict[input[k]][idx[k]])
        codons = ["&&&" if input[k] == "[" else trans_dict[input[k]][idx[k]] for k in range(len(idx))]
        codons_list.append("".join(codons))
    return codons_list


def ban_table_add(ban_codon_table, ban_pro_seqs, pro_tail, prefix, codon):
    """
    ban_codon_table: {(prefix, pro_tail): [False, False, False, False, False, False]}
    ban_pro_seqs: [seq1, seq2, ...]
    pro_tail: "ABC" the protein behind the current codon
    prefix: "abcdef" the prefix of the current codon
    codon: "abc" the current codon
    """
    # print("prefix: {}, pro_tail: {}, codon: {}".format(prefix, pro_tail, codon))
    if pro_tail in ban_pro_seqs:
        return
    pro = codon2pro[codon]
    if (prefix, pro+pro_tail) not in ban_codon_table.keys():
        ban_codon_table[(prefix, pro+pro_tail)] = ~MASK[pro2id[pro]]
    ban_codon_table[(prefix, pro+pro_tail)][codon2idx([[codon]])[0][0]] = True
    
    if torch.all(ban_codon_table[(prefix, pro+pro_tail)]):
        # print("delete prefix: {}, pro_tail: {}".format(prefix, pro+pro_tail))
        # print(ban_codon_table[(prefix, pro+pro_tail)])
        del ban_codon_table[(prefix, pro+pro_tail)]
        if prefix == "":
            ban_pro_seqs.append(pro+pro_tail)
        elif len(prefix)<3:
            for codon in codonlist:
                if codon.endswith(prefix):
                    ban_table_add(ban_codon_table, ban_pro_seqs, pro+pro_tail, "", codon)
            
        else:
            ban_table_add(ban_codon_table, ban_pro_seqs, pro + pro_tail, prefix[:-3], prefix[-3:])
    return

def ban_table_gen(ban_seqs):
    ban_codon_table = {} # {(prefix, pro_tail): [False, False, False, False, False, False]}
    ban_pro_seqs = []

    if ban_seqs.__len__() == 0:
        return None, None
    for k in range(1,4):
        for ban_seq in ban_seqs:
            prefix = ban_seq[:-k]
            tail = ban_seq[-k:]
            for codon in codonlist:
                if codon.startswith(tail):
                    ban_table_add(ban_codon_table, ban_pro_seqs, "", prefix, codon)
                    # pro = codon2pro[codon]
                    # if (pre_seq, "pro") not in ban_codon_table.keys():
                    #     ban_codon_table[(pre_seq, pro)] = torch.tensor([False,False,False,False,False,False])
                    # ban_codon_table[(pre_seq, pro)][codon2idx([[codon]])[0][0]] = True
    
    return ban_codon_table, ban_pro_seqs
