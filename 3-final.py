from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import os
import json
import glob
import numpy as np

info_dict = {'Index':[],
             'Cyclic sequence':[],
             'pLDDT':[],
             'Molecular weight':[], 
             'Isoelectric point':[], 
             'Aromaticity':[], 
             'Instability index':[], 
             'Hydrophobicity':[],
             'Hydrophilicity':[],
            }

# Hopp-Woods hydrophilicity scale
hopp_woods = {
    'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0,
    'C': -1.0, 'Q': 0.2, 'E': 3.0, 'G': 0.0,
    'H': -0.5, 'I': -1.8, 'L': -1.8, 'K': 3.0,
    'M': -1.3, 'F': -2.5, 'P': 0.0, 'S': 0.3,
    'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
}

def calculate_hydrophilicity(sequence: str, scale: dict = hopp_woods):
    values = [scale.get(aa, 0.0) for aa in sequence]
    return sum(values) / len(values) if values else 0.0

def cyclic_sequence(fasta_path):
    file_in = open(fasta_path, 'r')
    seq_list = []
    for line in file_in.readlines():
        tmp = line.strip().split()
        if tmp[0][0] != '>':
            seq_list.append(tmp[0])
    return seq_list

def info(seq):
    analysis = ProteinAnalysis(seq)
    mw  = analysis.molecular_weight()
    ip  = analysis.isoelectric_point()
    aro = analysis.aromaticity()
    ins = analysis.instability_index()
    gra = analysis.gravy()
    hyd = calculate_hydrophilicity(seq)
    sec = analysis.secondary_structure_fraction()
    return mw, ip, aro, ins, gra, hyd, sec

def score(seq_list):
    score_dict = {}
    for i in range(len(seq_list)):
        file_list = glob.glob('./output/pep'+str(i+1)+'_scores_rank_00*_alphafold2_model_*_seed_000.json')
        score_list = []
        for file in file_list:
            file_in = open(file, 'r')
            data = json.load(file_in)
            tmp_list = []
            if "plddt" in data:
                score_list.append(np.mean(data["plddt"]))
        score_dict[seq_list[i]] = np.mean(score_list)
    return score_dict


fasta_path = './output/predict.fasta'
seq_list = cyclic_sequence(fasta_path)
score_dict = score(seq_list)

i = 1
for seq in seq_list:
    mw, ip, aro, ins, gra, hyd, sec = info(seq)
    info_dict['Index'].append(i)
    info_dict['Cyclic sequence'].append(seq)
    info_dict['pLDDT'].append(score_dict[seq])
    info_dict['Molecular weight'].append(mw)
    info_dict['Isoelectric point'].append(ip)
    info_dict['Aromaticity'].append(aro)
    info_dict['Instability index'].append(ins)
    info_dict['Hydrophobicity'].append(gra)
    info_dict['Hydrophilicity'].append(hyd)
    i += 1

df = pd.DataFrame(info_dict)
df.to_csv('output.csv', index=False)

