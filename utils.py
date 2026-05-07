import torch
import os
import numpy as np
import pandas as pd
import sqlite3
from rdkit import Chem

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)

def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))

def get_protein_sequence(protein_name):

    filename = os.path.join("data", "kinase", protein_name, "keepmean-norepeat_"+protein_name+".tsv") # filepath of given protein's dataset
    data = pd.read_csv(filename, sep="\t", index_col=0)
    a = data["protein_id"].value_counts()
    b = a.idxmax()
    pid_dict = data[["target_sequence","protein_id"]].set_index("protein_id").to_dict(orient="dict")["target_sequence"] # generate a dict with protein_id as index.
    protein_sequence = pid_dict[b]
    return protein_sequence

def load_patent_keys(db_path="drug_patent.db"):
    """
    读取整库 InChIKey；返回两个 set：
        stereo_set      —— 完整 InChIKey
        nostereo_set    —— 去立体 InChIKey
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT inchikey, inchikey_wostereo FROM schembl_inchikey")
    stereo, nostereo = set(), set()
    for ik, ik_ns in cur.fetchall():
        if ik:      stereo.add(ik)
        if ik_ns:   nostereo.add(ik_ns)
    conn.close()
    return stereo, nostereo