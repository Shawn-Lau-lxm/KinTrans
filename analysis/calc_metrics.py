#!/usr/bin/env python
from multiprocessing import Pool
from scipy.spatial.distance import cosine as cos_distance
from utils import mapper, valid_smiles
from utils import disable_rdkit_log, enable_rdkit_log
from utils import read_smiles_csv, read_score_csv, get_ledock_score_parallel
from metrics.utils import compute_fragments, calc_agg_tanimoto, calc_self_tanimoto, \
    compute_scaffolds, fingerprints, canonic_smiles, get_mol
import argparse
import numpy as np
import pandas as pd

# --- Added imports for QED / Lipinski ---
from rdkit.Chem import QED, Descriptors

def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(valid_smiles, gen)
    return gen.count(True) / len(gen)

def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]

def fraction_unique(gen, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        canonic.remove(None)
    return len(canonic) / len(gen)

def moses_novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)

def scaffold_novelty(gen, train, n_jobs=1):
    gen_scaffolds = set(compute_scaffolds(gen, n_jobs=n_jobs))
    train_scaffolds = set(compute_scaffolds(train, n_jobs=n_jobs))
    scaffold_novelty_score = len(gen_scaffolds - train_scaffolds) / len(gen)
    return scaffold_novelty_score

def scaffold_diversity(gen, n_jobs=1):
    scaffolds = compute_scaffolds(gen, n_jobs=n_jobs)
    scaffold_diversity_score = len(scaffolds) / len(gen)
    return scaffold_diversity_score

def novelty(gen, train, n_jobs=1, device='cpu', fp_type='morgan', gen_fps=None, train_fps=None, p=1):
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    if train_fps is None:
        train_fps = fingerprints(train, fp_type=fp_type, n_jobs=n_jobs)
    sim = calc_agg_tanimoto(gen_fps, train_fps, agg='max', device=device, p=p)
    return 1 - np.mean(sim), 1 - np.array(sim)

def internal_diversity_full(gen, n_jobs=1, device='cpu', fp_type='morgan', gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    Returns (diversity_mean, distribution_of_(1-sim))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    sim = calc_self_tanimoto(gen_fps, agg='mean', device=device, p=p)
    return 1 - np.mean(sim), 1 - np.array(sim)

def recovery(gen:list, ref:list):
    """
    Computes recovery rate of ref by gen
    """
    if len(ref) == 0:
        return np.nan
    if len(gen) == 0:
        return 0.0
    _covery = [i for i in ref if i in gen]
    return len(_covery)/len(ref)

# Predictor-based active rate (default threshold configurable)
def active_rate(score, thr=0.7):
    return sum(i > thr for i in score) / len(score) if len(score) else float('nan')

def success_rate(gen_valid, train, score_valid, n_jobs=1, device='cuda', fp_type='morgan', gen_fps=None, train_fps=None, p=1, sim_thr=0.7, aff_thr=0.7):
    """
    Success: similarity < sim_thr AND affinity > aff_thr (predictor output).
    Computed on VALID generated molecules to stay aligned with rank_eval_samples.py.
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen_valid, fp_type=fp_type, n_jobs=n_jobs)
    if train_fps is None:
        train_fps = fingerprints(train, fp_type=fp_type, n_jobs=n_jobs)
    sim = calc_agg_tanimoto(gen_fps, train_fps, agg='max', device=device, p=p)
    assert len(sim) == len(score_valid)
    return sum(i < sim_thr and j > aff_thr for i, j in zip(sim, score_valid)) / len(score_valid) if len(score_valid) else float('nan')

# --- New helpers from rank_eval_samples.py logic (adapted) ---
def lipinski_pass(mol):
    if mol is None: return False
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    mw  = Descriptors.MolWt(mol)
    logp= Descriptors.MolLogP(mol)
    return (hbd <= 5 and hba <= 10 and mw <= 500 and logp <= 5)

def compute_qed_and_lipinski(gen_valid, n_jobs=1):
    mols = mapper(n_jobs)(get_mol, gen_valid)
    # QED
    qed_vals = [QED.qed(m) if m is not None else 0.0 for m in mols]
    qed_mean = float(np.mean(qed_vals)) if len(qed_vals) else float('nan')
    # Lipinski
    lip_pass = sum(1 for m in mols if lipinski_pass(m))
    lip_rate = 100.0 * lip_pass / len(mols) if len(mols) else float('nan')
    return qed_mean, lip_rate

def affinity_basic_stats(score_valid):
    if not len(score_valid):
        return float('nan'), float('nan')
    arr = np.array(score_valid, dtype=float)
    aff_mean = float(np.mean(arr))
    k = max(1, int(0.1 * len(arr)))
    top10_mean = float(np.mean(np.sort(arr)[-k:]))
    return aff_mean, top10_mean

def get_valid_mask(gen, n_jobs=1):
    mask = mapper(n_jobs)(valid_smiles, gen)
    return [i for i, ok in enumerate(mask) if ok]

class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pgen, pref)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError

class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return recovery(pgen['frag'], pref['frag'])

class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return recovery(pgen['scaf'], pref['scaf'])

def get_all_metrics(gen, n_jobs=1,
                    device='cpu', batch_size=512, pool=None,
                    test=None, train=None, score=None,
                    aff_active_threshold=0.7):
    disable_rdkit_log()
    metrics = {}
    # Start the process at the beginning and avoid repeating the process
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1

    # Validity / Uniqueness / Diversity (scaffold-based)
    metrics['Validity'] = fraction_valid(gen, n_jobs=pool)
    gen_valid = remove_invalid(gen, canonize=True)
    metrics['Uniqueness'] = fraction_unique(gen, pool)
    metrics['Diversity'] = scaffold_diversity(gen_valid, n_jobs=pool)

    # Alignment between valid SMILES and scores
    valid_idx = get_valid_mask(gen, n_jobs=pool)
    score_valid = [score[i] for i in valid_idx] if (score is not None and len(valid_idx)) else []

    # Novelty metrics (tanimoto & scaffold)
    if train is not None:
        nov_value, nov_distribution = novelty(gen_valid, train, pool, device=device)
        metrics['Novelty_tanimoto'] = nov_value
        metrics['Novelty'] = scaffold_novelty(gen_valid, train, n_jobs=pool)

    # Recovery metrics
    if test is not None:
        kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
        metrics['Recovery/Frag'] = FragMetric(**kwargs)(gen=gen_valid, ref=test)
        metrics['Recovery/Scaf'] = ScafMetric(**kwargs)(gen=gen_valid, ref=test)

    # Predictor-based activity/success and affinity stats
    # (If your score comes from docking, these names still compute numerically but may be semantically different.)
    metrics['Active_rate'] = active_rate(score_valid, thr=aff_active_threshold) if score_valid else float('nan')
    if len(score_valid):
        aff_mean, top10_mean = affinity_basic_stats(score_valid)
        metrics['aff_mean'] = aff_mean
        metrics['aff_top10_mean'] = top10_mean
        metrics['aff>'+str(aff_active_threshold)+'_%'] = 100.0 * active_rate(score_valid, thr=aff_active_threshold)
    else:
        metrics['aff_mean'] = float('nan')
        metrics['aff_top10_mean'] = float('nan')
        metrics['aff>'+str(aff_active_threshold)+'_%'] = float('nan')

    # Success rate on valid molecules only (consistent with rank_eval_samples.py)
    if train is not None and len(score_valid):
        metrics['Success_rate'] = success_rate(gen_valid, train, score_valid, pool, device=device,
                                               sim_thr=0.7, aff_thr=aff_active_threshold)
    elif train is not None:
        metrics['Success_rate'] = float('nan')

    # QED mean & Lipinski %
    qed_mean, lip_rate = compute_qed_and_lipinski(gen_valid, n_jobs=pool)
    metrics['qed_mean'] = qed_mean
    metrics['lipinski_%'] = lip_rate

    # Internal diversity (full-set) & mean Tanimoto
    int_div_value, _ = internal_diversity_full(gen_valid, pool, device=device)
    metrics['int_div'] = float(int_div_value)
    metrics['mean_tanimoto'] = float(1.0 - int_div_value)

    # Scaffold counts
    scafs = compute_scaffolds(gen_valid, n_jobs=pool)
    metrics['scaffold_unique'] = int(len(set(scafs)))
    metrics['scaffold_total'] = int(len(scafs))

    enable_rdkit_log()
    if close_pool:
        pool.close()
        pool.join()
    return metrics

def main(config):
    gen = read_smiles_csv(config.gen_path)
    train = None
    test = None
    score = None

    if config.train_path is not None:
        # FIX: use train_path to determine sep
        if config.train_path.endswith('.csv'):
            train = read_smiles_csv(config.train_path, sep=',')
        elif config.train_path.endswith('.tsv'):
            train = read_smiles_csv(config.train_path, sep='\t')

    if config.test_path is not None:
        if config.test_path.endswith('.csv'):
            test = read_smiles_csv(config.test_path, sep=',')
        elif config.test_path.endswith('.tsv'):
            test = read_smiles_csv(config.test_path, sep='\t')

    if config.score_path is not None:
        score = read_score_csv(config.score_path)
    else:
        assert config.dock_file_dir is not None
        score = get_ledock_score_parallel(gen, n=config.n_jobs, pool='process',
                    dock_file_dir=config.dock_file_dir, work_dir=config.dock_file_dir+'_eval_',
                    save_work_dir=False)
        pd.DataFrame({"Smiles": gen, "Ledock": score}).to_csv(config.gen_path, index=None)

    metrics = get_all_metrics(gen=gen, n_jobs=config.n_jobs,
                                 device=config.device,
                                 test=test, train=train,
                                 score=score,
                                 aff_active_threshold=config.aff_active_threshold)

    if config.print_metrics:
        for key, value in metrics.items():
            try:
                print('{}, {:.4}'.format(key, value))
            except Exception:
                print('{}, {}'.format(key, value))
    else:
        return metrics

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path',
                        type=str, required=False,
                        help='Path to test molecules csv')
    parser.add_argument('--train_path',
                        type=str, required=False,
                        help='Path to train molecules csv')
    parser.add_argument('--gen_path',
                        type=str, required=True,
                        help='Path to generated molecules csv')
    parser.add_argument('--output',
                        type=str, required=True,
                        help='Path to save results csv')
    parser.add_argument('--print_metrics', action='store_true',
                        help="Print results of metrics or not? [Default: False]")
    parser.add_argument('--score_path',
                        type=str, required=False,
                        help='Path to read ledock score')
    parser.add_argument('--dock_file_dir',
                        type=str, required=False,
                        help='Path to structure file required by ledock')
    parser.add_argument('--n_jobs',
                        type=int, default=1,
                        help='Number of processes to run metrics')
    parser.add_argument('--device',
                        type=str, default='cpu',
                        help='GPU device id (`cpu` or `cuda:n`)')
    parser.add_argument('--aff_active_threshold',
                        type=float, default=0.7,
                        help='Threshold used for activity metrics (e.g., aff>[aff_active_threshold]_%% and Active_rate).')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_known_args()[0]
    metrics = main(config)

    # Save input settings
    metrics = metrics or {}
    metrics["train_path"] = config.train_path
    metrics["test_path"] = config.test_path
    metrics["gen_path"] = config.gen_path
    metrics["score_path"] = config.score_path

    table = pd.DataFrame([metrics]).T
    table.to_csv(config.output, header=False)
