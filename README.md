# KinTrans
KinTrans: A Transformer-based Reinforcement Learning Framework for Kinase Inhibitor Design with Target Adaptation and Patent Awareness

## Download Required Data
Download the necessary environment packages and data from the [Zenodo repository](https://doi.org/10.5281/zenodo.20066519). After downloading, place tar -xzvf in the KinTrans folder and extract it using the following command：
```bash
tar -xzvf tar -xzvf
```

After extraction, the file hierarchy of the Kintrans folder will look like this:
```
KinTrans/
│
├── analysis/
│   ├── metrics/
│   ├── calc_metrics.py
│   └── utils.py
│
├── data/
│   ├── finetune/
│   ├── kinase/
│   ├── payrus_H/
│   ├── payrus_H_M/
│   ├── payrus_H_M_L/
│   ├── results/
│   └── ... 
│
├── predictor/
│   ├── data/
│   ├── saved_models/
│   └── src/
│
└── ...
```

## Install Required Packages
Use `conda pack` to easily install required packages in your environment.
```bash
# Install conda-pack in your base environment.
conda activate base
pip install conda-pack

# Create kintrans directory under your conda envs directory.
mkdir  ~/software/miniconda3/envs/kintrans
tar -xzvf  kintrans-cu126.tar.gz -C ~/software/miniconda3/envs/kintrans
conda activate kintrans
```

## Extract Vocabulary of Input Datasets
There are three types of datasets in our study, including papyrus bioactive compounds dataset, 17 kinases' bioactive ligands datasets and patent compounds dataset. You should extract vocabulary from all datasets and remove the duplicates.
```bash
python data_structs.py
```

## Train Prior Models

```bash
python pretrain_trans.py
```

## Train Agent MOdels

```bash
python train_agent_trans-v1.py # train KinTrans
python train_agent_trans-v2.py # train KInTrans-Patent
```

## Evaluate Generated Molecules

```bash
python analysis/calc_metrics.py
```