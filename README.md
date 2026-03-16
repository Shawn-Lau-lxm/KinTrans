# KinTrans
KinTrans: A Transformer-based Reinforcement Learning Framework for Kinase Inhibitor Design with Target Adaptation and Patent Awareness

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
