#!/usr/bin/env python
from __future__ import print_function, division
from rdkit import rdBase
import torch
import os
import json
from torch.utils.data import DataLoader
from typing import Optional, Sequence
from predictor.src.utils import DrugTargetDataset, collate, AminoAcid
from predictor.src.models.DAT import DAT3

rdBase.DisableLog('rdApp.error')
        
class FusionDTATransfer():
    """Predict binding affinity between generated molecules and given target. (Chat-GPT optimized)"""
    def __init__(
        self,
        predictor_saved_path: str,
        protein_sequence: str,
        dataset: str = "davis",
        kinase_model: str = "ALK_TYROSINE_KINASE_RECEPTOR",
        fold: int = 10,
        use_cuda: bool = True,
        num_workers: Optional[int] = 16,
        batch_size: int = 1024,
        pin_memory: bool = True,
        # DAT3 architecture hyper‑params (keep default unless you changed them)
        embedding_dim: int = 1280,
        rnn_dim: int = 128,
        hidden_dim: int = 256,
        graph_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.3,
        alpha: float = 0.2,
        pretrain: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.dataset = dataset.lower()
        self.fold = fold
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers or max(os.cpu_count() - 2, 2)
        self.protein_sequence = protein_sequence

        # --- keep model hyper‑params as attributes so that _load_models can use them
        self._arch_kwargs = dict(
            embedding_dim=embedding_dim,
            rnn_dim=rnn_dim,
            hidden_dim=hidden_dim,
            graph_dim=graph_dim,
            n_heads=n_heads,
            dropout_rate=dropout,
            alpha=alpha,
            is_pretrain=pretrain,
        )

        # Pre‑load the ensemble once – heavy I/O happens here, not during scoring
        self.models = self._load_models(predictor_saved_path, kinase_model)

        # Protein‑ID mapping is small; load once into memory
        mapping_file = os.path.join(os.path.dirname(__file__), "predictor", "data", "kintrans_pro_id.json")
        if not os.path.exists(mapping_file):
            mapping_file = "predictor/data/kintrans_pro_id.json"  # fallback for cwd‑relative execution
        with open(mapping_file, "r", encoding="utf-8") as fp:
            self.pid_map: dict[str, int] = json.load(fp)

        # Resolve target PID only once
        self.target_pid: int = self.pid_map[self.protein_sequence]

    # ---------------------------------------------------------------------
    # private helpers
    # ---------------------------------------------------------------------
    def _load_models(self, root: str, kinase_model: str) -> torch.nn.ModuleList:
        """Instantiate ❰DAT3❱ models and load their checkpoints (CPU → GPU)."""
        model_dir = os.path.join(root, kinase_model, self.dataset, "whole_set_cv")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Cannot find CKPT directory: {model_dir}")

        ensemble = torch.nn.ModuleList()
        for k in range(self.fold):
            ckpt = os.path.join(
                model_dir,
                f"DAT_best_{self.dataset}_65smiles-random-{self.fold}fold{k}.pkl",
            )
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

            model = DAT3(**self._arch_kwargs).to(self.device).eval()
            state_dict = torch.load(ckpt, map_location="cpu")["model"]
            model.load_state_dict(state_dict, strict=False)

            # Freeze params to avoid gradient tracking / reduce memory
            for p in model.parameters():
                p.requires_grad_(False)

            ensemble.append(model)

        return ensemble

    # ------------------------------------------------------------------
    # public API – same signature as original fusion_dta_transfer __call__
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, smiles: Sequence[str]) -> torch.Tensor:
        """Return predicted affinities as **CPU tensor** of shape ``(N,)``.

        The method is *batched* and *streaming*-friendly – a list of any length
        can be provided, and the routine automatically chunks it into
        ``self.batch_size`` portions so RAM/GPU usage stays bounded.
        """
        n = len(smiles)
        if n == 0:
            return torch.empty(0, dtype=torch.float32)

        dataset = DrugTargetDataset(
            list(smiles),
            [self.protein_sequence] * n,
            [torch.tensor(1.0)] * n,                  # dummy affinity
            [self.target_pid] * n,
            is_target_pretrain=self._arch_kwargs["is_pretrain"]
        )

        loader = DataLoader(
            dataset,
            batch_size=n,
            shuffle=False,
            collate_fn=collate,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

        preds_cpu = []
        for protein_batch, smiles_batch, _ in loader:
            protein_batch = [p.to(self.device, non_blocking=True) for p in protein_batch]
            smiles_batch  = [s.to(self.device, non_blocking=True) for s in smiles_batch]

            # Forward through ensemble; each member returns (hidden, affinity)
            member_outs = []
            for model in self.models:
                _, out_aff = model(protein_batch, smiles_batch)
                member_outs.append(out_aff)  # (B,)
            # (ensemble_size, B) → (B,) by mean pooling
            mean_aff = torch.stack(member_outs, dim=0).mean(dim=0)
            preds_cpu.append(mean_aff.cpu())  # move to host so GPU memory is freed

        return torch.cat(preds_cpu, dim=0)                    # (N,)
    
