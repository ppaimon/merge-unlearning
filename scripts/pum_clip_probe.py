#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PUM clipping probe — v3.8 (clean)
- HF repo id or local path only (YAML support removed)
- bfloat16-safe loading via safetensors.torch -> torch.float32
- Per-layer quantile tables with RMS row (i=51) + per-layer alpha table
- Optional sanity/debug CSVs for task vectors
- Legacy aggregated outputs kept for continuity
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple, Any

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as st_load_torch
import torch

# ------------------------------
# HF snapshot & index
# ------------------------------

@dataclass
class HFModelIndex:
    root: str
    weight_map: Dict[str, str]  # param name -> abs file path

def _find_index_json(root: str) -> Tuple[str | None, str | None]:
    st_index = os.path.join(root, "model.safetensors.index.json")
    pt_index = os.path.join(root, "pytorch_model.bin.index.json")
    return (st_index if os.path.exists(st_index) else None,
            pt_index if os.path.exists(pt_index) else None)

def _load_safetensors_index(root: str) -> Dict[str, str]:
    st_index_path, _ = _find_index_json(root)
    if st_index_path is None:
        single = os.path.join(root, "model.safetensors")
        if os.path.exists(single):
            tensor_names = list(st_load_torch(single).keys())
            return {k: single for k in tensor_names}
        files = [f for f in os.listdir(root) if f.endswith(".safetensors")]
        if len(files) == 1:
            p = os.path.join(root, files[0])
            tensor_names = list(st_load_torch(p).keys())
            return {k: p for k in tensor_names}
        elif len(files) > 1:
            m: Dict[str, str] = {}
            for fname in files:
                p = os.path.join(root, fname)
                for k in st_load_torch(p).keys():
                    m[k] = p
            return m
        raise FileNotFoundError("No safetensors index or files found under: " + root)
    with open(st_index_path, "r") as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    return {k: os.path.join(root, v) for k, v in wm.items()}

def snapshot_or_local(path_or_repo: str, revision: str | None = None, token: str | None = None) -> str:
    if os.path.isdir(path_or_repo):
        return os.path.abspath(path_or_repo)
    return snapshot_download(repo_id=path_or_repo, revision=revision, token=token, local_files_only=False)

def build_index(path_or_repo: str, revision: str | None = None, token: str | None = None) -> HFModelIndex:
    root = snapshot_or_local(path_or_repo, revision=revision, token=token)
    wm = _load_safetensors_index(root)
    return HFModelIndex(root=root, weight_map=wm)

def load_param_numpy(index: HFModelIndex, param_name: str) -> np.ndarray:
    fpath = index.weight_map[param_name]
    td = st_load_torch(fpath)                 # handles bf16 tensors
    t = td[param_name]
    return t.to(dtype=torch.float32, device="cpu").numpy()

# ------------------------------
# Model structure helpers
# ------------------------------

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
COMP_SPECS = [
    ("W_q", r"\.self_attn\.q_proj\.weight$"),
    ("W_k", r"\.self_attn\.k_proj\.weight$"),
    ("W_v", r"\.self_attn\.v_proj\.weight$"),
    ("W_o", r"\.self_attn\.o_proj\.weight$"),
    ("W1_gate", r"\.mlp\.gate_proj\.weight$"),
    ("W1_up", r"\.mlp\.up_proj\.weight$"),
    ("W_2", r"\.mlp\.down_proj\.weight$"),
    ("input_norm", r"\.input_layernorm\.weight$"),
    ("post_attn_norm", r"\.post_attention_layernorm\.weight$"),
]
COMP_ORDER = [c for c, _ in COMP_SPECS]

def detect_num_layers(weight_map: Mapping[str, str]) -> int:
    max_idx = -1
    for name in weight_map.keys():
        m = LAYER_RE.search(name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0

def build_component_layer_map(weight_map: Mapping[str, str]) -> Dict[str, Dict[int, str]]:
    comp_map: Dict[str, Dict[int, str]] = {c: {} for c, _ in COMP_SPECS}
    compiled = [(c, re.compile(pat)) for c, pat in COMP_SPECS]
    for pn in weight_map.keys():
        m = LAYER_RE.search(pn)
        if not m:
            continue
        idx = int(m.group(1))
        for cname, creg in compiled:
            if creg.search(pn):
                comp_map[cname][idx] = pn
    return comp_map

# ------------------------------
# Per-layer tables (quantiles + RMS row) and alpha
# ------------------------------

def _quantiles_with_optional_sampling(abs_vals: np.ndarray, q_grid: List[float], sample_frac: float, rng: np.random.Generator) -> np.ndarray:
    if sample_frac < 1.0 and abs_vals.size > 0:
        k = max(1, int(math.ceil(abs_vals.size * sample_frac)))
        idx = rng.integers(0, abs_vals.size, size=k, endpoint=False)
        abs_vals = abs_vals[idx]
    return np.quantile(abs_vals, q_grid, method="linear")

def per_layer_quantiles_abs(base_index: HFModelIndex, comp_layers: Dict[str, Dict[int, str]], L: int, q_grid: List[float], sample_frac: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Returns {component -> DataFrame(52 x L)} with rows i=0..50 (Q_{0.02*i} of |θ_base|) and i=51 (RMS of θ_base).
    """
    out: Dict[str, pd.DataFrame] = {}
    row_index = list(range(len(q_grid))) + [51]
    rng = np.random.default_rng(12345)
    for cname in COMP_ORDER:
        df = pd.DataFrame(index=row_index, columns=[f"layer_{j+1}" for j in range(L)], dtype=np.float64)
        layer_map = comp_layers.get(cname, {})
        for j in range(L):
            col = f"layer_{j+1}"
            pn = layer_map.get(j)
            if pn is None:
                df.loc[:, col] = np.nan
                continue
            arr = load_param_numpy(base_index, pn).ravel().astype(np.float64, copy=False)
            abs_arr = np.abs(arr)
            df.loc[range(len(q_grid)), col] = _quantiles_with_optional_sampling(abs_arr, q_grid, sample_frac, rng)
            n = max(1, arr.size)
            df.at[51, col] = float(np.linalg.norm(arr) / math.sqrt(n))  # RMS
        out[cname] = df
    return out

def per_layer_quantiles_abs_delta(base_index: HFModelIndex, r1_index: HFModelIndex, comp_layers: Dict[str, Dict[int, str]], L: int, q_grid: List[float], sample_frac: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Returns {component -> DataFrame(52 x L)} with rows i=0..50 (Q_{0.02*i} of |θ_{r-1}-θ_base|) and i=51 (RMS of delta).
    """
    out: Dict[str, pd.DataFrame] = {}
    row_index = list(range(len(q_grid))) + [51]
    rng = np.random.default_rng(12345)
    for cname in COMP_ORDER:
        df = pd.DataFrame(index=row_index, columns=[f"layer_{j+1}" for j in range(L)], dtype=np.float64)
        layer_map = comp_layers.get(cname, {})
        for j in range(L):
            col = f"layer_{j+1}"
            pn = layer_map.get(j)
            if pn is None or pn not in base_index.weight_map or pn not in r1_index.weight_map:
                df.loc[:, col] = np.nan
                continue
            b = load_param_numpy(base_index, pn).ravel().astype(np.float64, copy=False)
            r = load_param_numpy(r1_index, pn).ravel().astype(np.float64, copy=False)
            d = (r - b)
            ad = np.abs(d)
            df.loc[range(len(q_grid)), col] = _quantiles_with_optional_sampling(ad, q_grid, sample_frac, rng)
            n = max(1, d.size)
            df.at[51, col] = float(np.linalg.norm(d) / math.sqrt(n))  # RMS(delta)
        out[cname] = df
    return out

def per_layer_alpha_table(base_index: HFModelIndex, r1_index: HFModelIndex, comp_layers: Dict[str, Dict[int, str]], L: int, alpha_q: float, gamma: float, rho_default: float, rho_norm: float) -> pd.DataFrame:
    """
    α = min(1, B / ||δ||_2) per (layer, component), with s = Q_{alpha_q}(|θ_base|), B = γ * s * sqrt(ρ * n).
    """
    df = pd.DataFrame(index=[f"layer_{j+1}" for j in range(L)], columns=COMP_ORDER, dtype=np.float64)
    for cname in COMP_ORDER:
        layer_map = comp_layers.get(cname, {})
        for j in range(L):
            row = f"layer_{j+1}"
            pn = layer_map.get(j)
            if pn is None or pn not in base_index.weight_map or pn not in r1_index.weight_map:
                df.at[row, cname] = np.nan
                continue
            b = load_param_numpy(base_index, pn).ravel().astype(np.float64, copy=False)
            r = load_param_numpy(r1_index, pn).ravel().astype(np.float64, copy=False)
            d = (r - b)
            n = d.size if d.size > 0 else 1
            delta_norm = float(np.linalg.norm(d))
            s = float(np.quantile(np.abs(b), alpha_q, method="linear"))
            rho = rho_norm if ("norm" in cname) else rho_default
            B = float(gamma * s * math.sqrt(rho * n))
            df.at[row, cname] = 1.0 if delta_norm == 0.0 else min(1.0, B / delta_norm)
    return df

# ------------------------------
# Sanity/debug helpers
# ------------------------------

def safetensor_total_size_bytes(root: str) -> int:
    total = 0
    for fn in os.listdir(root):
        if fn.endswith(".safetensors"):
            p = os.path.join(root, fn)
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
    return total

def per_layer_delta_debug_tables(base_index: HFModelIndex, r1_index: HFModelIndex, comp_layers: Dict[str, Dict[int, str]], L: int, nz_eps: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = [f"layer_{j+1}" for j in range(L)]
    cols = COMP_ORDER
    delta_l2 = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)
    delta_nz = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)
    delta_max = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)
    for cname in COMP_ORDER:
        layer_map = comp_layers.get(cname, {})
        for j in range(L):
            row = f"layer_{j+1}"
            pn = layer_map.get(j)
            if pn is None or pn not in base_index.weight_map or pn not in r1_index.weight_map:
                delta_l2.at[row, cname] = np.nan
                delta_nz.at[row, cname] = np.nan
                delta_max.at[row, cname] = np.nan
                continue
            b = load_param_numpy(base_index, pn).ravel().astype(np.float64, copy=False)
            r = load_param_numpy(r1_index, pn).ravel().astype(np.float64, copy=False)
            d = (r - b)
            n = d.size if d.size > 0 else 1
            delta_l2.at[row, cname] = float(np.linalg.norm(d))
            delta_nz.at[row, cname] = 100.0 * (float(np.count_nonzero(np.abs(d) > nz_eps)) / float(n))
            delta_max.at[row, cname] = float(np.max(np.abs(d))) if n > 0 else 0.0
    return delta_l2, delta_nz, delta_max

# ------------------------------
# Aggregated outputs (legacy)
# ------------------------------

DEFAULT_GROUP_RULES: List[Tuple[str, str]] = [
    ("W_Q", ".self_attn.q_proj.weight"),
    ("W_K", ".self_attn.k_proj.weight"),
    ("W_V", ".self_attn.v_proj.weight"),
    ("W_O", ".self_attn.o_proj.weight"),
    ("W1_gate", ".mlp.gate_proj.weight"),
    ("W1_up", ".mlp.up_proj.weight"),
    ("W_2", ".mlp.down_proj.weight"),
    ("embed", "model.embed_tokens.weight"),
    ("lm_head", "lm_head.weight"),
    ("input_norm", ".input_layernorm.weight"),
    ("post_attn_norm", ".post_attention_layernorm.weight"),
    ("final_norm", "model.norm.weight"),
]

def build_groups(weight_map: Mapping[str, str], grouping: str = "per-type") -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    if grouping == "per-param":
        for k in weight_map.keys():
            groups[k] = [k]
        return groups
    for gname, needle in DEFAULT_GROUP_RULES:
        groups[gname] = [k for k in weight_map.keys() if needle in k or k == needle]
    return {g: names for g, names in groups.items() if len(names) > 0}

def _intersect_group_names(groups: Dict[str, List[str]], other_index: HFModelIndex) -> Dict[str, List[str]]:
    other_names = set(other_index.weight_map.keys())
    return {g: [pn for pn in names if pn in other_names] for g, names in groups.items()}

def _iter_params_in_group(group_names: List[str], index: HFModelIndex) -> Iterable[Tuple[str, np.ndarray]]:
    # Open each shard once, yield tensors listed in group_names
    by_file: Dict[str, List[str]] = {}
    for pn in group_names:
        fpath = index.weight_map.get(pn)
        if fpath is not None:
            by_file.setdefault(fpath, []).append(pn)
    for fpath, keys in by_file.items():
        tensors = st_load_torch(fpath)
        for k in keys:
            arr = tensors[k].to(dtype=torch.float32, device="cpu").numpy()
            yield k, np.asarray(arr, dtype=np.float32)

def compute_quantiles_abs_values(index: HFModelIndex, groups: Dict[str, List[str]], qlist: List[float], sample_frac: float = 0.25) -> Tuple[pd.DataFrame, Dict[str, int]]:
    cols = list(groups.keys())
    out = pd.DataFrame(index=[f"{q:.2f}" for q in qlist], columns=cols, dtype=np.float64)
    n_entries: Dict[str, int] = {}
    rng = np.random.default_rng(12345)
    for g, names in groups.items():
        samples: List[np.ndarray] = []
        total_elems = 0
        for _, arr in _iter_params_in_group(names, index):
            flat = np.abs(arr.ravel())
            total_elems += flat.size
            if sample_frac >= 1.0 or flat.size == 0:
                samples.append(flat)
            else:
                k = max(1, int(math.ceil(flat.size * sample_frac)))
                idx = rng.integers(0, flat.size, size=k, endpoint=False)
                samples.append(flat[idx])
        n_entries[g] = int(total_elems)
        merged = np.concatenate(samples, dtype=np.float32) if len(samples) else np.array([], dtype=np.float32)
        if merged.size == 0:
            out.loc[:, g] = np.nan
        else:
            for q in qlist:
                out.at[f"{q:.2f}", g] = float(np.quantile(merged, q, method="linear"))
    return out, n_entries

def compute_quantiles_abs_delta(base_index: HFModelIndex, r1_index: HFModelIndex, groups: Dict[str, List[str]], qlist: List[float], sample_frac: float = 0.25) -> pd.DataFrame:
    cols = list(groups.keys())
    out = pd.DataFrame(index=[f"{q:.2f}" for q in qlist], columns=cols, dtype=np.float64)
    rng = np.random.default_rng(12345)
    for g, names in groups.items():
        samples: List[np.ndarray] = []
        # Preload r1 shards used by this group
        needed_files: Dict[str, List[str]] = {}
        for pn in names:
            fpr1 = r1_index.weight_map.get(pn)
            if fpr1 is not None:
                needed_files.setdefault(fpr1, []).append(pn)
        r1_loaded: Dict[str, Dict[str, torch.Tensor]] = {fp: st_load_torch(fp) for fp in needed_files.keys()}
        for pn, arr_base in _iter_params_in_group(names, base_index):
            fpr1 = r1_index.weight_map.get(pn)
            if fpr1 is None:
                continue
            arr_r1 = r1_loaded[fpr1][pn].to(dtype=torch.float32, device="cpu").numpy()
            flat = np.abs((arr_r1 - arr_base).ravel())
            if sample_frac >= 1.0 or flat.size == 0:
                samples.append(flat)
            else:
                k = max(1, int(math.ceil(flat.size * sample_frac)))
                idx = rng.integers(0, flat.size, size=k, endpoint=False)
                samples.append(flat[idx])
        merged = np.concatenate(samples, dtype=np.float32) if len(samples) else np.array([], dtype=np.float32)
        if merged.size == 0:
            out.loc[:, g] = np.nan
        else:
            for q in qlist:
                out.at[f"{q:.2f}", g] = float(np.quantile(merged, q, method="linear"))
    return out

def compute_clipping_and_entrywise(base_quant: pd.DataFrame, base_counts: Dict[str, int], base_index: HFModelIndex, r1_index: HFModelIndex, groups: Dict[str, List[str]], qlist: List[float], gamma: float, rho_default: float, rho_norm: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = list(groups.keys())
    idx = [f"{q:.2f}" for q in qlist]
    l2_alpha = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)
    l2_pct = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)
    entry_above_tau_pct = pd.DataFrame(index=idx, columns=cols, dtype=np.float64)
    for g, names in groups.items():
        n_l = base_counts.get(g, 0)
        if n_l == 0:
            l2_alpha.loc[:, g] = np.nan
            l2_pct.loc[:, g] = np.nan
            entry_above_tau_pct.loc[:, g] = np.nan
            continue
        rho = rho_norm if "norm" in g else rho_default
        s_vec = np.array([base_quant.at[f"{q:.2f}", g] for q in qlist], dtype=np.float64)
        tau_vec = gamma * s_vec
        B_vec = gamma * s_vec * math.sqrt(rho * n_l)
        sumsq = 0.0
        counts = np.zeros_like(tau_vec, dtype=np.int64)
        needed_files: Dict[str, List[str]] = {}
        for pn in names:
            fpr1 = r1_index.weight_map.get(pn)
            if fpr1 is not None:
                needed_files.setdefault(fpr1, []).append(pn)
        r1_loaded: Dict[str, Dict[str, torch.Tensor]] = {fp: st_load_torch(fp) for fp in needed_files.keys()}
        for pn, arr_base in _iter_params_in_group(names, base_index):
            fpr1 = r1_index.weight_map[pn]
            arr_r1 = r1_loaded[fpr1][pn].to(dtype=torch.float32, device="cpu").numpy()
            d = (arr_r1 - arr_base).astype(np.float32, copy=False).ravel()
            sumsq += float(np.dot(d, d))
            ad = np.abs(d)
            for j, tau in enumerate(tau_vec):
                counts[j] += int(np.count_nonzero(ad > tau))
        delta_norm = math.sqrt(sumsq)
        for j, q in enumerate(qlist):
            alpha = 1.0 if delta_norm == 0.0 else min(1.0, float(B_vec[j] / delta_norm))
            l2_alpha.at[f"{q:.2f}", g] = alpha
            l2_pct.at[f"{q:.2f}", g] = 0.0 if alpha >= 1.0 else 100.0
            entry_above_tau_pct.at[f"{q:.2f}", g] = 100.0 * (counts[j] / n_l)
    return l2_pct, l2_alpha, entry_above_tau_pct

# ------------------------------
# CLI
# ------------------------------

def str2bool(v):
    if isinstance(v, bool): return v
    if v is None: return False
    s = str(v).strip().lower()
    if s in ("1","true","t","yes","y","on"): return True
    if s in ("0","false","f","no","n","off"): return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def parse_args():
    ap = argparse.ArgumentParser(description="PUM clipping probe — v3.8 (clean)")
    ap.add_argument("--base-id", type=str, required=True, help="HF repo id or local path for θ_base")
    ap.add_argument("--r1-id", type=str, required=True, help="HF repo id or local path for θ_{r-1}")
    ap.add_argument("--revision-base", type=str, default=None)
    ap.add_argument("--revision-r1", type=str, default=None)
    ap.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", None))

    ap.add_argument("--gamma", type=float, default=1.3)
    ap.add_argument("--rho-default", type=float, default=0.02)
    ap.add_argument("--rho-norm", type=float, default=1.0)
    ap.add_argument("--alpha-q", type=float, default=0.95)

    ap.add_argument("--sample-frac", type=float, default=0.25, help="Sampling for legacy aggregates")
    ap.add_argument("--per-layer-sample-frac", type=float, default=1.0, help="Sampling for per-layer quantiles")

    # Per-layer toggles
    ap.add_argument("--base_quantile", "--base-quantile", nargs="?", const=True, default=False, type=str2bool)
    ap.add_argument("--task_quantile", "--task-quantile", nargs="?", const=True, default=False, type=str2bool)
    ap.add_argument("--alpha_table", "--alpha-table", nargs="?", const=True, default=False, type=str2bool)

    # Sanity/debug
    ap.add_argument("--sanity", nargs="?", const=True, default=False, type=str2bool)
    ap.add_argument("--nz-eps", type=float, default=0.0)

    ap.add_argument("--outdir", type=str, default="outputs/pum_clip_probe")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("=== PUM clipping probe (v3.8 clean) ===")
    print(f"base id     : {args.base_id}")
    print(f"r1 id       : {args.r1_id}")
    print(f"gamma/rho   : gamma={args.gamma} | rho_default={args.rho_default} | rho_norm={args.rho_norm}")
    print(f"alpha-q     : {args.alpha_q}")
    print(f"toggles     : base_q={args.base_quantile} | task_q={args.task_quantile} | alpha_table={args.alpha_table}")
    print(f"sampling    : legacy={args.sample_frac} | per-layer={args.per_layer_sample_frac}")
    print(f"outdir      : {args.outdir}")
    sys.stdout.flush()

    t0 = time.time()
    base_idx = build_index(args.base_id, revision=args.revision_base, token=args.hf_token)
    r1_idx   = build_index(args.r1_id,   revision=args.revision_r1,   token=args.hf_token)
    print(f"[ok] snapshots ready in {(time.time()-t0):.1f}s")

    L = detect_num_layers(base_idx.weight_map)
    print(f"[info] detected number of layers: {L}")
    comp_layers = build_component_layer_map(base_idx.weight_map)

    # Sanity/debug
    if args.sanity:
        base_sz = safetensor_total_size_bytes(base_idx.root)
        r1_sz   = safetensor_total_size_bytes(r1_idx.root)
        print(f"[sanity] .safetensors sizes — base: {base_sz/1e9:.3f} GB | r1: {r1_sz/1e9:.3f} GB")
        if r1_sz < 1.0e9:
            print("[sanity][warn] r1 total size < 1 GB; repo might not host full weights.")
        delta_l2, delta_nz, delta_max = per_layer_delta_debug_tables(base_idx, r1_idx, comp_layers, L, nz_eps=args.nz_eps)
        delta_l2.to_csv(os.path.join(args.outdir, "per_layer_delta_l2.csv"))
        delta_nz.to_csv(os.path.join(args.outdir, "per_layer_delta_nonzero_pct.csv"))
        delta_max.to_csv(os.path.join(args.outdir, "per_layer_delta_maxabs.csv"))

    # Per-layer quantiles (q = 0..1 step 0.02) + RMS row
    q_grid = [i / 50.0 for i in range(51)]

    if args.base_quantile:
        print("[step PL-1] per-layer base quantiles (+RMS row)...")
        pl_base = per_layer_quantiles_abs(base_idx, comp_layers, L, q_grid, sample_frac=args.per_layer_sample_frac)
        for cname, df in pl_base.items():
            df.to_csv(os.path.join(args.outdir, f"{cname}_quantile.csv"), index_label="i")

    if args.task_quantile:
        print("[step PL-2] per-layer task-vector quantiles (+RMS row)...")
        pl_task = per_layer_quantiles_abs_delta(base_idx, r1_idx, comp_layers, L, q_grid, sample_frac=args.per_layer_sample_frac)
        for cname, df in pl_task.items():
            df.to_csv(os.path.join(args.outdir, f"{cname}_task_quantile.csv"), index_label="i")

    if args.alpha_table:
        print("[step PL-3] per-layer alpha table...")
        alpha_df = per_layer_alpha_table(base_idx, r1_idx, comp_layers, L, alpha_q=args.alpha_q, gamma=args.gamma, rho_default=args.rho_default, rho_norm=args.rho_norm)
        alpha_df.to_csv(os.path.join(args.outdir, "per_layer_alpha.csv"), index_label="layer")

    # Legacy aggregated outputs
    qlist = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.96,0.98,0.99]
    groups_full = build_groups(base_idx.weight_map, grouping="per-type")
    groups = _intersect_group_names(groups_full, r1_idx)
    groups = {g: [pn for pn in names if pn in base_idx.weight_map and pn in r1_idx.weight_map] for g, names in groups.items() if len(names) > 0}
    if groups:
        print("[step AGG-1] aggregated base quantiles...")
        base_quant, n_entries = compute_quantiles_abs_values(base_idx, groups, qlist, sample_frac=args.sample_frac)
        base_quant.to_csv(os.path.join(args.outdir, "base_quantiles.csv"))
        print("[step AGG-2] aggregated delta quantiles...")
        delta_quant = compute_quantiles_abs_delta(base_idx, r1_idx, groups, qlist, sample_frac=args.sample_frac)
        delta_quant.to_csv(os.path.join(args.outdir, "delta_quantiles.csv"))
        print("[step AGG-3] aggregated clipping diagnostics...")
        l2_pct, l2_alpha, entry_pct = compute_clipping_and_entrywise(base_quant, n_entries, base_idx, r1_idx, groups, qlist, gamma=args.gamma, rho_default=args.rho_default, rho_norm=args.rho_norm)
        l2_pct.to_csv(os.path.join(args.outdir, "l2_clipped_pct.csv"))
        l2_alpha.to_csv(os.path.join(args.outdir, "l2_scale_alpha.csv"))
        entry_pct.to_csv(os.path.join(args.outdir, "entrywise_above_tau_pct.csv"))

    print("\nDone.")
    print(f"Outputs in: {args.outdir}")

if __name__ == "__main__":
    main()

# ------------------------------
# Sample usage
# ------------------------------
# Base + Task + Alpha (with RMS row i=51 in the per-layer CSVs):
# python pum_clip_probe.py \
#   --base-id meta-llama/Llama-3.2-1B-Instruct \
#   --r1-id open-unlearning/pos_tofu_Llama-3.2-1B-Instruct_full_lr2e-05_wd0.01_epoch10 \
#   --base_quantile True \
#   --task_quantile True \
#   --alpha_table True \
#   --alpha-q 0.95 \
#   --gamma 1.3 --rho-default 0.02 --rho-norm 1.0 \
#   --per-layer-sample-frac 1.0 \
#   --outdir outputs/clip_test_all