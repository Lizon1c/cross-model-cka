"""Residualized full scan + position-aware domain enrichment."""
import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np, time
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

device = torch.device("cuda:0")
t0 = time.time()

evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()

L = 422

def regress_poly(X, deg):
    """Remove polynomial position trends from stacked features (N, L, C) or (L, C)."""
    orig_dim = X.dim()
    if orig_dim == 2:
        X = X.unsqueeze(0)
    N, Lx, C = X.shape
    t = torch.linspace(0, 1, Lx, device=device)
    basis = torch.stack([t**d for d in range(deg+1)], dim=1)  # (L, deg+1)
    # Solve X_i ≈ basis @ betas_i for each i
    # Use lstsq: (L, deg+1) @ (deg+1, N*C) stacked? No, do per sample
    X_res = torch.zeros_like(X)
    for i in range(N):
        betas = torch.linalg.lstsq(basis, X[i]).solution  # (deg+1, C)
        X_res[i] = X[i] - basis @ betas
    if orig_dim == 2:
        return X_res[0]
    return X_res

# ── Precompute all features ──
# RF3
rf3_feats = {}; rf3_tags = []
for k, v in rf3_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() != 3 or v.shape[1] != 1145: continue
    rf3_feats[k.rsplit("__",1)[0]] = extract_rf3_chain_a(v[0]).float()
    rf3_tags.append(k.rsplit("__",1)[0])
rf3_tags = sorted(rf3_tags)

# EVO2
evo2_feats = {}; evo2_tags = []
for k, v in evo2_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() < 2: continue
    h = v.squeeze(0)
    if h.dim() == 2 and h.shape[0] == 1857:
        evo2_feats[k.rsplit("__",1)[0]] = codon_pool_evo2_to_aa(h).float()
        evo2_tags.append(k.rsplit("__",1)[0])
evo2_tags = sorted(evo2_tags)

# Stack and to GPU
E = torch.stack([evo2_feats[t] for t in evo2_tags]).to(device)  # (160, 422, 128)

# RF3: max dim varies, pad to max
C_max_rf3 = max(rf3_feats[t].shape[1] for t in rf3_tags)
R = torch.zeros(len(rf3_tags), L, C_max_rf3, device=device)
for i, t in enumerate(rf3_tags):
    f = rf3_feats[t].to(device)
    R[i, :, :f.shape[1]] = f
# Pad dims ≠ homogeneous — use group for CKA

def batch_cka(E_stack, R_stack):
    """E: (Ne, L, Ce), R: (Nr, L, Cr) — compute CKA matrix (Ne, Nr)."""
    Ne, Lx, Ce = E_stack.shape
    Nr, _, Cr = R_stack.shape
    Ec = E_stack - E_stack.mean(dim=1, keepdim=True)
    Rc = R_stack - R_stack.mean(dim=1, keepdim=True)
    Eg = torch.bmm(Ec, Ec.transpose(1,2))  # (Ne, L, L)
    Rg = torch.bmm(Rc, Rc.transpose(1,2))  # (Nr, L, L)
    Ev = Eg.reshape(Ne, -1)  # (Ne, L²)
    Rv = Rg.reshape(Nr, -1)  # (Nr, L²)
    En = Eg.norm(dim=(1,2))  # (Ne,)
    Rn = Rg.norm(dim=(1,2))  # (Nr,)
    numer = Ev @ Rv.T
    denom = En.unsqueeze(1) * Rn.unsqueeze(0)
    return numer / denom.clamp_min(1e-8)

# ── 1. Position-aware domain enrichment ──
print("=== 1. Position-aware domain enrichment ===")
# Best pair per-residue scores
eh_best = evo2_feats["evo2__mod0004__h28"].to(device)
rh_best = rf3_feats["rf3__mod0076__h00"].to(device)
d_rh = rh_best.shape[1]
eh_best_d = eh_best[:, :d_rh]  # truncate to match RF3 dims for distance

def pearson(x, y):
    xc = x - x.mean(); yc = y - y.mean()
    return (xc @ yc) / (xc.norm() * yc.norm()).clamp_min(1e-12)

per_res = torch.zeros(L)
for i in range(L):
    de = (eh_best_d - eh_best_d[i]).norm(dim=1)
    dr = (rh_best - rh_best[i]).norm(dim=1)
    per_res[i] = pearson(de, dr)

domains = [("REC",0,126),("WED",126,211),("Linker",211,220),("RuvC-I",220,370),("TNB",370,399),("RuvC-II",399,420)]

print(f"{'Domain':<12s} {'obs':>8s} {'interval_null':>14s} {'circ_shift':>12s} {'local':>12s} {'residual_CKA':>14s}")
for name, s, e in domains:
    m = e - s
    obs = per_res[s:e].mean().item()
    
    # Interval null: all contiguous windows of same length
    interval_scores = [per_res[i:i+m].mean().item() for i in range(L-m+1)]
    p_interval = (sum(1 for v in interval_scores if v >= obs) + 1) / (len(interval_scores) + 1)
    
    # Circular shift null
    circ_scores = []
    for shift in range(L):
        start = (s + shift) % L
        end = start + m
        if end <= L:
            circ_scores.append(per_res[start:end].mean().item())
        else:
            circ_scores.append((per_res[start:].sum() + per_res[:end-L].sum()).item() / m)
    p_circ = (sum(1 for v in circ_scores if v >= obs) + 1) / (len(circ_scores) + 1)
    
    # Local matched null: windows within ±50 residues
    local_start = max(0, s - 50)
    local_end = min(L - m, e + 50)
    local_scores = [per_res[i:i+m].mean().item() for i in range(local_start, local_end+1) if i+m <= L]
    p_local = (sum(1 for v in local_scores if v >= obs) + 1) / (len(local_scores) + 1)
    
    # Residual CKA after cubic removal (domain-wise)
    # Remove cubic trend from features, recompute CKA on residuals
    ea_res = regress_poly(eh_best_d, 3)
    ra_res = regress_poly(rh_best, 3)
    # Compute per-residue on residuals
    pr_res = torch.zeros(L)
    for i in range(L):
        de = (ea_res - ea_res[i]).norm(dim=1)
        dr = (ra_res - ra_res[i]).norm(dim=1)
        pr_res[i] = pearson(de, dr)
    domain_res = pr_res[s:e].mean().item()
    
    print(f"{name:<12s} {obs:>8.4f}  p={p_interval:.4f}{'**' if p_interval<0.01 else ('*' if p_interval<0.05 else '')}  p={p_circ:.4f}{'**' if p_circ<0.01 else ('*' if p_circ<0.05 else '')}  p={p_local:.4f}{'**' if p_local<0.01 else ('*' if p_local<0.05 else '')}  res_r={domain_res:.4f}")

print(f"\n  Interval null: all contiguous {m}-residue windows across protein")
print(f"  Circ shift: circular shift of domain interval")
print(f"  Local: windows within ±50 residues of domain")
print(f"  Residual CKA: per-residue score on cubic-regressed features")

# ── 2. Residualized full scan ──
print(f"\n=== 2. Residualized full scan ===")
for deg, label in [(1, "linear"), (3, "cubic")]:
    print(f"\n--- {label} residualization ---")
    E_res = regress_poly(E, deg)
    R_res = regress_poly(R, deg)
    CKA_res = batch_cka(E_res, R_res)
    max_val = CKA_res.max().item()
    bi, bj = CKA_res.argmax().item() // len(rf3_tags), CKA_res.argmax().item() % len(rf3_tags)
    print(f"  Raw max CKA:       0.851")
    print(f"  {label} residual max CKA: {max_val:.4f}")
    print(f"  Best pair: {evo2_tags[bi]} ↔ {rf3_tags[bj]}")
    print(f"  Residual mean: {CKA_res.mean().item():.4f}  std: {CKA_res.std().item():.4f}")
    # Top-5
    vals, idxs = CKA_res.flatten().topk(5)
    for r, (v, idx) in enumerate(zip(vals, idxs)):
        i, j = idx.item() // len(rf3_tags), idx.item() % len(rf3_tags)
        print(f"    #{r+1}: {evo2_tags[i]:25s} {rf3_tags[j]:25s} CKA={v.item():.4f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
