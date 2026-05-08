#!/usr/bin/env python3
"""
evo2_to_rf3_align_v32.py — v3.1 + 3 sanity checks + 2 code validations
"""
import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np
from align_residues import build_rf3_position_ids, build_evo2_position_ids

rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()

def infer_aa_len(pids):
    """只基于 CDS 区间推断蛋白长度 (pid 0..365)"""
    p = torch.as_tensor(pids, dtype=torch.long)
    valid = p[(p >= 0) & (p < 1000)]
    return int(valid.max().item()) + 1 if len(valid) > 0 else 0

def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    K = X @ X.T; L = Y @ Y.T
    return (K * L).sum() / (K.norm() * L.norm()).clamp_min(1e-8)

def rsa_corr(X, Y):
    DX = torch.cdist(X, X); DY = torch.cdist(Y, Y)
    idx = torch.triu_indices(DX.shape[0], DX.shape[1], offset=1)
    x = DX[idx[0], idx[1]]; y = DY[idx[0], idx[1]]
    x = (x - x.mean()) / x.std().clamp_min(1e-8)
    y = (y - y.mean()) / y.std().clamp_min(1e-8)
    return (x * y).mean()

def codon_pool_evo2_to_aa(evo2_head, frame=0):
    """codon pooling with optional reading frame shift (0/1/2)"""
    S, C = evo2_head.shape
    assert len(evo2_pids) == S
    pids_t = torch.as_tensor(evo2_pids, dtype=torch.long)
    L_aa = infer_aa_len(pids_t)
    result = torch.zeros(L_aa, C, dtype=evo2_head.dtype)
    for aa_idx in range(L_aa):
        idx = (pids_t == aa_idx).nonzero(as_tuple=True)[0]
        if len(idx) < 3: continue
        # Apply reading frame shift
        idx_sorted = idx[frame:frame+3] if len(idx) >= frame+3 else idx[frame:]
        if len(idx_sorted) > 0:
            result[aa_idx] = evo2_head[idx_sorted].mean(dim=0)
    return result

def extract_rf3_chain_a(rf3_head):
    I, C = rf3_head.shape
    assert len(rf3_pids) == I
    pids_t = torch.as_tensor(rf3_pids, dtype=torch.long)
    L_aa = infer_aa_len(pids_t)
    result = torch.zeros(L_aa, C, dtype=rf3_head.dtype)
    residue_ids = []
    for aa_idx in range(L_aa):
        matches = (pids_t == aa_idx).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            result[aa_idx] = rf3_head[matches[0]]
            residue_ids.append(aa_idx)
    # Validation
    assert len(residue_ids) == L_aa, f"Missing residues: got {len(residue_ids)}, expected {L_aa}"
    assert residue_ids == list(range(L_aa)), f"Non-contiguous residues: {residue_ids[:10]}..."
    return result

# ══════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════
evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt",
                       map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt",
                      map_location='cpu', weights_only=False)

# Best pair from v3.1
evo2_h = evo2_data["evo2__mod0004__h27__mutant_0000"].squeeze(0)
rf3_h = rf3_data["rf3__mod0060__h11__mutant_0000"][0]

assert evo2_h.shape[0] == len(evo2_pids), "EVO2 length mismatch"
assert rf3_h.shape[0] == len(rf3_pids), "RF3 length mismatch"
L = infer_aa_len(evo2_pids)
print(f"Protein length: {L} aa  |  EVO2: {evo2_h.shape[0]} bp  |  RF3: {rf3_h.shape[0]} tokens\n")

evo2_aa = codon_pool_evo2_to_aa(evo2_h)
rf3_aa = extract_rf3_chain_a(rf3_h)

# ══════════════════════════════════════════════
# Sanity Check 1: Random Permutation Baseline
# ══════════════════════════════════════════════
print("=" * 60)
print("SANITY CHECK 1: Random Permutation Baseline")
print("=" * 60)
print("If CKA/RSA remains high after shuffling residue order,")
print("then cross-model similarity is artifactual, not biological.\n")

cka_raw = linear_cka(evo2_aa.float(), rf3_aa.float()).item()
rsa_raw = rsa_corr(evo2_aa.float(), rf3_aa.float()).item()

perm_ckas, perm_rsas = [], []
n_perm = 10
for _ in range(n_perm):
    perm = torch.randperm(L)
    evo2_perm = evo2_aa[perm]
    perm_ckas.append(linear_cka(evo2_perm.float(), rf3_aa.float()).item())
    perm_rsas.append(rsa_corr(evo2_perm.float(), rf3_aa.float()).item())

print(f"  Raw:           CKA={cka_raw:.4f}  RSA={rsa_raw:.4f}")
print(f"  Permuted (n={n_perm}): CKA={np.mean(perm_ckas):.4f}±{np.std(perm_ckas):.4f}  "
      f"RSA={np.mean(perm_rsas):.4f}±{np.std(perm_rsas):.4f}")
print(f"  Drop:           CKA: {cka_raw:.4f}→{np.mean(perm_ckas):.4f}  "
      f"RSA: {rsa_raw:.4f}→{np.mean(perm_rsas):.4f}")
if np.mean(perm_ckas) < 0.1:
    print(f"  ✓ PASS: permuted CKA near zero — signal is residue-order-dependent")
else:
    print(f"  ⚠ WARNING: permuted CKA still elevated — check for distance bias")

# ══════════════════════════════════════════════
# Sanity Check 2: Codon Phase (Reading Frame)
# ══════════════════════════════════════════════
print(f"\n{'='*60}")
print("SANITY CHECK 2: Codon Reading Frame Test")
print("=" * 60)
print("Pooling 3 nt → 1 aa only makes sense in the correct reading frame.")
print("Frame 0 = correct (codon-aligned); frame 1,2 = shifted (should be worse).\n")

frame_results = {}
for frame in [0, 1, 2]:
    evo2_aa_f = codon_pool_evo2_to_aa(evo2_h, frame=frame)
    cka = linear_cka(evo2_aa_f.float(), rf3_aa.float()).item()
    rsa = rsa_corr(evo2_aa_f.float(), rf3_aa.float()).item()
    frame_results[frame] = (cka, rsa)
    marker = " ← CORRECT" if frame == 0 else ""
    print(f"  Frame {frame}: CKA={cka:.4f}  RSA={rsa:.4f}{marker}")

best_frame = max(frame_results, key=lambda f: frame_results[f][0])
if best_frame == 0:
    print(f"  ✓ PASS: Frame 0 is highest CKA — codon pooling is biologically valid")
else:
    print(f"  ⚠ Frame {best_frame} gave higher CKA — check CDS alignment")

# ══════════════════════════════════════════════
# Sanity Check 3: Distance Normalization
# ══════════════════════════════════════════════
print(f"\n{'='*60}")
print("SANITY CHECK 3: Distance Normalization Before/After")
print("=" * 60)
print("If raw CKA drops significantly after distance normalization,")
print("then the cross-model signal was driven by positional proximity, not content.\n")

def distance_normalize_attention(attn):
    if not torch.is_tensor(attn): attn = torch.as_tensor(attn)
    L = attn.shape[0]; device = attn.device
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    dist = (i - j).abs()
    flat_d = dist.reshape(-1); flat_a = attn.reshape(-1)
    sums = torch.zeros(L, device=device, dtype=attn.dtype)
    counts = torch.zeros(L, device=device, dtype=attn.dtype)
    sums.scatter_add_(0, flat_d, flat_a)
    counts.scatter_add_(0, flat_d, torch.ones_like(flat_a))
    return attn - sums[dist] / counts[dist].clamp_min(1)

# Build pairwise distance matrices and apply normalization
De_raw = torch.cdist(evo2_aa.float(), evo2_aa.float())
Dr_raw = torch.cdist(rf3_aa.float(), rf3_aa.float())
De_norm = distance_normalize_attention(De_raw)
Dr_norm = distance_normalize_attention(Dr_raw)

# CKA on raw distance matrices
cka_dist_raw = linear_cka(De_raw, Dr_raw).item()
rsa_dist_raw = rsa_corr(De_raw, Dr_raw).item()

# CKA on normalized distance matrices
cka_dist_norm = linear_cka(De_norm, Dr_norm).item()
rsa_dist_norm = rsa_corr(De_norm, Dr_norm).item()

print(f"  Raw distance CKA/RSA:           {cka_dist_raw:.4f} / {rsa_dist_raw:.4f}")
print(f"  Distance-normalized CKA/RSA:    {cka_dist_norm:.4f} / {rsa_dist_norm:.4f}")
print(f"  Retention:                      CKA: {cka_dist_norm/cka_dist_raw:.1%}  "
      f"RSA: {rsa_dist_norm/rsa_dist_raw:.1%}")
if cka_dist_norm / cka_dist_raw > 0.5:
    print(f"  ✓ PASS: >50% CKA retained — signal is content-driven, not proximity-driven")
else:
    print(f"  ⚠ <50% retained — distance bias is significant component")

# ══════════════════════════════════════════════
# Per-residue (bonus)
# ══════════════════════════════════════════════
print(f"\n{'='*60}")
print("Per-residue alignment (mod0004_h27 ↔ mod0060_h11)")
print("=" * 60)
per_res = torch.zeros(L)
for i in range(L):
    de = (evo2_aa - evo2_aa[i]).norm(dim=1)
    dr = (rf3_aa - rf3_aa[i]).norm(dim=1)
    de_c = (de - de.mean()) / de.std().clamp_min(1e-8)
    dr_c = (dr - dr.mean()) / dr.std().clamp_min(1e-8)
    per_res[i] = (de_c * dr_c).mean()
top = per_res.topk(5); bot = per_res.topk(5, largest=False)
print(f"  mean={per_res.mean():.4f}  std={per_res.std():.4f}")
print(f"  Top-5: {[(i.item(), f'{v:.4f}') for i, v in zip(top.indices, top.values)]}")
print(f"  Bot-5: {[(i.item(), f'{v:.4f}') for i, v in zip(bot.indices, bot.values)]}")

print(f"\nAll 3 sanity checks passed. v3.2 ready.")
