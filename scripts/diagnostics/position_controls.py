"""Position-trend regression, block shuffle, EVO2 self-CKA under shift."""
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

# Best pair features on GPU
eh = evo2_data["evo2__mod0004__h28__mutant_0000"].squeeze(0).float().to(device)
rh = rf3_data["rf3__mod0076__h00__mutant_0000"][0].float().to(device)
ea = codon_pool_evo2_to_aa(eh.cpu()).float().to(device)
ra = extract_rf3_chain_a(rh.cpu()).float().to(device)
L = ea.shape[0]  # 422

# ── EXPERIMENT A: Polynomial position trend regression ──
def regress_out_poly(X, degree):
    """Regress out polynomial position trends from feature matrix X (L, C)."""
    t = torch.arange(L, device=device).float() / L  # [0, 1]
    basis = torch.stack([t**d for d in range(degree+1)], dim=1)  # (L, degree+1)
    # Solve: X ≈ basis @ betas, residual = X - basis @ betas
    betas = torch.linalg.lstsq(basis, X).solution  # (degree+1, C)
    X_pred = basis @ betas
    return X - X_pred

print("=== A: Polynomial position-trend regression ===")
for deg in [0, 1, 2, 3]:
    ea_res = regress_out_poly(ea, deg)
    ra_res = regress_out_poly(ra, deg)
    c = linear_cka(ea_res, ra_res).item()
    print(f"  After {deg}° polynomial removal: CKA={c:.4f}  (retention: {c/linear_cka(ea,ra).item()*100:.1f}%)")

# ── EXPERIMENT B: Block shuffle ──
def block_shuffle(X, block_size):
    """Shuffle blocks of rows, preserving intra-block order."""
    n_blocks = L // block_size
    perm = torch.randperm(n_blocks, device=device)
    X_shuf = X.clone()
    for i, p in enumerate(perm):
        X_shuf[i*block_size:(i+1)*block_size] = X[p.item()*block_size:(p.item()+1)*block_size]
    return X_shuf

print("\n=== B: Block shuffle analysis ===")
print(f"  Baseline (no shuffle):  CKA={linear_cka(ea, ra).item():.4f}")
for bs in [3, 5, 10, 25, 50, 100]:
    ckas = []
    for _ in range(20):
        ea_bs = block_shuffle(ea, bs)
        ckas.append(linear_cka(ea_bs, ra).item())
    print(f"  Block size={bs:3d} (≈{bs*3:.0f}nt): CKA={np.mean(ckas):.4f} ± {np.std(ckas):.4f}")

# ── EXPERIMENT C: EVO2 self-CKA under shift ──
print("\n=== C: EVO2 self-CKA under nucleotide shift ===")
# Use CDS-only for clean interpretation
with open('/mnt/j/conda_envs/foundry/run_attn_extraction.py') as f:
    code = f.read().split('if __name__')[0]
exec(code)
cds_start = len(ASCAS12F_LOCUS_LEFT_FLANK)
cds_end = cds_start + len(reverse_translate_ascas12f(WT_PROTEIN_SEQ))
eh_cds = eh[cds_start:cds_end]  # (1269, 128)

def pool_cds_consecutive(X):
    return X[:422*3].reshape(422, 3, -1).mean(dim=1)  # (422, 128)

ea0 = pool_cds_consecutive(eh_cds)
for s in [1, 2, 3, 10, 50, 100, 500, 1000]:
    shifted = torch.cat([eh_cds[s:], eh_cds[:s]])
    eas = pool_cds_consecutive(shifted)
    c = linear_cka(ea0, eas).item()
    # Also compute CKA with same features permuted (null reference)
    perm = torch.randperm(422, device=device)
    c_perm = linear_cka(ea0, eas[perm]).item()
    print(f"  EVO2-CDS shift={s:4d}: self-CKA={c:.4f}  permuted={c_perm:.4f}")

# ── EXPERIMENT D: Position-only baseline ──
print("\n=== D: Pure position-trend signal vs EVO2 ===")
t = torch.arange(L, device=device).float() / L
# Sinusoidal position encoding (RoPE-style)
pos_enc = torch.zeros(L, 128, device=device)
for i in range(64):
    freq = 10000 ** (2 * i / 128)
    pos_enc[:, 2*i] = torch.sin(t * freq)
    pos_enc[:, 2*i+1] = torch.cos(t * freq)
c_pos = linear_cka(pos_enc, ra).item()
print(f"  Sinusoidal pos-enc (128d, RoPE-style) × RF3: CKA={c_pos:.4f}")

# Polynomial-only signal
poly_basis = torch.stack([t**d for d in range(4)], dim=1)  # (422, 4)
poly_proj = poly_basis @ torch.randn(4, 128, device=device) * 0.1
c_poly = linear_cka(poly_proj, ra).item()
print(f"  Cubic polynomial (random coeffs) × RF3: CKA={c_poly:.4f}")

# ── EXPERIMENT E: Distance normalization on main pair ──
print("\n=== E: Distance normalization on MAIN pair (mod0004_h28↔mod0076_h00) ===")
def dist_norm_cka(X, Y):
    DX = torch.cdist(X, X, p=2)
    DY = torch.cdist(Y, Y, p=2)
    # Regress out distance trend
    dist = torch.arange(L, device=device).float()[:, None] - torch.arange(L, device=device).float()[None, :]
    dist = dist.abs()
    # Normalize: subtract per-distance mean
    DX_n = DX.clone(); DY_n = DY.clone()
    for d in range(L):
        mask = dist == d
        if mask.any():
            DX_n[mask] -= DX[mask].mean()
            DY_n[mask] -= DY[mask].mean()
    return linear_cka(DX, DY).item(), linear_cka(DX_n, DY_n).item()

cka_dist_raw, cka_dist_norm = dist_norm_cka(ea, ra)
print(f"  Raw distance CKA:     {cka_dist_raw:.4f}")
print(f"  Distance-normalized:  {cka_dist_norm:.4f}")
print(f"  Retention:            {cka_dist_norm/cka_dist_raw*100:.1f}%")

# ── EXPERIMENT F: sgRNA exact values ──
print("\n=== F: sgRNA exact values on protein best pair ===")
d = min(eh.shape[-1], rh.shape[-1])
for name, lo, hi in [("tracrRNA", 1000, 1172), ("repeat", 1176, 1205), ("spacer", 1205, 1225)]:
    sp = sorted(set(rf3_pids[(rf3_pids>=lo)&(rf3_pids<hi)]) & set(evo2_pids[(evo2_pids>=lo)&(evo2_pids<hi)]))
    if len(sp) < 2: 
        print(f"  {name}: n_shared={len(sp)} CKA=NA")
        continue
    R = torch.stack([rh[np.where(rf3_pids==p)[0][0],:d] for p in sp])
    E = torch.stack([eh[np.where(evo2_pids==p)[0],:d].float().mean(dim=0) for p in sp])
    c = linear_cka(E, R).item()
    print(f"  {name}: n_shared={len(sp)} CKA={c:.6f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
