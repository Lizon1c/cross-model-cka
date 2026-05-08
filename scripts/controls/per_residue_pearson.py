"""Fix per-residue RSA to strict Pearson correlation ([-1,1]), recompute all tables."""
import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()

# ── Strict Pearson ──
def pearson(x, y):
    xc = x - x.mean(); yc = y - y.mean()
    return (xc @ yc) / (xc.norm() * yc.norm()).clamp_min(1e-12)

def per_residue_pearson(ea, ra):
    """Strict Pearson correlation of distance profiles — guaranteed [-1,1]."""
    L = ea.shape[0]
    result = torch.zeros(L)
    for i in range(L):
        de = (ea - ea[i]).norm(dim=1)
        dr = (ra - ra[i]).norm(dim=1)
        result[i] = pearson(de, dr)
    assert result.max() <= 1.0 + 1e-6, f"BUG: max={result.max()}"
    assert result.min() >= -1.0 - 1e-6, f"BUG: min={result.min()}"
    return result

def perm_test_enrichment(scores, mask, n_perm=10000):
    torch.manual_seed(42)
    L = len(scores)
    w = mask.sum().item()
    obs_d = scores[mask].mean().item() - scores[~mask].mean().item()
    null = torch.empty(n_perm)
    for k in range(n_perm):
        idx = torch.randperm(L)[:w]
        m = torch.zeros(L, dtype=torch.bool); m[idx] = True
        null[k] = scores[m].mean().item() - scores[~m].mean().item()
    p = (int((null >= obs_d).sum().item()) + 1) / (n_perm + 1)
    z = float((obs_d - null.mean().item()) / null.std().clamp_min(1e-8).item())
    return scores[mask].mean().item(), obs_d, z, p

# ── Best protein pair ──
PE = "evo2__mod0004__h28__mutant_0000"
PR = "rf3__mod0076__h00__mutant_0000"
eh = evo2_data[PE].squeeze(0)
rh = rf3_data[PR][0]
ea = codon_pool_evo2_to_aa(eh)
ra = extract_rf3_chain_a(rh)
L = ea.shape[0]

pr = per_residue_pearson(ea.float(), ra.float())
print(f"Per-residue Pearson: [{pr.min().item():.4f}, {pr.max().item():.4f}]  ✓ in [-1,1]")

# Domain enrichment
domains = [("REC",0,126),("WED",126,211),("Linker",211,220),("RuvC-I",220,370),("TNB",370,399),("RuvC-II",399,420)]
print(f"\n{'Domain':<12s} {'mean':>8s} {'Δ':>10s} {'Z':>8s} {'p':>10s} {'sig':>4s}")
for name, s, e in domains:
    mask = torch.zeros(L, dtype=torch.bool); mask[s:e] = True
    mean_r, delta, z, p = perm_test_enrichment(pr, mask)
    sig = "**" if p<0.01 else ("*" if p<0.05 else "ns")
    print(f"{name:<12s} {mean_r:>8.4f} {delta:>+10.4f} {z:>+8.2f} {p:>10.4f} {sig:>4s}")

# Binding site enrichment
sites = {"Active225":224,"Active324":323,"Active401":400,"Bind372":371,"Bind375":374,"Bind391":390,"Bind394":393}
print(f"\nBinding/catalytic sites:")
for name, pos in sites.items():
    rank = (pr.argsort(descending=True) == pos).nonzero()[0].item() + 1
    print(f"  {name}: pos={pos+1} rank={rank}/{L} ({rank/L*100:.1f}%) score={pr[pos].item():.4f}")

# 4-site binding enrichment
bs_mask = torch.zeros(L, dtype=torch.bool)
for pos in [371,374,390,393]: bs_mask[pos] = True
bs_mean, bs_delta, bs_z, bs_p = perm_test_enrichment(pr, bs_mask)
print(f"  Combined 4 binding sites: mean={bs_mean:.4f} Δ={bs_delta:+.4f} Z={bs_z:+.2f} p={bs_p:.4f}")

# ── Best sgRNA pair ──
print(f"\n=== sgRNA ===")
SE = "evo2__mod0002__h06__mutant_0000"
SR = "rf3__mod0014__h12__mutant_0000"
eh_s = evo2_data[SE].squeeze(0)
rh_s = rf3_data[SR][0]
d = min(eh_s.shape[-1], rh_s.shape[-1])

def sgcka(lo, hi):
    sp = sorted(set(rf3_pids[(rf3_pids>=lo)&(rf3_pids<hi)]) & set(evo2_pids[(evo2_pids>=lo)&(evo2_pids<hi)]))
    if len(sp) < 2: return 0, 0, len(sp), 0
    R = torch.stack([rh_s[np.where(rf3_pids==p)[0][0],:d].float() for p in sp])
    E = torch.stack([eh_s[np.where(evo2_pids==p)[0],:d].float().mean(dim=0) for p in sp])
    c = linear_cka(E, R).item()
    sh = np.mean([linear_cka(E[torch.randperm(len(sp))], R).item() for _ in range(30)])
    return c, sh, len(sp), len(sp)

for name, lo, hi in [("tracrRNA",1000,1172),("repeat",1176,1205),("spacer",1205,1225)]:
    c, sh, n, total = sgcka(lo, hi)
    print(f"  {name}: n_shared={n} CKA={c:.4f} shuffle→{sh:.4f}")

print(f"\nAll values recomputed with strict Pearson. r_max={pr.max().item():.6f} ✓")
