import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()

# ── TRUE BEST PAIRS from full scan ──
best_prot_evo2 = "evo2__mod0004__h28__mutant_0000"
best_prot_rf3  = "rf3__mod0076__h00__mutant_0000"
best_sg_evo2   = "evo2__mod0002__h06__mutant_0000"
best_sg_rf3    = "rf3__mod0014__h12__mutant_0000"

# ── Protein analysis ──
evo2_h = evo2_data[best_prot_evo2].squeeze(0)
rf3_h = rf3_data[best_prot_rf3][0]
evo2_aa = codon_pool_evo2_to_aa(evo2_h)
rf3_aa = extract_rf3_chain_a(rf3_h)
L = evo2_aa.shape[0]

cka = linear_cka(evo2_aa.float(), rf3_aa.float()).item()
# shuffle
shuf = [linear_cka(evo2_aa[torch.randperm(L)].float(), rf3_aa.float()).item() for _ in range(50)]

print(f"=== PROTEIN: {best_prot_evo2.split('__')[0]} ↔ {best_prot_rf3.split('__')[0]} ===")
print(f"CKA={cka:.4f}  shuffle→{np.mean(shuf):.4f}±{np.std(shuf):.4f}")

# Per-residue
per_res = torch.zeros(L)
for i in range(L):
    de = (evo2_aa - evo2_aa[i]).norm(dim=1)
    dr = (rf3_aa - rf3_aa[i]).norm(dim=1)
    de_c = (de - de.mean()) / de.std().clamp_min(1e-8)
    dr_c = (dr - dr.mean()) / dr.std().clamp_min(1e-8)
    per_res[i] = (de_c * dr_c).mean()

top = per_res.topk(10)
print(f"Per-residue top-10: {[(i.item(), f'{v:.4f}') for i,v in zip(top.indices, top.values)]}")

# Domain enrichment (UniProt)
domains = [("REC",0,126),("WED",126,211),("Linker",211,220),("RuvC-I",220,370),("TNB",370,399),("RuvC-II",399,420)]
def perm_test(mask, n_perm=10000):
    torch.manual_seed(42)
    obs_d = per_res[mask].mean().item() - per_res[~mask].mean().item()
    w = mask.sum().item(); null = torch.empty(n_perm)
    for k in range(n_perm):
        idx = torch.randperm(L)[:w]
        m = torch.zeros(L, dtype=torch.bool); m[idx]=True
        null[k] = per_res[m].mean().item() - per_res[~m].mean().item()
    p = ((null>=obs_d).sum().item()+1)/(n_perm+1)
    z = (obs_d - null.mean().item()) / null.std().clamp_min(1e-8).item()
    return per_res[mask].mean().item(), obs_d, z, p

print(f"\nDomain enrichment (UniProt):")
print(f"{'Domain':<14s} {'mean_r':>8s} {'Δ':>10s} {'Z':>8s} {'p':>10s}")
for name, s, e in domains:
    mask = torch.zeros(L, dtype=torch.bool); mask[s:e] = True
    mr, delta, z, p = perm_test(mask)
    print(f"{name:<14s} {mr:>8.4f} {delta:>+10.4f} {z:>+8.2f} {p:>10.4f} {'**' if p<0.01 else ('*' if p<0.05 else '')}")

# ── sgRNA analysis ──
evo2_s = evo2_data[best_sg_evo2].squeeze(0)
rf3_s = rf3_data[best_sg_rf3][0]
d = min(evo2_s.shape[-1], rf3_s.shape[-1])

def sgRNA_cka_by_region(lo, hi):
    sp = sorted(set(rf3_pids[(rf3_pids>=lo)&(rf3_pids<hi)]) & set(evo2_pids[(evo2_pids>=lo)&(evo2_pids<hi)]))
    if len(sp) < 2: return 0
    R = torch.stack([rf3_s[np.where(rf3_pids==p)[0][0],:d].float() for p in sp])
    E = torch.stack([evo2_s[np.where(evo2_pids==p)[0],:d].float().mean(dim=0) for p in sp])
    c = linear_cka(E, R).item()
    sh = np.mean([linear_cka(E[torch.randperm(len(sp))], R).item() for _ in range(30)])
    return c, sh, len(sp)

print(f"\n=== sgRNA: {best_sg_evo2.split('__')[0]} ↔ {best_sg_rf3.split('__')[0]} ===")
for name, lo, hi in [("tracrRNA",1000,1172),("repeat",1176,1205),("spacer",1205,1225)]:
    c, sh, n = sgRNA_cka_by_region(lo, hi)
    print(f"  {name:<12s}: n={n:3d}  CKA={c:.4f}  shuf→{sh:.4f}")
