import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()

# TRUE BEST from full scan
PE = "evo2__mod0004__h28__mutant_0000"
PR = "rf3__mod0076__h00__mutant_0000"
SE = "evo2__mod0002__h06__mutant_0000"
SR = "rf3__mod0014__h12__mutant_0000"

def prot_cka(Etag, Rtag):
    eh = evo2_data[Etag].squeeze(0)
    rh = rf3_data[Rtag][0]
    ea = codon_pool_evo2_to_aa(eh)
    ra = extract_rf3_chain_a(rh)
    c = linear_cka(ea.float(), ra.float()).item()
    return c, ea, ra

# ── PROTEIN ANALYSIS ──
cka_p, ea, ra = prot_cka(PE, PR)
L = ea.shape[0]

shuf_p = [linear_cka(ea[torch.randperm(L)].float(), ra.float()).item() for _ in range(50)]

per_res = torch.zeros(L)
for i in range(L):
    de = (ea - ea[i]).norm(dim=1)
    dr = (ra - ra[i]).norm(dim=1)
    de_c = (de - de.mean()) / de.std().clamp_min(1e-8)
    dr_c = (dr - dr.mean()) / dr.std().clamp_min(1e-8)
    per_res[i] = (de_c * dr_c).mean()

sort_v, sort_i = per_res.sort(descending=True)

domains = [("REC",0,126),("WED",126,211),("Linker",211,220),("RuvC-I",220,370),("TNB",370,399),("RuvC-II",399,420)]
sites = {"Active225":224,"Active324":323,"Active401":400,"Bind372":371,"Bind375":374,"Bind391":390,"Bind394":393}

rpt = []
rpt.append(f"# Cross-Model Attention Alignment Report\n")
rpt.append(f"## evo2__mod0004__h28 (L31) ↔ rf3__mod0076__h00 (Diffusion block 18)\n")
rpt.append(f"")
rpt.append(f"### 1. Global CKA/RSA")
rpt.append(f"CKA = {cka_p:.4f}")
rpt.append(f"shuffle (50×) → {np.mean(shuf_p):.4f} ± {np.std(shuf_p):.4f}  ({(1-np.mean(shuf_p)/cka_p)*100:.1f}% reduction)")
rpt.append(f"")
rpt.append(f"### 2. Per-Residue Alignment (top-20)")
for k in range(20):
    rpt.append(f"  {sort_i[k].item():4d}: {sort_v[k].item():.4f}")
rpt.append(f"  ...")
for k in range(5):
    rpt.append(f"  {sort_i[-5+k].item():4d}: {sort_v[-5+k].item():.4f}")
rpt.append(f"")

# Domain enrichment
rpt.append(f"### 3. Domain Enrichment (UniProt A0A2U3D0N8)")
rpt.append(f"| Domain | Span | mean_r | Δ | Z | p |")
rpt.append(f"|--------|------|--------|----|----|----|")
for name, s, e in domains:
    mask = torch.zeros(L, dtype=torch.bool); mask[s:e] = True
    mr = per_res[mask].mean().item()
    delta = mr - per_res[~mask].mean().item()
    torch.manual_seed(42)
    w = mask.sum().item(); null = torch.empty(10000)
    for k in range(10000):
        idx = torch.randperm(L)[:w]
        m = torch.zeros(L, dtype=torch.bool); m[idx]=True
        null[k] = per_res[m].mean().item() - per_res[~m].mean().item()
    p = ((null>=delta).sum().item()+1)/10001
    z = (delta - null.mean().item()) / null.std().clamp_min(1e-8).item()
    sig = "**" if p<0.01 else ("*" if p<0.05 else "")
    rpt.append(f"| {name:<6s} | {s:3d}-{e:3d} | {mr:.4f} | {delta:+7.4f} | {z:+6.2f} | {p:.4f} | {sig} |")
rpt.append(f"")

# Catalytic site ranking
rpt.append(f"### 4. Catalytic & Binding Site Ranking")
for name, pos0 in sites.items():
    rank = (sort_i == pos0).nonzero()[0].item() + 1
    rpt.append(f"  {name}: pos={pos0+1} rank={rank}/{L} (top {rank/L*100:.1f}%) r={per_res[pos0].item():.4f}")
rpt.append(f"  Binding enrichment p={((sort_i[:4]>=371)*(sort_i[:4]<=393)).sum().item()/L:.4f}")

# Multi-width window
rpt.append(f"\n### 5. Multi-Width Window CKA")
for width in [5,10,20,30,50,100]:
    windows = []
    for i in range(L-width+1):
        c = linear_cka(ea[i:i+width].float(), ra[i:i+width].float()).item()
        windows.append(c)
    rpt.append(f"  width={width:3d}: mean={np.mean(windows):.4f}  max={np.max(windows):.4f}  max_pos={np.argmax(windows)}")

# ── sgRNA ANALYSIS ──
rpt.append(f"\n---\n")
rpt.append(f"## evo2__mod0002__h06 (L17) ↔ rf3__mod0014__h12 (Pairformer block 14)\n")

eh = evo2_data[SE].squeeze(0)
rh = rf3_data[SR][0]
d = min(eh.shape[-1], rh.shape[-1])

def sgcka(lo, hi):
    sp = sorted(set(rf3_pids[(rf3_pids>=lo)&(rf3_pids<hi)]) & set(evo2_pids[(evo2_pids>=lo)&(evo2_pids<hi)]))
    if len(sp) < 2: return 0, 0, len(sp)
    R = torch.stack([rh[np.where(rf3_pids==p)[0][0],:d].float() for p in sp])
    E = torch.stack([eh[np.where(evo2_pids==p)[0],:d].float().mean(dim=0) for p in sp])
    c = linear_cka(E, R).item()
    sh = np.mean([linear_cka(E[torch.randperm(len(sp))], R).item() for _ in range(30)])
    return c, sh, len(sp)

rpt.append(f"| Region | n | CKA | shuffle |")
rpt.append(f"|--------|---|-----|---------|")
for name, lo, hi in [("tracrRNA",1000,1172),("repeat",1176,1205),("spacer",1205,1225)]:
    c, sh, n = sgcka(lo, hi)
    rpt.append(f"| {name:<9s} | {n:3d} | {c:.4f} | {sh:.4f} |")

# ── Full head distribution ──
rpt.append(f"\n---\n")
rpt.append(f"## Top-10 Protein CKA Head Pairs\n")
cka_mat = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2_full10/full_cka_matrix.pt", map_location='cpu', weights_only=False)
Mp = cka_mat['M_prot']
Ms = cka_mat['M_sg']
etags = cka_mat['evo2_tags']
rtags = cka_mat['rf3_tags']

for rank in range(10):
    idx = Mp.flatten().argsort(descending=True)[rank]
    i, j = idx // Mp.shape[1], idx % Mp.shape[1]
    rpt.append(f"  #{rank+1}: {etags[i]} ↔ {rtags[j]} CKA={Mp[i,j]:.4f}")

rpt.append(f"\n## Top-10 sgRNA CKA Head Pairs\n")
for rank in range(10):
    idx = Ms.flatten().argsort(descending=True)[rank]
    i, j = idx // Ms.shape[1], idx % Ms.shape[1]
    rpt.append(f"  #{rank+1}: {etags[i]} ↔ {rtags[j]} CKA={Ms[i,j]:.4f}")

with open("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2_full10/CROSS_MODEL_FINAL_REPORT.md","w") as f:
    f.write("\n".join(rpt))
print("\n".join(rpt))
