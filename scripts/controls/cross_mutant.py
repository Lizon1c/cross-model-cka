"""Cross-mutant robustness: 10 mutants vs fixed RF3 WT."""
import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np, random, time
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

device = torch.device("cuda:0")
t0 = time.time()

# Fixed RF3 WT
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_feats = {}; rf3_tags = []
for k, v in rf3_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() != 3 or v.shape[1] != 1145: continue
    rf3_feats[k.rsplit("__",1)[0]] = extract_rf3_chain_a(v[0]).float()
    rf3_tags.append(k.rsplit("__",1)[0])
rf3_tags = sorted(rf3_tags)

# Pad and to GPU
C_max = max(rf3_feats[t].shape[1] for t in rf3_tags)
R = torch.zeros(len(rf3_tags), 422, C_max, device=device)
for i, t in enumerate(rf3_tags):
    f = rf3_feats[t].to(device); R[i, :, :f.shape[1]] = f
Rc = R - R.mean(dim=1, keepdim=True)
Rg = torch.bmm(Rc, Rc.transpose(1,2))
Rv = Rg.reshape(len(rf3_tags), -1)
Rn = Rg.norm(dim=(1,2))

# Pick 10 random mutants + WT
random.seed(42)
mutant_ids = sorted(random.sample(range(1, 1000), 10))
mutant_ids = [0] + mutant_ids  # include WT
print(f"Mutants: {mutant_ids}")

results = []
for mid in mutant_ids:
    t1 = time.time()
    evo2_data = torch.load(f"/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_{mid:04d}.pt", map_location='cpu', weights_only=False)
    
    evo2_feats = {}; evo2_tags = []
    for k, v in evo2_data.items():
        if not isinstance(v, torch.Tensor) or v.dim() < 2: continue
        h = v.squeeze(0)
        if h.dim() == 2 and h.shape[0] == 1857:
            evo2_feats[k.rsplit("__",1)[0]] = codon_pool_evo2_to_aa(h).float()
            evo2_tags.append(k.rsplit("__",1)[0])
    evo2_tags = sorted(evo2_tags)
    
    E = torch.stack([evo2_feats[t].to(device) for t in evo2_tags])
    Ec = E - E.mean(dim=1, keepdim=True)
    Eg = torch.bmm(Ec, Ec.transpose(1,2))
    Ev = Eg.reshape(len(evo2_tags), -1)
    En = Eg.norm(dim=(1,2))
    
    CKA_mat = (Ev @ Rv.T) / (En.unsqueeze(1) * Rn.unsqueeze(0)).clamp_min(1e-8)
    max_val, idx = CKA_mat.max(dim=0)  # best EVO2 per RF3 head, or overall max
    global_max = CKA_mat.max().item()
    gi, gj = CKA_mat.argmax().item() // len(rf3_tags), CKA_mat.argmax().item() % len(rf3_tags)
    
    # Also: best CKA per EVO2 head, averaged
    mean_cka = CKA_mat.mean().item()
    
    results.append({
        'mutant': mid,
        'max_cka': global_max,
        'best_evo2': evo2_tags[gi],
        'best_rf3': rf3_tags[gj],
        'mean_cka': mean_cka,
        'time': time.time() - t1,
    })
    print(f"  mutant_{mid:04d}: max={global_max:.4f}  best_E={evo2_tags[gi]}  best_R={rf3_tags[gj]}  mean={mean_cka:.4f}  ({time.time()-t1:.0f}s)")

print(f"\n=== CROSS-MUTANT SUMMARY ({len(mutant_ids)} mutants) ===")
max_vals = [r['max_cka'] for r in results]
print(f"  Max CKA:      {np.mean(max_vals[1:]):.4f} ± {np.std(max_vals[1:]):.4f}  (WT={max_vals[0]:.4f})")
best_es = [r['best_evo2'] for r in results]
best_rs = [r['best_rf3'] for r in results]
from collections import Counter
print(f"  Best EVO2:    {Counter(best_es).most_common(3)}")
print(f"  Best RF3:     {Counter(best_rs).most_common(3)}")
print(f"  Idential pair: {sum(1 for r in results if r['best_evo2']==best_es[0] and r['best_rf3']==best_rs[0])}/{len(results)}")

# Top-5 EVO2 heads by frequency as best
evo2_freq = Counter()
rf3_freq = Counter()
for r in results:
    evo2_freq[r['best_evo2']] += 1
    rf3_freq[r['best_rf3']] += 1
print(f"\n  EVO2 heads appearing as best in ≥2 mutants:")
for h, c in evo2_freq.most_common():
    if c >= 2: print(f"    {h}: {c}/{len(results)}")
print(f"  RF3 heads appearing as best in ≥2 mutants:")
for h, c in rf3_freq.most_common():
    if c >= 2: print(f"    {h}: {c}/{len(results)}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
