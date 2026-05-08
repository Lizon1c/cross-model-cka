import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np, time
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

t0 = time.time()
evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()
print(f"Loaded in {time.time()-t0:.0f}s")

# ── Precompute protein features for ALL heads ──
print("Precomputing protein features...")
t0 = time.time()

# EVO2: for each of 160 heads, codon_pool → (422, 128)
evo2_prot = {}  # tag → (422, 128)
for k, v in evo2_data.items():
    if isinstance(v, torch.Tensor) and v.dim() >= 2:
        h = v.squeeze(0)
        if h.dim() == 2 and h.shape[0] == 1857:
            evo2_prot[k.rsplit("__",1)[0]] = codon_pool_evo2_to_aa(h).float()

# RF3: for each token-level head, extract_rf3_chain_a
rf3_prot = {}  # tag → (422, C)
rf3_sg = {}    # tag → (218, C)
for k, v in rf3_data.items():
    if not isinstance(v, torch.Tensor): continue
    if v.dim() == 3 and v.shape[1] == 1145:
        h = v[0]
        tag = k.rsplit("__",1)[0]
        rf3_prot[tag] = extract_rf3_chain_a(h).float()
        # sgRNA
        sp = sorted(set(rf3_pids[rf3_pids>=1000]) & set(evo2_pids[evo2_pids>=1000]))
        d = h.shape[-1]
        R = torch.stack([h[np.where(rf3_pids==p)[0][0], :].float() for p in sp])
        rf3_sg[tag] = R

# sgRNA for EVO2
evo2_sg = {}
for k, v in evo2_data.items():
    if isinstance(v, torch.Tensor) and v.dim() >= 2:
        h = v.squeeze(0)
        if h.dim() == 2 and h.shape[0] == 1857:
            sp = sorted(set(rf3_pids[rf3_pids>=1000]) & set(evo2_pids[evo2_pids>=1000]))
            E = torch.stack([h[np.where(evo2_pids==p)[0], :].float().mean(dim=0) for p in sp])
            evo2_sg[k.rsplit("__",1)[0]] = E

print(f"Precomputed in {time.time()-t0:.0f}s: {len(evo2_prot)} EVO2 + {len(rf3_prot)} RF3 heads")

# ── Batch CKA via GPU ──
def batch_cka(X_list, Y_list):
    """Compute CKA for all X×Y pairs. Returns matrix (len(X), len(Y))."""
    M = torch.zeros(len(X_list), len(Y_list))
    # Precompute X @ X^T and norms for each X
    X_gram = []; X_norm = []
    for x in X_list:
        xc = x - x.mean(dim=0)
        K = xc @ xc.T
        X_gram.append(K)
        X_norm.append(K.norm())
    for j, y in enumerate(Y_list):
        yc = y - y.mean(dim=0)
        L = yc @ yc.T
        L_norm = L.norm()
        for i in range(len(X_list)):
            M[i, j] = (X_gram[i] * L).sum() / (X_norm[i] * L_norm).clamp_min(1e-8)
    return M

print("\nComputing CKA matrix...")
t0 = time.time()

evo2_tags = sorted(evo2_prot.keys())
rf3_tags = sorted(rf3_prot.keys())
X_vals = [evo2_prot[t] for t in evo2_tags]
Y_vals = [rf3_prot[t] for t in rf3_tags]
M_prot = batch_cka(X_vals, Y_vals)

# sgRNA
X_sg = [evo2_sg[t] for t in evo2_tags]
Y_sg = [rf3_sg[t] for t in rf3_tags]
M_sg = batch_cka(X_sg, Y_sg)

# Best for each
best_i_p, best_j_p = np.unravel_index(M_prot.numpy().argmax(), M_prot.shape)
best_i_s, best_j_s = np.unravel_index(M_sg.numpy().argmax(), M_sg.shape)

print(f"Computed in {time.time()-t0:.0f}s")
print(f"\n=== BEST PAIRS ===")
print(f"Protein: {evo2_tags[best_i_p]} ↔ {rf3_tags[best_j_p]}  CKA={M_prot[best_i_p, best_j_p]:.4f}")
print(f"sgRNA:   {evo2_tags[best_i_s]} ↔ {rf3_tags[best_j_s]}  CKA={M_sg[best_i_s, best_j_s]:.4f}")

# Top-5 each
print(f"\nTop-5 protein pairs:")
pi = M_prot.flatten().argsort(descending=True)[:5]
for idx in pi:
    i, j = idx // M_prot.shape[1], idx % M_prot.shape[1]
    print(f"  {evo2_tags[i]:20s} ↔ {rf3_tags[j]:24s}  CKA={M_prot[i,j]:.4f}")

print(f"\nTop-5 sgRNA pairs:")
si = M_sg.flatten().argsort(descending=True)[:5]
for idx in si:
    i, j = idx // M_sg.shape[1], idx % M_sg.shape[1]
    print(f"  {evo2_tags[i]:20s} ↔ {rf3_tags[j]:24s}  CKA={M_sg[i,j]:.4f}")

# Save
torch.save({'evo2_tags': evo2_tags, 'rf3_tags': rf3_tags, 'M_prot': M_prot, 'M_sg': M_sg},
           "/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2_full10/full_cka_matrix.pt")
print(f"\nSaved full CKA matrix")
