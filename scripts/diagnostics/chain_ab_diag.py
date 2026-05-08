"""Chain A vs Chain B: asymmetry in RF3 homodimer alignment with EVO2."""
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

# ── Extract Chain B: second match for each protein pid ──
def extract_rf3_chain_b(rf3_head):
    I, C = rf3_head.shape
    pids_t = torch.as_tensor(rf3_pids, dtype=torch.long)
    L_aa = 422
    result = torch.zeros(L_aa, C, dtype=rf3_head.dtype)
    for aa_idx in range(L_aa):
        matches = (pids_t == aa_idx).nonzero(as_tuple=True)[0]
        if len(matches) >= 2:
            result[aa_idx] = rf3_head[matches[1]]  # second match = chain B
        elif len(matches) == 1:
            result[aa_idx] = rf3_head[matches[0]]  # fallback (shouldn't happen)
    return result

# ── Build features: Chain A, Chain B, A/B mean ──
rf3_A, rf3_B, rf3_AB, rf3_tags = [], [], [], []
for k, v in rf3_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() != 3 or v.shape[1] != 1145: continue
    tag = k.rsplit("__",1)[0]
    h = v[0].float()
    rf3_A.append(extract_rf3_chain_a(h))
    rf3_B.append(extract_rf3_chain_b(h))
    rf3_AB.append((extract_rf3_chain_a(h) + extract_rf3_chain_b(h)) / 2)
    rf3_tags.append(tag)

N_R = len(rf3_tags)
C_max = max(max(f.shape[1] for f in rf3_A), max(f.shape[1] for f in rf3_B))

def to_gpu_padded(feats):
    R = torch.zeros(N_R, 422, C_max, device=device)
    for i, f in enumerate(feats):
        R[i, :, :f.shape[1]] = f.to(device)
    Rc = R - R.mean(dim=1, keepdim=True)
    G = torch.bmm(Rc, Rc.transpose(1,2))
    return G.reshape(N_R, -1), G.norm(dim=(1,2))

R_vec_A, R_norm_A = to_gpu_padded(rf3_A)
R_vec_B, R_norm_B = to_gpu_padded(rf3_B)
R_vec_AB, R_norm_AB = to_gpu_padded(rf3_AB)

# EVO2
evo2_feats, evo2_tags = [], []
for k, v in evo2_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() < 2: continue
    h = v.squeeze(0)
    if h.dim() == 2 and h.shape[0] == 1857:
        evo2_feats.append(codon_pool_evo2_to_aa(h).float())
        evo2_tags.append(k.rsplit("__",1)[0])

N_E = len(evo2_feats)
E = torch.stack(evo2_feats).to(device)
Ec = E - E.mean(dim=1, keepdim=True)
Eg = torch.bmm(Ec, Ec.transpose(1,2))
E_vec = Eg.reshape(N_E, -1)
E_norm = Eg.norm(dim=(1,2))

def compute_cka_mat(E_vec, E_norm, R_vec, R_norm):
    n = E_vec @ R_vec.T
    d = E_norm.unsqueeze(1) * R_norm.unsqueeze(0)
    return n / d.clamp_min(1e-8)

CKA_A = compute_cka_mat(E_vec, E_norm, R_vec_A, R_norm_A)
CKA_B = compute_cka_mat(E_vec, E_norm, R_vec_B, R_norm_B)
CKA_AB = compute_cka_mat(E_vec, E_norm, R_vec_AB, R_norm_AB)

print(f"Chains built in {time.time()-t0:.0f}s\n")

# ── Top pairs per chain ──
def top_pairs(CKA_mat, k=5):
    vals, idxs = CKA_mat.flatten().topk(k)
    results = []
    for v, idx in zip(vals, idxs):
        i, j = idx.item() // CKA_mat.shape[1], idx.item() % CKA_mat.shape[1]
        results.append((evo2_tags[i], rf3_tags[j], v.item()))
    return results

print("=== TOP-5 PROTEIN PAIRS: Chain A ===")
for r, (et, rt, c) in enumerate(top_pairs(CKA_A)):
    print(f"  #{r+1}: {et} ↔ {rt}  CKA={c:.4f}")

print("\n=== TOP-5 PROTEIN PAIRS: Chain B ===")
for r, (et, rt, c) in enumerate(top_pairs(CKA_B)):
    print(f"  #{r+1}: {et} ↔ {rt}  CKA={c:.4f}")

print("\n=== TOP-5 PROTEIN PAIRS: A/B mean ===")
for r, (et, rt, c) in enumerate(top_pairs(CKA_AB)):
    print(f"  #{r+1}: {et} ↔ {rt}  CKA={c:.4f}")

# ── CKA(A) vs CKA(B) correlation ──
cka_a_flat = CKA_A.flatten().cpu()
cka_b_flat = CKA_B.flatten().cpu()
r_ab = ((cka_a_flat - cka_a_flat.mean()) * (cka_b_flat - cka_b_flat.mean())).sum() / \
       (cka_a_flat.std() * cka_b_flat.std() * len(cka_a_flat)).clamp_min(1e-8)
print(f"\nCKA(A) vs CKA(B) correlation across all {len(cka_a_flat)} pairs: r={r_ab:.4f}")

# ── Per-RF3-head chain preference ──
chain_diff = CKA_A.max(dim=0).values - CKA_B.max(dim=0).values  # (1216,)
top_A_rf3 = chain_diff.topk(5)
top_B_rf3 = (-chain_diff).topk(5)
print(f"\nMost Chain-A-biased RF3 heads (max A CKA - max B CKA):")
for idx, val in zip(top_A_rf3.indices, top_A_rf3.values):
    print(f"  {rf3_tags[idx.item()]:30s}  Δ={val.item():+.4f}  A={CKA_A[:,idx].max().item():.4f}  B={CKA_B[:,idx].max().item():.4f}")

print(f"\nMost Chain-B-biased RF3 heads:")
for idx, val in zip(top_B_rf3.indices, top_B_rf3.values):
    print(f"  {rf3_tags[idx.item()]:30s}  Δ={val.item():+.4f}  A={CKA_A[:,idx].max().item():.4f}  B={CKA_B[:,idx].max().item():.4f}")

# ── BEST head pair: which chain? ──
best_i_a, best_j_a = CKA_A.argmax().item() // N_R, CKA_A.argmax().item() % N_R
best_i_b, best_j_b = CKA_B.argmax().item() // N_R, CKA_B.argmax().item() % N_R
best_i_ab, best_j_ab = CKA_AB.argmax().item() // N_R, CKA_AB.argmax().item() % N_R
print(f"\nOverall max:     A={CKA_A.max().item():.4f}  B={CKA_B.max().item():.4f}  AB={CKA_AB.max().item():.4f}")
print(f"Same RF3 head?   A_best={rf3_tags[best_j_a]}  B_best={rf3_tags[best_j_b]}  AB_best={rf3_tags[best_j_ab]}")
print(f"Same EVO2 head?  A_best={evo2_tags[best_i_a]}  B_best={evo2_tags[best_i_b]}  AB_best={evo2_tags[best_i_ab]}")

# ── Distribution spread ──
print(f"\nCKA distribution per chain:")
print(f"  Chain A:  mean={CKA_A.mean().item():.4f}  std={CKA_A.std().item():.4f}  max={CKA_A.max().item():.4f}")
print(f"  Chain B:  mean={CKA_B.mean().item():.4f}  std={CKA_B.std().item():.4f}  max={CKA_B.max().item():.4f}")
print(f"  A/B mean: mean={CKA_AB.mean().item():.4f}  std={CKA_AB.std().item():.4f}  max={CKA_AB.max().item():.4f}")

# ── Save ──
torch.save({'CKA_A': CKA_A.cpu(), 'CKA_B': CKA_B.cpu(), 'CKA_AB': CKA_AB.cpu(),
            'evo2_tags': evo2_tags, 'rf3_tags': rf3_tags},
           "/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2_full10/chain_ab_comparison.pt")
print(f"\nSaved to chain_ab_comparison.pt")
