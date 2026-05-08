"""Max-null v6: FULL 160E × 1216R on GPU. No pruning."""
import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, time
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

device = torch.device("cuda:0")
t0 = time.time()

evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)

# ── Build RF3 features on GPU: (N_RF3, 422, C_max) ──
rf3_feats = []; rf3_tags_list = []
for k, v in rf3_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() != 3 or v.shape[1] != 1145: continue
    rf3_feats.append(extract_rf3_chain_a(v[0]).float())
    rf3_tags_list.append(k.rsplit("__",1)[0])

N_R = len(rf3_feats)
C_max = max(f.shape[1] for f in rf3_feats)
R_pad = torch.zeros(N_R, 422, C_max, device=device)
for i, f in enumerate(rf3_feats):
    R_pad[i, :, :f.shape[1]] = f.to(device)
# Center
R_c = R_pad - R_pad.mean(dim=1, keepdim=True)
# Batch gram matrices: (N_R, 422, 422)
R_gram = torch.bmm(R_c, R_c.transpose(1,2))
# Flatten + norms: (N_R, 422*422)
R_vec = R_gram.reshape(N_R, -1)
R_norm = R_gram.norm(dim=(1,2))  # (N_R,)
print(f"RF3: {N_R} heads on {device} in {time.time()-t0:.0f}s  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Build EVO2 features on GPU ──
evo2_feats = []; evo2_tags_list = []
for k, v in evo2_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() < 2: continue
    h = v.squeeze(0)
    if h.dim() == 2 and h.shape[0] == 1857:
        evo2_feats.append(codon_pool_evo2_to_aa(h).float())
        evo2_tags_list.append(k.rsplit("__",1)[0])

N_E = len(evo2_feats)
E = torch.stack(evo2_feats).to(device)  # (160, 422, 128)
E_c = E - E.mean(dim=1, keepdim=True)
E_gram = torch.bmm(E_c, E_c.transpose(1,2))  # (160, 422, 422)
E_vec = E_gram.reshape(N_E, -1)  # (160, 422*422)
E_norm = E_gram.norm(dim=(1,2))  # (160,)
print(f"EVO2: {N_E} heads on {device} in {time.time()-t0:.0f}s  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Observed max CKA ──
# CKA matrix: (N_E, N_R) = (E_vec @ R_vec.T) / outer(E_norm, R_norm)
numer = E_vec @ R_vec.T  # (160, 1216)
denom = E_norm.unsqueeze(1) * R_norm.unsqueeze(0)  # (160, 1216)
CKA_mat = numer / denom.clamp_min(1e-8)
obs_max = CKA_mat.max().item()
best_i, best_j = CKA_mat.argmax().item() // N_R, CKA_mat.argmax().item() % N_R
print(f"Observed max: {obs_max:.6f}  ({evo2_tags_list[best_i]} ↔ {rf3_tags_list[best_j]})")
print(f"Time so far: {time.time()-t0:.0f}s")

# ── Max-null: shuffle EVO2 features, recompute gram, compute max CKA ──
N_PERM = 200
null_max = torch.zeros(N_PERM, device=device)
t1 = time.time()

for p in range(N_PERM):
    perm = torch.randperm(422, device=device)
    E_c_p = E_c[:, perm, :]  # (160, 422, 128)
    E_gram_p = torch.bmm(E_c_p, E_c_p.transpose(1,2))  # (160, 422, 422)
    E_vec_p = E_gram_p.reshape(N_E, -1)
    E_norm_p = E_gram_p.norm(dim=(1,2))
    
    numer_p = E_vec_p @ R_vec.T
    denom_p = E_norm_p.unsqueeze(1) * R_norm.unsqueeze(0)
    CKA_p = numer_p / denom_p.clamp_min(1e-8)
    null_max[p] = CKA_p.max()
    
    if (p+1) % 20 == 0:
        print(f"  {p+1}/{N_PERM}  max_so_far={null_max[:p+1].max().item():.4f}  elapsed={time.time()-t1:.0f}s")

null_max_cpu = null_max.cpu()
p_val = (int((null_max_cpu >= obs_max).sum()) + 1) / (N_PERM + 1)
print(f"\n=== MAX-NULL v6 (FULL {N_E}×{N_R} on GPU, {N_PERM} perms) ===")
print(f"Observed max:  {obs_max:.6f}")
print(f"Null max:      {null_max_cpu.mean():.6f} ± {null_max_cpu.std():.6f}")
print(f"Null range:    [{null_max_cpu.min():.6f}, {null_max_cpu.max():.6f}]")
print(f"p = {p_val:.6f}  ({int((null_max_cpu>=obs_max).sum())}/{N_PERM})")
print(f"Time: {time.time()-t1:.0f}s")

torch.save({'obs_max': obs_max, 'null_max': null_max_cpu, 'n_perm': N_PERM,
            'evo2_best': evo2_tags_list[best_i], 'rf3_best': rf3_tags_list[best_j]},
           "/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2_full10/max_null.pt")
