"""Systematic reading-frame shift test: offset 0..1856, see where CKA collapses."""
import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry/DMS_Project")
sys.path.insert(0,"/mnt/j/conda_envs/foundry")
import torch, numpy as np, time
from align_residues import build_rf3_position_ids, build_evo2_position_ids
from evo2_to_rf3_align_v32 import *

device = torch.device("cuda:0")
evo2_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_data = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/rf3/mutant_0000.pt", map_location='cpu', weights_only=False)
rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()

# Best pair features on GPU
PE = "evo2__mod0004__h28__mutant_0000"
PR = "rf3__mod0076__h00__mutant_0000"
eh = evo2_data[PE].squeeze(0)  # (1857, 128)
rh = rf3_data[PR][0]           # (1145, C)

# RF3 protein features (fixed)
ra = extract_rf3_chain_a(rh).float().to(device)  # (422, C)

# EVO2 raw features on GPU
eh_gpu = eh.float().to(device)  # (1857, 128)

def codon_pool_shifted(evo2_head, shift):
    """Codon pool with circular shift applied BEFORE pooling."""
    S, C = evo2_head.shape
    # Circular shift the EVO2 activations
    shifted = torch.cat([evo2_head[shift:], evo2_head[:shift]])  # (1857, 128)
    # Now pool as before: use original position IDs
    L_aa = 422
    pids_t = torch.as_tensor(evo2_pids, dtype=torch.long, device=device)
    result = torch.zeros(L_aa, C, dtype=evo2_head.dtype, device=device)
    for aa_idx in range(L_aa):
        idx = (pids_t == aa_idx).nonzero(as_tuple=True)[0]
        if len(idx) >= 3:
            result[aa_idx] = shifted[idx[:3]].mean(dim=0)
        elif len(idx) > 0:
            result[aa_idx] = shifted[idx].mean(dim=0)
    return result

# Test shifts from 0 to 1856 (step=1 for fine, step=10 for coarse)
print("Testing codon reading frame shifts...")
shifts = list(range(0, 20))  # fine: 0-19
shifts += list(range(20, 200, 10))  # medium
shifts += list(range(200, 1857, 50))  # coarse
shifts = sorted(set(shifts))

# Also random permutations
N_RAND = 100
t0 = time.time()
results = {}

for s in shifts:
    ea = codon_pool_shifted(eh_gpu, s)
    cka = linear_cka(ea.float(), ra).item()
    results[s] = cka
    if s % 50 == 0 or s < 20:
        print(f"  shift={s:4d}  CKA={cka:.4f}")

# Random baselines
rand_ckas = []
for _ in range(N_RAND):
    perm = torch.randperm(1857, device=device)
    ea = codon_pool_shifted(eh_gpu[perm], 0)  # shuffle nucleotide order, then pool
    cka = linear_cka(ea.float(), ra).item()
    rand_ckas.append(cka)

print(f"\nRandom permutation baseline (n={N_RAND}): {np.mean(rand_ckas):.4f} ± {np.std(rand_ckas):.4f}")
print(f"Frame 0 CKA: {results[0]:.4f}")
print(f"Frame 1 CKA: {results[1]:.4f}")
print(f"Frame 2 CKA: {results[2]:.4f}")

# Find where CKA drops below threshold
threshold = np.mean(rand_ckas) + 2 * np.std(rand_ckas)
print(f"\nNull threshold (mean+2σ): {threshold:.4f}")
print(f"CKA at frame 3: {results.get(3, 'N/A')}")

# Window analysis: when does CKA enter the null zone?
print(f"\n=== TRANSITION ZONES ===")
for s in sorted(results.keys()):
    if results[s] < threshold:
        print(f"  CKA drops below null at shift={s}: CKA={results[s]:.4f}")
        break

# Also test: what if we just permute within each codon (same amino acid, different codon)?
print(f"\n=== PERMUTE WITHIN CODONS (same aa, shuffled codon order) ===")
ea = codon_pool_shifted(eh_gpu, 0)
ea_shuf = ea.clone()
for aa_idx in range(422):
    idx = (torch.as_tensor(evo2_pids, dtype=torch.long, device=device) == aa_idx).nonzero(as_tuple=True)[0]
    if len(idx) >= 3:
        perm_codon = torch.randperm(len(idx[:3]), device=device)
        shifted = eh_gpu[idx[:3]][perm_codon]
        ea_shuf[aa_idx] = shifted.mean(dim=0)
cka_within = linear_cka(ea_shuf.float(), ra).item()
print(f"  Within-codon shuffle CKA: {cka_within:.4f}")

torch.save({'shifts': shifts, 'results': results, 'rand_ckas': rand_ckas, 'threshold': threshold},
           "/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2_full10/frame_shift_scan.pt")
print(f"\nSaved. Time: {time.time()-t0:.0f}s")
