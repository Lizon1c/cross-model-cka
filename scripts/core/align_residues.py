#!/usr/bin/env python3
"""
Physical residue alignment between RF3 and EVO2 attention heads.

Strategy: assign the same position_id to physically identical residues:
  - Protein:  position 0-421 (Cas12f1, 422 aa per chain; UniProt A0A2U3D0N8)
  - sgRNA:    position 1000-1221 (chain C in RF3, tracrRNA+repeat+spacer in EVO2)
  - Target/non-target DNA: EXCLUDED (foreign to CRISPR locus)
  - Flanks/PAM: EXCLUDED (not in 5-chain complex)

Output: position_id vectors for RF3 (1145 tokens) and EVO2 (1857 bp),
        plus a function to extract aligned activations from head outputs.
"""
import sys; sys.path.insert(0,"/mnt/j/conda_envs/foundry")
exec(open('/mnt/j/conda_envs/foundry/run_attn_extraction.py').read().split("if __name__")[0])

import torch
import numpy as np

SENTINEL = -1  # ID for unaligned positions

# ══════════════════════════════════════════════
# RF3 position_id map
# ══════════════════════════════════════════════
# RF3 token layout: 1145 = 3 special + 422(A) + 422(B) + 222(C) + 38(D) + 38(E)
# 5-chain complex: Chain A=AsCas12f1(422aa), B=AsCas12f1(422aa), C=sgRNA(222nt),
#   D=target DNA(38nt), E=non-target DNA(38nt)
RF3_N_TOKENS = 1145

def build_rf3_position_ids():
    """Returns np.array of shape (RF3_N_TOKENS,) with physical residue IDs."""
    pids = np.full(RF3_N_TOKENS, SENTINEL, dtype=np.int32)
    
    # Chain layout: 3 special tokens + Chains A/B/C/D/E packed consecutively
    # Verified against head_registry.json and RF3 5-chain complex structure
    
    N_SPECIAL = RF3_N_TOKENS - (422 + 422 + 222 + 38 + 38)  # = 3
    offset_a = N_SPECIAL
    offset_b = offset_a + 422
    offset_c = offset_b + 422
    
    # Chain A/B: protein residues 0-421 (UniProt A0A2U3D0N8, 422 aa)
    L_aa_rf3 = 422
    pids[offset_a:offset_a + L_aa_rf3] = np.arange(L_aa_rf3)
    pids[offset_b:offset_b + L_aa_rf3] = np.arange(L_aa_rf3)
    # Chain C: sgRNA positions 1000-1221 (offset avoids collision with protein 0-421)
    pids[offset_c:offset_c+222] = np.arange(1000, 1222, dtype=np.int32)
    # Chain D, E: EXCLUDED (foreign target/non-target DNA, SENTINEL)
    
    return pids

# ══════════════════════════════════════════════
# EVO2 position_id map  
# ══════════════════════════════════════════════
WT = WT_PROTEIN_SEQ
cds = reverse_translate_ascas12f(WT)
cds_start = len(ASCAS12F_LOCUS_LEFT_FLANK)  # 168
evo2_len = len(build_ascas12f_minimal_crispr_locus(WT))

def build_evo2_position_ids():
    """Returns np.array of shape (evo2_len,) with physical residue IDs."""
    pids = np.full(evo2_len, SENTINEL, dtype=np.int32)
    
    # CDS: each codon (3 bp) → 1 protein residue
    # EVO2 CDS = 1269 bp = 423 codons = 422 aa + stop (TAA)
    n_aa = len(cds) // 3  # 423 codons
    for aa_idx in range(n_aa - 1):  # 0..421, exclude stop codon
        codon_start = cds_start + 3 * aa_idx
        pids[codon_start:codon_start+3] = aa_idx
    # Stop codon (TAA): no pid — not an amino acid
    
    # tracrRNA: sgRNA positions 1000-1171 (172 nt)
    tracr_start = cds_start + len(cds)
    pids[tracr_start:tracr_start+172] = np.arange(1000, 1172)
    
    # CRISPR repeat (in array): sgRNA positions 1176-1204
    repeat_start = tracr_start + len(ASCAS12F_TRACRRNA_DNA) + len(ASCAS12F_LOCUS_RIGHT_FLANK)
    pids[repeat_start:repeat_start+29] = np.arange(1176, 1205)
    
    # Spacer: sgRNA positions 1205-1224
    spacer_start = repeat_start + 29
    pids[spacer_start:spacer_start+20] = np.arange(1205, 1225)
    
    # PAM, target DNA, non-target DNA, flanks: EXCLUDED (SENTINEL)
    
    return pids

# ══════════════════════════════════════════════
# Build and verify
# ══════════════════════════════════════════════
rf3_pids = build_rf3_position_ids()
evo2_pids = build_evo2_position_ids()

print(f"RF3:  {len(rf3_pids)} tokens, {np.sum(rf3_pids >= 0)} aligned, {np.sum(rf3_pids < 0)} excluded")
print(f"EVO2: {len(evo2_pids)} bp,     {np.sum(evo2_pids >= 0)} aligned, {np.sum(evo2_pids < 0)} excluded")

# Verify shared position IDs
shared = set(rf3_pids[rf3_pids >= 0]) & set(evo2_pids[evo2_pids >= 0])
print(f"\nShared position IDs: {len(shared)}")
print(f"  Protein residues: 0-421 (422) ✓" if 421 in shared else "  Missing!")
print(f"  sgRNA positions:  1000-1221 (222)")
print(f"  Total shared:     {len(shared)} (expected: 422+222=644; linker 4 nt excluded)")

# Show mapping examples
print(f"\n=== PROTEIN RESIDUE 0 (Met) ===")
rf3_idx_a0 = np.where(rf3_pids == 0)[0]
evo2_idx_0 = np.where(evo2_pids == 0)[0]
print(f"  RF3  tokens: {list(rf3_idx_a0)}  (Chains A and B, Chain A used for analysis)")
print(f"  EVO2 bp:     {list(evo2_idx_0)}  (codon ATG)")

print(f"\n=== sgRNA TRACRRNA POS 0 (RF3 pid=1000) ===")
sgRNA_in_rf3 = np.where(rf3_pids >= 1000)[0]
print(f"  RF3 sgRNA token range: [{sgRNA_in_rf3[0]}..{sgRNA_in_rf3[-1]}] ({len(sgRNA_in_rf3)} tokens)")

# ══════════════════════════════════════════════
# Alignment utility
# ══════════════════════════════════════════════
def extract_aligned_activations(rf3_head_tensor, evo2_head_tensor, 
                                 rf3_pids=rf3_pids, evo2_pids=evo2_pids):
    """Extract activations at shared physical positions.
    
    Args:
        rf3_head_tensor:  (I, C) where I=1145 tokens
        evo2_head_tensor: (S, C) where S=1857 bp
        
    Returns:
        rf3_aligned:  (N_shared, C) — RF3 activations at shared positions
        evo2_aligned: (N_shared, C) — EVO2 activations at shared positions
        shared_pids:  (N_shared,) — corresponding position IDs
    
    Note: For protein residues, uses RF3 Chain A (first match).
    For sgRNA, uses the only matching chain C position.
    """
    shared_pids = sorted(set(rf3_pids[rf3_pids >= 0]) & set(evo2_pids[evo2_pids >= 0]))
    
    rf3_aligned = []
    evo2_aligned = []
    for pid in shared_pids:
        rf3_indices = np.where(rf3_pids == pid)[0]
        evo2_indices = np.where(evo2_pids == pid)[0]
        
        # RF3: take mean across chains A/B (homodimer)
        rf3_vec = rf3_head_tensor[rf3_indices].mean(dim=0)  # (C,)
        # EVO2: take mean across codon positions (3 bp/aa) or single nt
        evo2_vec = evo2_head_tensor[evo2_indices].mean(dim=0)  # (C,)
        
        rf3_aligned.append(rf3_vec)
        evo2_aligned.append(evo2_vec)
    
    return (torch.stack(rf3_aligned), torch.stack(evo2_aligned), 
            torch.tensor(shared_pids, dtype=torch.int32))

print(f"\n{'='*60}")
print(f"Alignment map ready.")
print(f"  extract_aligned_activations(rf3_head, evo2_head)")
print(f"  → ({len(shared)} shared positions, C channels)")
