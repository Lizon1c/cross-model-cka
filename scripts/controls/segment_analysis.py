"""Segment-level analysis with position-controlled baselines."""
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
L = 422

# ── Precompute features ──
# RF3 Chain A and B
rf3_A = {}; rf3_B = {}; rf3_tags = []
for k, v in rf3_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() != 3 or v.shape[1] != 1145: continue
    tag = k.rsplit("__",1)[0]
    h = v[0].float()
    rf3_A[tag] = extract_rf3_chain_a(h)
    # Chain B
    pids_t = torch.as_tensor(rf3_pids, dtype=torch.long)
    rb = torch.zeros(422, h.shape[1], dtype=h.dtype)
    for aa in range(422):
        matches = (pids_t == aa).nonzero(as_tuple=True)[0]
        if len(matches) >= 2: rb[aa] = h[matches[1]]
        elif len(matches) >= 1: rb[aa] = h[matches[0]]
    rf3_B[tag] = rb
    rf3_tags.append(tag)
rf3_tags = sorted(rf3_tags)

# Top-20 RF3 heads by raw max with any EVO2 head (approximate via top sinusoid responders)
# Actually just use the full CKA matrix
cka_mat = torch.load("/mnt/j/conda_envs/foundry/DMS_Project/output_heads/evo2_full10/full_cka_matrix.pt", map_location='cpu', weights_only=False)
etags = cka_mat['evo2_tags']
rtags_all = cka_mat['rf3_tags']
Mp = cka_mat['M_prot']

top20_evo2_idx = Mp.max(dim=1).values.topk(20).indices.tolist()
top20_rf3_idx = Mp.max(dim=0).values.topk(20).indices.tolist()
top20_evo2 = [etags[i] for i in top20_evo2_idx]
top20_rf3 = [rtags_all[i] for i in top20_rf3_idx]

# EVO2 features
evo2_feats = {}
for k, v in evo2_data.items():
    if not isinstance(v, torch.Tensor) or v.dim() < 2: continue
    h = v.squeeze(0)
    if h.dim() == 2 and h.shape[0] == 1857:
        evo2_feats[k.rsplit("__",1)[0]] = codon_pool_evo2_to_aa(h).float()

# Sinusoid position encoding
t_pos = torch.linspace(0, 1, L, device=device)
pe = torch.zeros(L, 128, device=device)
for i in range(64):
    f = 10000 ** (2*i/128)
    pe[:,2*i] = torch.sin(t_pos * f); pe[:,2*i+1] = torch.cos(t_pos * f)

# ── Segment scan ──
window_sizes = [60, 80, 100, 120]
stride = 10
n_top_e = min(10, len(top20_evo2))
n_top_r = min(20, len(top20_rf3))

print(f"Segment scan: {n_top_e} EVO2 × {n_top_r} RF3, windows={window_sizes}, stride={stride}")
print(f"Total: {sum((L-w)//stride+1 for w in window_sizes)} windows × {n_top_e}×{n_top_r} pairs")

def segment_cka(e_vec, r_vec, start, width):
    """CKA on a contiguous segment of residue features."""
    e_seg = e_vec[start:start+width]
    r_seg = r_vec[start:start+width]
    ec = e_seg - e_seg.mean(dim=0)
    rc = r_seg - r_seg.mean(dim=0)
    K = ec @ ec.T; Lg = rc @ rc.T
    return (K * Lg).sum() / (K.norm() * Lg.norm()).clamp_min(1e-8)

def segment_excess(e_real, r_real, start, width):
    """Raw CKA minus position-baseline CKA on segment."""
    raw = segment_cka(e_real, r_real, start, width)
    # Position baseline: use sinusoid against the same RF3 segment
    d_r = r_real.shape[1]
    pos = segment_cka(pe[:, :d_r], r_real, start, width)
    # Also cubic residual CKA
    basis = torch.stack([t_pos**d for d in range(4)], dim=1)
    e_d = e_real[:, :d_r]
    be = torch.linalg.lstsq(basis, e_d).solution
    br = torch.linalg.lstsq(basis, r_real).solution
    er = e_d - basis @ be; rr = r_real - basis @ br
    res = segment_cka(er, rr, start, width)
    return raw.item(), pos.item(), res.item()

# ── TABLE 2: Domain-level with excess ──
domains = [("REC",0,126),("WED",126,211),("Linker",211,220),("RuvC-I",220,370),("TNB",370,399),("RuvC-II",399,420)]
print(f"\n{'='*100}")
print(f"TABLE 2: Domain-level analysis (position-controlled, top heads)")
print(f"{'Domain':<12s} {'width':>6s} {'raw_max':>10s} {'pos_base':>10s} {'excess':>10s} {'resid':>10s} {'top_EVO2':>25s} {'top_RF3':>25s}")
print(f"{'-'*100}")

for dname, s, e in domains:
    w = e - s
    best_raw = -1; best_pos = 0; best_res = 0; best_e = ""; best_r = ""
    for etag in top20_evo2[:n_top_e]:
        eh = evo2_feats[etag].to(device)
        for rtag in top20_rf3[:n_top_r]:
            rh = rf3_A[rtag].to(device)
            raw, pos, res = segment_excess(eh, rh, s, w)
            if raw > best_raw:
                best_raw = raw; best_pos = pos; best_res = res; best_e = etag; best_r = rtag
    excess = best_raw - best_pos
    print(f"{dname:<12s} {w:>6d} {best_raw:>10.4f} {best_pos:>10.4f} {excess:>10.4f} {best_res:>10.4f} {best_e:>25s} {best_r:>25s}")

# ── TABLE 3: Top segment head-pair recurrence (window=100, top results) ──
print(f"\n{'='*100}")
print(f"TABLE 3: Top-20 segment results (window=100, stride=10)")
print(f"{'start':>6s} {'end':>6s} {'raw':>8s} {'excess':>8s} {'resid':>8s} {'EVO2':>25s} {'RF3':>25s}")
print(f"{'-'*100}")

w = 100
all_segs = []
for start in range(0, L - w + 1, stride):
    for etag in top20_evo2[:n_top_e]:
        eh = evo2_feats[etag].to(device)
        for rtag in top20_rf3[:n_top_r]:
            rh = rf3_A[rtag].to(device)
            raw, pos, res = segment_excess(eh, rh, start, w)
            all_segs.append((start, raw, raw-pos, res, etag, rtag))

all_segs.sort(key=lambda x: -x[1])  # sort by raw CKA
for start, raw, excess, res, etag, rtag in all_segs[:20]:
    print(f"{start:>6d} {start+w:>6d} {raw:>8.4f} {excess:>8.4f} {res:>8.4f} {etag:>25s} {rtag:>25s}")

# ── TABLE 4: A/B chain segment comparison (fixed best pair) ──
print(f"\n{'='*100}")
print(f"TABLE 4: Chain A/B segment comparison (window=100, best pair)")
print(f"{'start':>6s} {'raw_A':>8s} {'raw_B':>8s} {'excess_A':>8s} {'excess_B':>8s} {'delta_excess':>12s}")
print(f"{'-'*100}")

eh = evo2_feats["evo2__mod0004__h28"].to(device)
best_rf3 = "rf3__mod0076__h00"
for start in range(0, L - 100 + 1, stride):
    ra_a, pa_a, res_a = segment_excess(eh, rf3_A[best_rf3].to(device), start, 100)
    ra_b, pa_b, res_b = segment_excess(eh, rf3_B[best_rf3].to(device), start, 100)
    exc_a = ra_a - pa_a; exc_b = ra_b - pa_b
    print(f"{start:>6d} {ra_a:>8.4f} {ra_b:>8.4f} {exc_a:>8.4f} {exc_b:>8.4f} {exc_a-exc_b:>10.4f}")

# ── TABLE 5: Position-only vs EVO2 real across all segments ──
print(f"\n{'='*100}")
print(f"TABLE 5: Excess distribution across all windows (all sizes)")
print(f"{'window':>8s} {'mean_excess':>12s} {'std_excess':>12s} {'frac>0.05':>10s} {'frac>0.10':>10s} {'max_excess':>12s}")
print(f"{'-'*100}")

for w in window_sizes:
    excesses = []
    for start in range(0, L - w + 1, stride):
        for etag in top20_evo2[:n_top_e]:
            eh = evo2_feats[etag].to(device)
            for rtag in top20_rf3[:n_top_r]:
                rh = rf3_A[rtag].to(device)
                raw, pos, res = segment_excess(eh, rh, start, w)
                excesses.append(raw - pos)
    ex = torch.tensor(excesses)
    print(f"{w:>8d} {ex.mean():>12.6f} {ex.std():>12.6f} {(ex>0.05).float().mean():>10.4f} {(ex>0.10).float().mean():>10.4f} {ex.max():>12.6f}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
