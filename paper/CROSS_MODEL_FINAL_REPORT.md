# Cross-Model CKA Alignment Between EVO2 and RFdiffusion3 Reflects Low-Frequency Smoothing Bias Rather Than Structural Convergence

## EVO2 (DNA LM, 7B) ↔ RFdiffusion3 (Structure Diffusion) on AsCas12f1

**Cheng Ding** (丁成) — ShanghaiTech University  
`dingcheng2024@shanghaitech.edu.cn` — [`github.com/Lizon1c/cross-model-cka`](https://github.com/Lizon1c/cross-model-cka)

---

### Abstract

We test whether a DNA language model (EVO2), a protein language model
(FAESM), and a protein-structure diffusion model (RFdiffusion3) produce
intermediate head-level residue representations that converge on the same
physical residues in the AsCas12f1 CRISPR-Cas locus. After correcting the
protein length to 422 aa (UniProt A0A2U3D0N8) and aligning coordinates via
codon pooling (3 nt → 1 aa mean pooling, a dimensionality reduction step),
we scan all 160 EVO2 MHA heads against all 1,216 RF3 token-level heads
(194,560 pairs).

The strongest raw protein head pair achieves full-length CKA = 0.851.
Figure 1B shows the max-null distribution: the null maximum CKA
distribution had mean 0.099, range [0.049, 0.522], with the 99.9th
percentile at 0.512. A full-scan
max-null permutation test (2,000 permutations, p ≈ 0.0005) confirms that
this alignment is not explained by random residue-order shuffling.

However, extensive controls reveal that the raw CKA is dominated by shared
low-frequency coordinate geometry rather than domain-specific functional
alignment. Key findings:

1. **Codon pooling does not reflect reading-frame decoding.** Frame-shift
   robustness arises from local nucleotide-feature correlation plus 3-mer
   averaging, not from codon-boundary specificity.

2. **Position-only baselines match or exceed real EVO2 at segment scales.**
   Across 60–120 aa sliding windows evaluated on the top-scoring head
   pairs, excess CKA (real EVO2 minus position baseline) is strongly negative
   (−0.43 to −0.48), and fewer than 1.1% of windows show excess > 0.05.

3. **Domain-level enrichment does not survive position-aware controls.**
   Apparent TNB and REC enrichment under naive residue-permutation nulls is
   eliminated by contiguous-interval and circular-shift nulls. Within the
   TNB domain, the position-only baseline (CKA = 0.672) substantially
   exceeds real EVO2 (CKA = 0.412; excess = −0.260).

4. **High CKA is head-specific and position-sensitive.** Only 18% of RF3
   heads respond to pure position encoding (CKA > 0.5). The best raw pair's
   RF3 head (mod0076_h00) shows sinusoid CKA = 0.595, confirming it as a
   position-sensitive head. Most head pairs lack this matched position
   geometry and remain at low CKA (mean ≈ 0.15).

5. **Homodimer chain asymmetry is negligible.** Chain A and Chain B CKA
   values correlate at r = 0.975 across all pairs.

**Conclusion:** The EVO2–RF3 raw full-length CKA is real, non-random, and
head-specific, but it reflects a generic low-frequency smoothing bias
induced by deep position mixing in both architectures, rather than
domain-level biological content alignment. At segment and domain
scales, explicit position bases systematically outperform real EVO2
features. Apparent domain enrichment under naive nulls is eliminated
by position-aware controls, and no position-weak head pair achieves
high CKA. The current single-locus WT/mutant analyses do not support
the interpretation that EVO2 captures domain-specific functional
information that exceeds coordinate geometry.

**Data and code:** `scripts/` and `output_heads/evo2_full10/`

---

## 1. Methods

### 1.1 Coordinates and Alignment

**Protein:** 422 aa per chain (UniProt A0A2U3D0N8, full AsCas12f1).

**RF3 tokens:** 1,145 = 3 special + 422 (chain A) + 422 (chain B) + 222
(chain C, sgRNA) + 38 (chain D, target DNA) + 38 (chain E, non-target DNA).
Protein analysis uses RF3 chain A unless noted.

**EVO2 locus:** 1,857 bp = 168 bp (native upstream flank, NCBI
NZ_MPDK01000047.1) + 1,269 bp CDS (422 aa × 3 + TAA stop) + 172 bp
(tracrRNA gene) + 78 bp (native downstream flank) + 78 bp (CRISPR array:
repeat–spacer–repeat) + 92 bp (PAM + target context). Codon table derived
from Sulfoacidibacillus thermotolerans genome (69 CDS, 20,028 codons,
GC 38%).

**Codon pooling:** 3 nt → 1 aa mean pooling maps the 1,857-bp EVO2 locus
to 422 residue-scale coordinates. This is a dimensionality reduction step,
not a reading-frame decoding claim (see §3.2 for frame-shift diagnostics).

**Position IDs:** protein 0–421, sgRNA 1000–1221 (non-overlapping).
640 shared positions (422 protein + 218 sgRNA; 4 nt GAAA linker excluded).

### 1.2 Metrics

- **linear_cka:** centered Gram-matrix CKA, K = X_c·X_c^T, range [0, 1].
- **Per-residue score:** strict Pearson correlation of distance profiles,
  guaranteed [−1, 1].
- **Segment CKA:** CKA computed on a contiguous window of residues,
  with window-local centering (numerically distinct from full-length CKA).
- **Excess CKA:** real EVO2 CKA minus position-only baseline CKA on the
  same segment and head pair.
- **Max-null:** for each permutation, EVO2 residue coordinates are shuffled
  identically across all 160 EVO2 heads (RF3 fixed), the complete 160 × 1,216
  CKA matrix is recomputed, and the maximum is recorded.

### 1.3 Position Baselines

- **Sinusoidal position encoding:** 128-dimensional RoPE-style encoding:
  PE(pos, 2i) = sin(pos·10000^(−2i/128)), PE(pos, 2i+1) = cos(pos·10000^(−2i/128)).
- **Polynomial position trend:** regression of residue features against
  polynomial basis {1, t, t², t³} where t ∈ [0, 1] across the 422 residues.

### 1.4 Head-Activation Extraction

For all models, analyses were performed on intermediate head-level output
activations rather than final model predictions or attention-probability
matrices. A "head" denotes the per-token feature output associated with one
attention submodule after projection into its native head dimension.

**EVO2.** MHA head activations were extracted from 5 deep modules (160 heads).
Nucleotide-level features over the 1,857-bp locus were mapped to 422
protein coordinates by non-overlapping 3-nt mean pooling over the
1,266 coding nucleotides. This pooling is a coordinate projection and
not a claim of reading-frame decoding (§2.2).

**RFdiffusion3.** Token-level activations were extracted from all eligible
heads whose token axis matched the 1,145-token RF3 input. Protein analyses
used chain A residues only, corresponding to token slice `3:3+422` after
the three RF3 special tokens. This yielded 1,216 token-level heads.

**FAESM.** Residue-level head activations were extracted from 33 layers
with 20 heads per layer, giving 660 heads. FAESM outputs have shape
`(1, 424, 64)`; the first and last tokens correspond to BOS/EOS, so
protein residues were taken as slice `[1:423]`.

Head features were retained in their native dimensionalities (EVO2: 128,
FAESM: 64, RF3: 24–256). Linear CKA was computed from centered
residue-by-residue Gram matrices, so the compared heads were not required
to have the same feature dimension.

### 1.5 Head Selection

160 EVO2 MHA heads (5 layers × 32 heads) and 660 FAESM heads (33 layers ×
20 heads) were scanned against 1,216 RF3 token-level heads
(heads with token count ≠ 1,145 excluded). Full CKA matrices were
precomputed on GPU.

---

## 2. Results

### 2.1 Raw Full-Length Scan

The strongest raw protein head pair:

| EVO2 head | RF3 head | RF3 type | CKA |
|-----------|----------|----------|-----|
| mod0004_h28 (L31) | mod0076_h00 | DiffusionTransformer block 18/50 | 0.851 |

**Max-null test (2,000 permutations):** null distribution mean 0.099,
sd 0.063, range [0.049, 0.522]. No permuted scan exceeded the observed
maximum (p = 1/2001 ≈ 0.0005). The raw alignment is not explained by
random residue-order shuffling.

**sgRNA control:** The same head pair shows CKA ≈ 0.000 on tracrRNA,
repeat, and spacer regions. The protein alignment is region-specific.

Most head pairs show low CKA (mean ≈ 0.15 across 194,560 pairs).
Only 9.1% of pairs exceed CKA > 0.5.

### 2.2 Codon Pooling Diagnostics

Systematic frame-shift tests show that CKA is insensitive to codon
boundaries:

| Signal | shift=0 | shift=10 | shift=50 | shift=100 | shift=500 |
|--------|---------|----------|----------|-----------|-----------|
| EVO2 real | 0.853 | 0.857 | 0.877 | 0.865 | 0.390 |
| Codon-mean repeated | 0.853 | 0.858 | 0.878 | 0.866 | 0.390 |
| Row-shuffled EVO2 | 0.010 | 0.011 | 0.011 | 0.010 | 0.004 |
| Gaussian noise | 0.032 | 0.032 | 0.033 | 0.033 | 0.037 |

Three conclusions: (1) the alignment is not a marginal-distribution artifact
(row-shuffle → 0.01); (2) shift robustness arises from local nucleotide
correlation plus 3-mer mean pooling, not codon-boundary decoding; (3) EVO2
features are nearly constant across the three codon positions (Δ < 0.001
when replaced by codon-mean repeats). Codon pooling is a convenient
coordinate projection, not evidence of reading-frame recognition.

### 2.3 Position-Only Baselines Match Real EVO2 at Segment Scales

**Sinusoid baseline:** A 128-d RoPE-style position encoding achieves
CKA = 0.595 against the best pair's RF3 head, and CKA = 0.602 against
the most position-sensitive RF3 head (mod0064_h05).

**Polynomial regression:** Regressing out a linear position trend from
the fixed best pair reduces CKA from 0.852 to 0.022. Cubic trend removal
reduces it to 0.004.

**Residualized full scan:** After removing cubic position trends from all
heads, the full-scan maximum drops to 0.543 and selects entirely different
head pairs (mod0003_h07 ↔ mod0059_h08).

**Segment-level excess CKA:** Across all 60–120 aa sliding windows and
top head pairs, excess CKA (real minus position baseline) is systematically
negative:

| Window | mean excess | std excess | frac > 0.05 | max excess |
|--------|-------------|------------|-------------|------------|
| 60 aa  | −0.482 | 0.270 | 0.7% | 0.224 |
| 80 aa  | −0.467 | 0.226 | 0.6% | 0.183 |
| 100 aa | −0.456 | 0.191 | 0.9% | 0.221 |
| 120 aa | −0.429 | 0.171 | 1.1% | 0.244 |

Fewer than 1.1% of windows exceed the position-only baseline by more than
0.05. Real EVO2 features do not provide a segment-level advantage over
explicit coordinate bases.

### 2.4 Domain-Level Analysis Under Position-Aware Controls

Per-residue distance-profile scores were computed for the best pair.
Domain enrichment was tested under three position-aware nulls:
(1) contiguous-interval: all same-length windows across the protein;
(2) circular-shift: circular domain shift; (3) local-matched: windows
within ±50 residues.

| Domain | raw score | cubic-residual | interval p | circ-shift p |
|--------|-----------|----------------|------------|--------------|
| TNB | 0.745 | −0.028 | 0.31 | 0.29 |
| REC | 0.395 | 0.073 | 0.41 | 0.58 |
| RuvC-I | 0.745 | 0.044 | 0.02 | 0.01 |
| RuvC-II | 0.726 | −0.074 | 0.40 | 0.38 |
| WED | −0.146 | 0.048 | 0.99 | 0.99 |
| Linker | 0.675 | 0.167 | 0.43 | 0.42 |

Under naive residue-permutation nulls, TNB and REC appear significantly
enriched, and WED appears significantly depleted. All of these patterns
disappear under position-aware controls. TNB's cubic-residual score is
negative (−0.028), and WED returns to background (+0.048).

**Domain-level segment CKA with position baseline:**

| Domain | raw max | pos baseline | excess |
|--------|---------|-------------|--------|
| TNB | 0.412 | 0.672 | −0.260 |
| REC | 0.275 | 0.333 | −0.058 |
| WED | 0.576 | 0.551 | +0.025 |
| RuvC-I | 0.242 | 0.391 | −0.150 |
| Linker | 0.718 | 0.907 | −0.189 |

All domains except WED show negative excess. TNB is substantially below
the position-only baseline. WED's +0.025 excess is negligible and does
not support domain-specific alignment. (The earlier WED depletion in
per-residue distance-profile scores is specific to that metric and does
not contradict the segment CKA result: WED's segment CKA is close to
its position baseline while its per-residue score reflects local
distance-profile anti-alignment with the global position gradient.) The apparent domain-level patterns
in the raw per-residue scores are consequences of where each domain falls
along the global positional gradient, not evidence of functional convergence.

### 2.5 Position Sensitivity Is Head-Specific

Sinusoidal position encoding was evaluated against all 1,216 RF3 heads:

| CKA range | heads | fraction |
|-----------|-------|----------|
| > 0.5 | 227 | 18% |
| > 0.3 | 457 | 38% |
| < 0.1 | 325 | 27% |

Only 18% of RF3 heads respond strongly to pure position encoding. The
selected RF3 head (mod0076_h00, sinusoid CKA = 0.595) is among them.
High cross-model CKA emerges only when a position-sensitive EVO2 head
pairs with a position-sensitive RF3 head. Most head pairs lack this
matched geometry.

The residualized full scan (cubic trend removal) selects entirely different
head pairs (mod0003_h07 ↔ mod0059_h08, CKA = 0.543), indicating that the
raw and residual signals use different head combinations. For the fixed
raw-best pair (mod0004_h28 ↔ mod0076_h00), cubic residualization collapses
CKA to 0.004, confirming that the raw-best pair's signal is almost entirely
positional.

### 2.6 Search for Position-Weak High-CKA Pairs

If high CKA strictly requires position-sensitive heads, then no head pair
with low position sensitivity on both sides should achieve substantial CKA.
We tested this by computing the sinusoid position-baseline CKA for every
EVO2 and RF3 head, then filtering pairs by their position-sensitivity
percentiles.

| Filter | pairs | max raw CKA | best EVO2 | best RF3 |
|--------|-------|-------------|-----------|----------|
| All pairs | 194,560 | 0.851 | mod0004_h28 | mod0076_h00 |
| Both bottom 50% | 48,640 | 0.341 | mod0003_h15 | mod0030_h03 |
| Both bottom 25% | 12,160 | 0.233 | mod0003_h28 | mod0013_h04 |
| Both pos < 0.10 (abs) | 2,874 | 0.094 | mod0001_h17 | mod0001_h08 |
| Both pos < 0.05 (abs) | 319 | 0.041 | mod0001_h18 | mod0015_h03 |

When both heads are position-weak (bottom 25% or absolute CKA < 0.10),
the maximum pair CKA drops to 0.09–0.23 — comparable to the all-pairs
mean (0.15). No position-weak pair exceeds CKA = 0.35 even under the most permissive
filter (bottom 50%, max = 0.341). This confirms that high cross-model
CKA requires matched position-sensitive heads on both sides.

### 2.7 Cross-Mutant Robustness

To test whether the best-pair alignment is stable across protein sequence
variation, we repeated the full 160 × 1,216 CKA scan on 10 additional
randomly selected mutants (all compared against the same WT RF3 structure).

| Statistic | Value |
|-----------|-------|
| WT max CKA | 0.851 |
| Mutant max CKA (mean ± sd) | 0.848 ± 0.009 |
| Mutant mean CKA (mean ± sd) | 0.163 ± 0.001 |
| Best RF3 head invariant | mod0076_h00 in 10/11 mutants |
| Best EVO2 heads | mod0000_h00 (6/11), mod0004_h28 (2/11), mod0004_h14 (2/11) |

The maximum CKA is extremely stable across mutants (coefficient of
variation ~1%). The best RF3 head is invariant. The best EVO2 head
varies among a small set of nearby deep EVO2 modules
(mod0000_h00, mod0004_h28, mod0004_h14). These findings
are consistent with the coordinate-geometry interpretation: protein
point mutations have negligible effect on the shared positional
structure that drives the alignment.

### 2.8 Homodimer Chain Asymmetry

Chain A and Chain B CKA values correlate at r = 0.975 across all 194,560
pairs. The best EVO2 head (mod0004_h28) is the same for both chains, but
the optimal RF3 heads differ (mod0076_h00 for Chain A, block 18/50;
mod0063_h02 for Chain B, block 5/50). Both are DiffusionTransformer blocks.
A/B averaging reduces the maximum CKA from 0.851 (single chain) to 0.800,
indicating that chain-specific features carry signal that is diluted by
averaging. At segment scales (window = 100 aa), chain differences are
minimal (|Δ excess| < 0.07).

### 2.9 Mutation-Induced Delta Analysis

To test whether mutation-specific cross-model alignment exists beyond the
static WT baseline, we computed mutation-induced feature deltas for 300
paired EVO2–RF3 mutants (both models run on the same protein variant):

\[
\Delta E_m = E_m - E_{WT}, \quad \Delta R_m = R_m - R_{WT}
\]

for the fixed head pair (mod0004_h28 ↔ mod0076_h00). RF3 mutant outputs
were verified to differ substantially from WT (‖R_m − R_WT‖ ≈ 300–400
for sampled mutants), ruling out file-reuse artifacts.

| Analysis | CKA (mean ± sd) | Interpretation |
|----------|-----------------|----------------|
| Observed ΔE_m vs ΔR_m | 0.207 ± 0.079 | Ordered delta structure exists |
| Pair-shuffle ΔE_m vs ΔR_{m'} | 0.212 ± 0.079 | No mutant-specific pairing (p = 0.42) |
| Residue-shuffle null | 0.005 ± 0.003 | Residue order required (p < 1e-10) |

The observed delta CKA is significantly above the residue-shuffle null,
confirming that mutation-induced feature changes are spatially ordered.
However, it is indistinguishable from the pair-shuffled null (Figure 2A):
randomly
pairing EVO2 deltas from one mutant with RF3 deltas from a different
mutant produces the same mean CKA. This means the delta signal reflects
a shared global perturbation mode that all mutants project onto, rather
than mutation-specific cross-model response alignment.

Per-residue analysis corroborates this interpretation. The correlation
between ‖ΔE_m‖ and ‖ΔR_m‖ across residues averages 0.133 ± 0.236
(weak, high variance), while ‖ΔE_m‖ correlates modestly but stably
with contact density (0.111 ± 0.030), suggesting that perturbation
magnitudes are biased toward structurally central residues. This
structural bias is not sufficient to establish mutation-specific
coupling, as it persists independently of correct mutant pairing.

**Conclusion:** Mutation-induced deltas do not rescue a content-level
interpretation. The perturbation response is spatially ordered but
mutation-nonspecific, consistent with a shared global perturbation
axis rather than matched mutant-specific structural responses.

Additional targeted tests further excluded residual mutation-specific
signal. After subtracting the mean delta across mutants, delta CKA
collapsed from 0.209 to 0.041, with no excess over pair-shuffled
controls. Removing leading PCs also failed to create matched-pair
enrichment. Retrieval analysis was indistinguishable from random: using
ΔE_m to retrieve its matched ΔR_m among 200 RF3 mutants gave top-1
accuracy 2/200, top-5 accuracy 4/200, and mean rank 102.2, close to
the random expectation of mean rank 100. Thus the delta signal is best
explained by a global mutation-nonspecific perturbation axis rather
than matched mutant-specific responses.

Mutation-site local and 3D contact-neighborhood delta scans further
closed the remaining possibilities. At mutation-site windows (±5, ±10,
±20 aa) and 3D contact neighborhoods (8 Å, 12 Å, long-range >24 aa),
observed delta CKA was effectively zero (all < 0.002) with zero excess
over pair-shuffled controls. The global delta CKA of 0.21 does not
localize to the mutation site or its structural neighborhood — the
perturbation signal is entirely global. Local retrieval was similarly
random: top-1 accuracy 0.5–1.0%, mean rank 95–97 (random expectation:
100). Figures 1–4 are described in the Data and Code section below.

### 2.10 FAESM–EVO2–RF3 Triangular Comparison

To distinguish shared low-frequency coordinate geometry from potential
evolution-derived protein semantics, we compared EVO2, FAESM (ESM2-650M,
a protein language model with learned rather than hard-coded position
encoding), and RF3
under a common progressive residualization ladder (Figure 4). All CKA
values use the consistent `linear_cka` implementation with native feature
dimensions (no truncation). Residualization was applied symmetrically to
both representations using identical residue-index bases. Fourier bases
include intercept, cubic polynomial terms, and sine/cosine pairs up to
the indicated frequency.

| Pair | Raw | Linear | Cubic | Fourier5 | Fourier10 | Fourier20 |
|------|-----|--------|-------|----------|-----------|-----------|
| EVO2 × RF3 | 0.851 | 0.101 | 0.032 | 0.002 | 0.000 | 0.000 |
| FAESM × RF3 | 0.870 | 0.401 | 0.272 | 0.063 | 0.072 | 0.024 |
| FAESM × EVO2 | 0.101 | 0.005 | 0.001 | 0.000 | 0.000 | 0.000 |

Three findings emerge. First, EVO2–RF3 alignment is dominated by
low-frequency coordinate structure, collapsing to near-zero after
cubic or Fourier residualization. Second, FAESM–RF3 retains substantial
mid-frequency structure (cubic residual 0.272) that only collapses under
Fourier bases, suggesting that the protein language model shares richer
residue-order geometry with the structure model than the DNA language
model does. Third, and most critically, FAESM–EVO2 alignment is weak
even before residualization (0.101) and vanishes after removing a linear
trend (0.005). If a shared evolution-derived protein semantics were
driving cross-model alignment, FAESM and EVO2 should retain non-positional
alignment. Their near-orthogonality argues against this interpretation.

Thus, FAESM is closer to RF3 not because it shares EVO2-like evolutionary
semantics, but because it shares richer mid-frequency residue-order
geometry with the protein structure model. The triangular comparison
demonstrates that the progressive residualization ladder can decompose
raw CKA into position-structure components of varying frequency scales,
and that cross-model alignment in this setting does not reflect shared
evolution-derived residue semantics.

### 2.11 FAESM Mutation Delta Analysis

To test whether the protein language model exhibits mutation-specific
cross-model alignment, we repeated the delta analysis on FAESM–RF3 using
300 paired mutants (same protein variants). All methods matched the EVO2
delta pipeline.

| Metric | EVO2 × RF3 | FAESM × RF3 |
|--------|-----------|------------|
| Observed delta CKA | 0.207 ± 0.079 | 0.252 ± 0.110 |
| Pair-shuffle null | 0.212 ± 0.079 | 0.257 ± 0.114 |
| Excess (obs − shuf) | −0.005 | −0.005 |
| Mean-removed observed | 0.041 | 0.310 |
| Mean-removed pair-shuffle | 0.041 | 0.309 |
| Retrieval top-1 | 1.0% | 0.0% |
| Retrieval mean rank | 102.2 | 155.5 (random: 150) |

FAESM delta CKA is indistinguishable from the pair-shuffled null
(excess = −0.005), and retrieval of matched mutants is at chance (top-1
0.0%, mean rank 155.5 vs. random 150). After removing the mean delta,
the residual CKA increases (0.252 → 0.310) but remains equal for
matched and shuffled pairs. Thus, even a protein-level language model
fails to produce mutation-specific cross-model alignment under these tests.

Together with the static analysis (§2.10), these results establish that
high static cross-model CKA — whether from a DNA LM or a protein LM —
does not imply recoverable mutation-specific or residue-level semantic
alignment. Both models exhibit shared residue-order geometry with RF3,
but neither shows matched perturbation-response specificity.

---

## 3. Discussion

### 3.1 Global CKA Reflects Coordinate Geometry, Not Functional Alignment

The raw CKA = 0.851 is a real, non-random, head-specific alignment signal.
It withstands full-scan max-null correction (p ≈ 0.0005) and residue-order
shuffle. However, it is primarily driven by shared low-frequency
coordinate geometry along the protein chain, not by domain-specific
functional content.

Several lines of evidence converge on this interpretation:

- A pure sinusoidal position encoding yields CKA = 0.598 against the
  same RF3 head, indicating that position-only representations can reach
  a similar order of magnitude. This comparison is not additive (CKA is
  non-linear) and should not be interpreted as exactly 70% variance
  explained.
- Removing a linear position trend collapses the fixed-pair CKA from 0.852
  to 0.022.
- Position-only baselines systematically outperform real EVO2 at segment
  scales (mean excess −0.43 to −0.48).
- Domain-level enrichment patterns disappear under position-aware nulls.
- The best residual head pair differs completely from the best raw pair.

### 3.2 Why Smoothness Dominates: Mechanical, Not Semantic

The high CKA does not require EVO2 to compute RF3-like chain topology
or domain-specific structure. Systematic synthetic-signal tests show
that the alignment is driven by shared low-frequency autocorrelation
along the residue axis:

- A simple linear position feature (128-d repeat of residue index)
  achieves CKA = 0.76 with the RF3 head — comparable to the sinusoidal
  encoding (CKA = 0.60) and approaching the real EVO2 value.
- A U-shaped chain-end topological distance feature produces CKA near
  zero (0.06), ruling out symmetric chain-end topology as the driver.
- AR(1) smooth noise, which contains no biological or structural
  information, achieves moderate CKA with RF3, scaling with the
  autocorrelation parameter ρ.

The mechanism is mechanical rather than semantic. Both architectures
perform extensive position mixing in their deep layers: RF3's
pair-bias propagation across 50 diffusion blocks smooths token features
along the chain axis, while EVO2's RoPE attention across 5 deep layers
cumulatively integrates sequence-position information. In both cases,
the result is a low-frequency signal along the residue coordinate.
CKA, as a Gram-matrix similarity metric, strongly rewards shared
low-frequency structure regardless of its biological origin.

The finding that neither chain-end topology (RF3 side) nor explicit
absolute position encoding (EVO2 side) is required for the alignment
confirms this interpretation. The CKA signal reflects a generic
smoothing bias induced by deep position mixing, not a specific
structural or functional convergence between the two models.

### 3.3 Limitations and Future Work

- **Mutant panel.** Cross-mutant analysis on 10 random mutants (§2.7)
  shows high stability (max CKA 0.848 ± 0.009, best RF3 head invariant).
  This supports the coordinate-geometry interpretation but does not
  distinguish content from coordinate variation: both are expected to be
  stable under sparse point mutations. A stronger perturbational test
  would correlate ΔCKA across mutants with mutation position, predicted
  structural disruption, or DMS fitness scores after removing the shared
  position baseline.

- **Codon table.** The native S. thermotolerans codon table was used.
  Alternative tables have not been tested. The frame-shift invariance
  suggests codon-table effects are likely minor, but this has not been
  verified.

- **Max-null scope.** 2,000 permutations establish p ≈ 0.0005. The
  99.9th null percentile (0.512) is well below the observed maximum
  (0.851).

---

### 3.4 Null Hierarchy Summary

The following table summarizes the full hierarchy of controls applied:

| Null / Control | What it tests | Result |
|---|---|---|
| Residue-shuffle | Residue-order dependence | CKA collapses (0.85 → 0.01): non-random order |
| Max-null full scan | Multiple testing (194,560 pairs) | p ≈ 0.0005: head pair significant |
| Position-only baseline | Coordinate geometry dominance | Sinusoid 0.60, linear 0.76: position explains majority |
| Polynomial residualization | Low-order position trend | Linear residual 0.02, cubic 0.004: fixed pair positional |
| Contiguous-interval null | Domain position confound | Enrichment vanishes: not domain-specific |
| Position-weak head filter | Need for position-sensitive heads | Max CKA 0.09 when both pos < 0.10 |
| Segment excess scan | Local EVO2 advantage | Mean excess −0.43 to −0.48: no segment advantage |
| Cross-mutant stability | Mutation sensitivity | CKA 0.848 ± 0.009: mutation-insensitive |
| Pair-shuffle delta | Mutant-specific response | Observed ≈ pair-shuffled (p=0.42) |
| Delta global-mode removal | Common perturbation axis | Collapses to 0.04 after mean removal |
| Local/contact delta | Local structural response alignment | Zero: mutation signals do not carry local structural information matching RF3 |
| Mutant retrieval | Matched mutant identification | Random (top-1 1%, mean rank 102) |

---

## 4. Conclusion

The EVO2–RF3 raw full-length CKA is real, non-random, and head-specific,
but it reflects a generic low-frequency smoothing bias induced by deep
position mixing in both architectures, rather than domain-level biological
content alignment. At segment and domain scales, explicit position bases
systematically outperform real EVO2 features. Apparent domain enrichment
under naive nulls is eliminated by position-aware controls, and no
position-weak head pair achieves high CKA.

Mutation-induced delta analysis further closes the content-level interpretation:
while mutation responses are spatially ordered, they do not show
mutation-specific cross-model alignment — pairing the correct mutant's
EVO2 and RF3 deltas gives no higher CKA than randomly pairing deltas from
different mutants (p = 0.42). Both the static and perturbation-based
analyses therefore converge on the same conclusion: the single-locus
WT/mutant analyses do not support the interpretation that EVO2 captures
domain-specific functional information that exceeds coordinate geometry.
Future work requiring mutant-panel perturbation specificity would need
to localize to mutation-site neighborhoods, contact interfaces, or
long-range pairwise coupling after removing position baselines.

---

## 5. Data and Code Availability

All analysis code, precomputed matrices, null distributions, sample mutant
activations, and figures are publicly available at the GitHub repository:
**[github.com/Lizon1c/cross-model-cka](https://github.com/Lizon1c/cross-model-cka)**.

### Repository Structure

| Directory | Contents |
|-----------|----------|
| `data/matrices/` | Precomputed CKA matrices (`full_cka_matrix.pt`, 160×1216), chain A/B comparison |
| `data/nulls/` | Max-null distributions (`max_null_2000.pt`, 2000 permutations), frame-shift diagnostics |
| `data/mutants/` | Sample head activations (17 .pt files: 11 FAESM, 5 EVO2, 1 RF3 WT; 916 MB via Git LFS) |
| `data/figures/` | Paper figures (4 PNGs: CKA matrix histograms, delta null, local delta, residual ladder) |
| `data/G00/` | Cross-group sensitivity analysis (1 of 10 mutant groups; G01–G09 available upon request) |
| `scripts/core/` | Core alignment libraries (`align_residues.py`, `evo2_to_rf3_align_v32.py`) |
| `scripts/scan/` | Full head-pair scan pipeline |
| `scripts/controls/` | Max-null, per-residue pearson, segment analysis |
| `scripts/diagnostics/` | Position baselines, frame-shift scan, residual scan, chain A/B diagnostics |
| `scripts/report/` | Best-pairs analysis, report generation |
| `paper/` | Final paper (PDF, DOCX, Markdown, Chinese version), alignment report |

### Large Data

Raw head activation data for all 8,000+ mutants per model (~339 GB total)
is available upon request from the corresponding author. Contact:
`dingcheng2024@shanghaitech.edu.cn`.

### Reproducibility

The 14 analysis scripts under `scripts/` reproduce all paper figures and
tables from the precomputed matrices in `data/`. Script documentation is
provided in `scripts/README.txt`. Dependencies are listed in
`requirements.txt`. Path configurations (hardcoded in the original
environment) can be overridden via `export PROJECT_ROOT=`.

### Figures

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `data/figures/fig_cka_matrix_null.png` | Full CKA matrix histogram (194,560 pairs) + max-null distribution (2,000 permutations) |
| Figure 2 | `data/figures/fig_delta_null.png` | Delta CKA: observed vs pair-shuffle vs residue-shuffle; global mode removal after mean subtraction |
| Figure 3 | `data/figures/fig_local_delta.png` | Delta CKA: global vs local sequence windows (±5/±10/±20 aa) and 3D contact neighborhoods (8Å, 12Å, long-range) |
| Figure 4 | `data/figures/fig_residual_ladder.png` | Progressive position-basis residualization: EVO2×RF3, FAESM×RF3, FAESM×EVO2 under linear, cubic, Fourier-5/10/20 bases |
