# Sprint 6 — errors, false statements and corrected interpretations

An honest log, in the tradition of `sprint3_errors.md` and `sprint4_errors.md`,
but with a wider scope: this sprint's most valuable output was **not** a kernel,
it was a list of claims that turned out to be wrong — some of them ours from
earlier sprints, most of them made *during* this sprint and then refuted by the
next measurement.

Everything below was measured on the target M4 (Apple M4, 4 P-cores + 6 E-cores,
SVL = 512 bits, `FEAT_SME` and `FEAT_SME2` both present), Release build.

---

## Resolution status (added after the log was written)

Every item below has since been worked through against the report. Status:

| item | resolution |
|---|---|
| §1.1 half-applied ABI fix | **Closed in-sprint** — all 17 entry points fixed, `[sprint6][abi]` test pins them |
| §1.2 wrong denominator | **Fixed in code and docs.** `main_norm` now measures the ceiling as a *curve* across footprints (64 KiB → 256 MiB, both modes) and every table divides by the ceiling at its own working set. All five affected report sections re-baselined; Sprint 5 needed none. The §1.2 table reproduced on a fresh run, with the 256 MiB row matching the standalone probes as its control |
| §1.3 Welford accuracy | **Corrected** in `norm_sprint2_layernorm.rst` and `ROADMAP.md`, including the wrong *mechanism* and the previously undocumented 3.5× accuracy cost of V6's FRSQRTE+NR |
| §1.4 ZA skip | **Corrected** — the report said ZA-SME2 was "not attempted"; both kernels are committed and measured. Report and ROADMAP now carry the retraction and the measured verdict |
| §2.1–2.3 threading | **Corrected** in `norm_sprint5.rst` with the alignment control table; the unidentified base-offset mechanism is recorded as unidentified, not explained |
| §2.4 V7 mechanism | **Corrected** in `norm_sprint6.rst` — the `v7x2` control table is now in the report, and the claim is narrowed to a threshold effect; the report previously said the MLP hypothesis "survives", which the control does not support |
| §2.5–2.6 GFLOPS lens | **No artifact** — never reached the report; a guard note added to Sprint 6 so the FMOPA figures are not misread as a target |
| §2.7–2.8 vendor claims | **No artifact** — never reached the report. Deliberately left out rather than added, since the vDSP baseline is our own composition and the traffic figure was assumed |
| §Part 3 process errors | Lessons; the assertion practice is the durable one |
| §Part 4 measurements | **Added to the report** (Sprint 6, "Instrument readings from this sprint"), including the finding that cache-resident RMSNorm V7 is FP-issue-bound at 100 % of the SSVE issue ceiling — which is what explains V7's ≈0 gain there |

One correction to this log's own §1.2 table: the **32 MiB** point is the least
stable on the curve (88.6 here, 92.8 and 96.0 on two later runs). It sits at the
L2→DRAM knee and should not be quoted precisely.

---

## Part 1 — False statements already in the repo (pre-existing)

### 1.1 "The Sprint-5 AAPCS64 fix is done"

**Claimed:** Sprint 5 recorded the `d8–d15` clobber as fixed.

**Reality:** it was applied to the two V6 winners only — the kernels TEIR calls —
and left everywhere else. A hardware probe over every entry point found **15 of
17 still destroying the caller's `d9–d15`**, including `bw_probe_ssve`, the
roofline probe every "% of peak" figure in the report is divided by.

**How it surfaced:** the first baseline run of `main_norm` printed **65 rows of
`0.00 GiB/s`**. `main_norm` defends itself with caller-side `volatile`, which the
Sprint-2 debug log already described as a workaround; workarounds hold only while
register allocation cooperates, and it had stopped.

**Fixed** across all 15 kernels + the probe. 65 zero rows → 0, and every table
reproduced its historically documented values.

**Why it drifted:** the ABI test covered only the two V6 kernels. A test that
pins two of seventeen cases does not protect the other fifteen. There is now one
covering all 17 (`[sprint6][abi]`).

### 1.2 "The single-core roofline is 59.4 GiB/s" — wrong denominator for most shapes

**Claimed:** since Sprint 2a, every "% of roofline" in the report divides by one
number, 59.4 GiB/s.

**Reality:** that figure was measured on 128 MiB arrays, i.e. it is the **DRAM**
ceiling. The ceiling is strongly footprint-dependent:

| total working set | NEON GiB/s | SSVE GiB/s | SSVE/NEON |
|---|---|---|---|
| 2 MiB   | 124.34 | 116.03 | 0.93 |
| 8 MiB   | 123.84 | 115.74 | 0.93 |
| 16 MiB  | 122.35 | 115.53 | 0.94 |
| 32 MiB  | 101.47 |  88.56 | 0.87 |
| 64 MiB  |  85.53 |  63.14 | 0.74 |
| 256 MiB |  79.47 |  59.50 | 0.75 |

**Consequence:** every cache-resident percentage in the report is inflated by
roughly 2×. The Sprint-2 claim that V6 reaches "~95 % of the moved-bytes
roofline" at 16 MiB is really **~50 %**.

This is the *same class of error* Sprint 2a itself identified and named ("peak of
**what**"). Sprint 2a fixed the execution-mode half of the question (NEON vs
streaming) and left the footprint half unasked.

**Second consequence, new:** streaming mode has ~25 % less DRAM bandwidth than
NEON (0.75×) but is nearly equal in cache (0.93×). For a memory-bound kernel at
large N, choosing streaming mode is a structural handicap no amount of kernel
quality recovers.

### 1.3 "Welford is ~100× less accurate; two-pass dominates on BOTH axes"

**Claimed (Sprint 2c):** the Welford LayerNorm variant is ~100× less accurate
than the two-pass on shifted data, so two-pass wins on speed *and* accuracy.

**Reality — measured on the actual kernels vs a float64 reference (M=16, N=512):**

| input shift | V6 (two-pass) | Welford | V0 (two-pass, exact sqrt) |
|---|---|---|---|
| 0    | 1.16e-5 | 1.09e-5 | 3.33e-6 |
| 1e2  | 1.16e-5 | 1.49e-5 | 3.33e-6 |
| 1e4  | 1.16e-5 | **5.37e-4** | 3.33e-6 |
| 1e5  | 4.30e-3 | **1.69e-3** | 4.30e-3 |

The claim holds at exactly one shift (1e4, and the factor is ~46× not ~100×),
is a wash at 0 and 1e2, and **reverses at 1e5**, where Welford is 2.5× *better*.
"Dominates on both axes" is not supported.

**The stated mechanism was also wrong.** The report blames Welford's variance
recurrence. Isolating the variance alone in FP32 (N=512) shows the opposite:

| shift | naive `E[x²]−μ²` | Welford | two-pass |
|---|---|---|---|
| 0    | 1.79e-8 | 1.79e-8 | 2.99e-6 |
| 1e2  | 1.10e-4 | 3.40e-7 | 2.99e-6 |
| 1e4  | 1.06e-1 | 3.40e-7 | 2.99e-6 |
| 1e5  | 3.46e+1 | 7.36e-6 | 1.12e-5 |
| 1e6  | 2.59e+2 | 7.89e-5 | 6.43e-6 |

Welford's **variance is the more accurate of the two**. The extra output error
comes from the running **mean**: `mean += delta/count` updates at full input
magnitude every element, while two-pass forms the mean once and then squares
already-centred values. LayerNorm's output is `(x−mean)·inv_std`, so mean error
is amplified by `inv_std`.

**Also missing from the ablation:** V6's accuracy floor of 1.16e-5 is not the
algorithm, it is FRSQRTE+NR. V0, which uses exact `FSQRT`+`FDIV`, reaches
3.33e-6 — V6 trades **3.5× accuracy** for that substitution, undocumented.

**What does NOT change:** the decision. Two-pass stays the incumbent because
Welford is 25–30 % slower for a measured reason (a per-column scalar `FDIV`).
That is a speed argument and it is unaffected.

**Context worth adding to the report:** the naive column above is the row that
makes the discussion meaningful. PyTorch's CPU LayerNorm moved *to* Welford
precisely to escape naive
([pytorch#59987](https://github.com/pytorch/pytorch/commit/963c9833668d239fcef59126a0dfe3df4b31e452),
`aten/src/ATen/native/cpu/layer_norm_kernel.cpp`). "Welford ≫ naive" and
"Welford ≈ two-pass" are both true; our report compared against the stronger
baseline without saying so.

### 1.4 Sprint 3's ZA skip rested on an untested factor

**Claimed:** ZA is `mova`-bound, and "even a 2× `mova` speed-up only reaches
~20 GiB/s, a tie at best" — so SME2 multi-vector `mova` cannot rescue it.

**Reality:** `mova` was never characterised as *issue*-bound vs *ZA-port*-bound.
Measured directly (no memory traffic, 16 vectors moved either way):
single-vector **3.88 G vec/s**, 4-vector **15.50 G vec/s** — exactly **4.00×**.
It is issue-bound, and the fold is 4:1, not 2:1.

Rebuilt on multi-vector `mova`, bit-identical to the Sprint-3 kernels on every
shape tested:

| | Sprint-3 ZA | Sprint-6 ZA-SME2 | gain | vs V7 |
|---|---|---|---|---|
| RMSNorm   | 10.07 | 16.35 | **+62 %**  | −37 % |
| LayerNorm |  4.83 | 10.34 | **+114 %** | −24 % |

**The verdict is unchanged — ZA still loses** — but it is now a measured verdict
rather than an extrapolation from an assumed factor. The residual cause is
different from the original diagnosis: the ZA kernels process **one** SVL-row
block per iteration (64 B per column touch) against V6/V7's **four** (256 B),
which is the Sprint-2b access-density lever, not `mova`.

---

## Part 2 — False statements and interpretations made *during* Sprint 6

These are mine, in the order they were made and refuted.

### 2.1 "The 6/10-thread performance cliff is the kernel's tail path"

OpenMP row-chunking showed 6 and 10 threads running ~45 % slower than 4 and 8.
I attributed it to the V6/V7 tail: those chunk sizes are not multiples of 4·VL,
so every thread ends in a predicated single-block tail at 64 B per column touch
instead of 256 B.

**Refuted.** Isolating the two effects single-threaded, same row count:

| case | GiB/s |
|---|---|
| base 0, 640 rows (whole groups) | 23.58 |
| base 0, 682 rows (**partial group**) | 23.94 |
| base 16, 640 rows (64 B-aligned) | 20.99 |
| base 682, 640 rows (misaligned) | 19.89 |

Partial groups cost **nothing**. And after deleting the tail entirely, the cliff
was still there (6 threads: 30.01 naive vs 41.36 aligned).

### 2.2 "Then it must be cache-line alignment"

Also wrong: `base 16` is exactly 64 B-aligned and still 11 % slower than
`base 0`. The effect tracks the *base offset* of the sub-block, with the
allocation start fastest, and I have **not** identified the mechanism.

**What stands:** group-aligned (256-row) chunking avoids it empirically —
+42.4 % at 6 threads, +48.8 % at 10, and 0 % where chunks were already aligned
(the control). The fix is justified by its control, not by my explanation.

### 2.3 "Scaling flattens past 8 threads because of P/E-core heterogeneity"

The "flattening" I described was mostly a **sawtooth produced by my own
harness's chunk misalignment**. With aligned chunking the curve is monotonic
(24.35 → 29.06 → 35.62 → 38.93 → 40.76 → 41.10 → 41.56 for 1→16 threads) and the
real shape is: knee at ~4 threads, genuinely flat from 8 (8→16 buys 2 %).

### 2.4 "V7's win is bytes-in-flight per load-queue entry, scaling with bytes/instruction"

Pre-registered prediction: a 2-vector control (128 B/instruction) should land
**halfway** between V6 and V7, ~+9 %.

**Refuted by the control:**

| variant | load insns/column | GiB/s | vs V6 |
|---|---|---|---|
| V6 (4 × 64 B)   | 4 | 20.94 | — |
| V7x2 (2 × 128 B)| 2 | 24.31 | **+16.1 %** |
| V7 (1 × 256 B)  | 1 | 24.67 | +17.9 % |

The curve **saturates** — 4→2 captures 90 % of the win, 2→1 adds 1.5 %. That is
inconsistent with a proportional bytes-per-slot model (mine) *and* with a
"256 B burst" prefetcher model. Both pre-registered hypotheses are dead; the
supportable statement is a threshold effect on load-instruction count, in
load-dominated loops, in the latency-exposed regime only.

### 2.5 "21 GFLOPS = 67 % of the 31 GFLOPS SSVE FMLA peak"

Not comparable. Their figure counts FMLA as 2 flops/lane; our kernel is
1 FMLA + 2 FMUL per element, so the flop-counting conventions differ.

**Correct unit is FP vector instructions/s** (justified by measurement: SSVE
FMUL and FMLA have *identical* issue rates, 1.00×). Redone:

| | FP vector instructions/s |
|---|---|
| SSVE ceiling (measured here) | 0.97 G (= 31.0 GFLOPS) |
| RMSNorm V7, cache-resident | **0.97 G — 100 %** |
| RMSNorm V7, true DRAM | 0.65 G — 67 % |

Fixing the unit changed the conclusion from a vague "67 %" to "**the
cache-resident kernel is FP-issue-bound at 100 % of the ceiling**", which is
what actually explains why V7 gains 0 % there.

### 2.6 "We are at 1.0 % of the SME FMOPA peak — how do we improve that?"

Wrong lens entirely, and I pursued it for two exchanges. Normalization has
arithmetic intensity ~0.33 flops/byte; reaching a 1984 GFLOPS matrix-unit peak
needs ~31 flops/byte, about 100× more. Decision E already settles this: the
metric is GiB/s, not GFLOPS. "1 % of FMOPA peak" is not a defect, it is what the
roofline permits, and optimising toward it is meaningless for this operator.

(The FMOPA measurements themselves stand and are recorded in Part 3 — they are
just not a target.)

### 2.7 "Apple Accelerate has RMSNorm and LayerNorm, and we are 16–37× faster"

Mislabelled on both legs, checked against the SDK:

* **Accelerate has no RMSNorm at all.** What the table called "Accelerate
  RMSNorm" was **my own composition** of `vDSP_svesq` + `vDSP_vsmul`.
* **`vDSP_normalize` is not LayerNorm** — it computes `(x−mean)/stddev`, with no
  eps, no gamma, no beta. A *smaller* operation than ours.
* The real Apple LayerNorm is `BNNSFilterCreateLayerNormalization`
  (`__API_DEPRECATED(macos(11.0, 15.0))`, superseded by BNNSGraph), and it is
  **untested**.

The 16–37× figure is also measuring the wrong thing: it is vDSP walking a
column-major matrix with `stride = ld`, i.e. a cache miss per element. Given its
own preferred contiguous layout, vDSP is **faster than us**:

| | ours (column-major) | vDSP, same layout | vDSP, contiguous |
|---|---|---|---|
| RMS, 16 MiB  | 38.47 | 1.08 | **65.77** |
| RMS, 256 MiB | 24.73 | 0.67 | **29.49** |
| LN, 16 MiB   | 16.49 | 1.03 | **50.97** |
| LN, 256 MiB  | 13.60 | 0.67 | **28.11** |

### 2.8 "We extract a higher fraction of our mode's bandwidth than Apple does of its own"

Technically 62 % vs 56 % at DRAM, but the reasoning is weak and should not be
used as a headline:

* it treats the execution mode as an external constraint, when it is **our
  choice** — "efficient within a self-imposed handicap" is not "faster";
* the 56 % baseline is **my own two-call composition**, which cannot fuse, so
  out-efficiency-ing it is largely "fused beats unfused";
* the vDSP traffic figure used to compute it was **assumed** (2R+1W), not known.

### 2.9 "A shift-corrected single pass" implied as established practice

It is not. PyTorch CPU LayerNorm uses **Welford**; GPU kernels commonly use the
two-moment form. The shifted-data variance algorithm is a textbook numerical
technique, and I can point to no production ML kernel using it. It remains a
*hypothesis of ours*, whose only real argument is from our own data: it avoids
the per-column scalar `FDIV` that killed our Welford, while keeping one pass.

---

## Part 3 — Process errors (how the mistakes were made)

1. **A scripted fix introduced a silent slot collision.** The ABI transformation
   script rewrote the eps stack slot *after* inserting the `d9` save, so the
   rewrite also matched the new `d9` line: `d9` and eps ended up sharing one
   slot. **Every functional test still passed** (eps stayed self-consistent);
   only the register probe caught it.
2. **A `.replace()` without an assertion silently no-opped.** Building the
   2-vector control, the counter-predicate edit failed to match (a label had been
   renamed first), so `pn9` was never initialised while the loads *were* already
   split — blocks 2–3 read an uninitialised predicate. The earlier ABI script
   asserted; this one did not.
3. **A stub was written into the tree.** The first ZA-SME2 attempt had
   placeholder control flow, `z2` as both accumulator and load target, and no
   pass 2. Removed rather than patched.
4. **An unverified addressing assumption.** ZA-SME2 assumed the vector-group
   cursor stepped by 4. It steps by **1** — `za.s[w8, 0, vgx4]` takes slice `w8`
   from each of the four tiles. Found by a clean failure boundary (N ≤ 16 pass,
   N ≥ 17 fail; 16 = SVL).
5. **The `mova` microbenchmark hit the very bug this sprint was about**,
   reporting `inf` because its own `smstart`/`smstop` kernels did not preserve
   `d8–d15`.
6. **Approval was over-read.** A short "yes" was taken as covering a commit it
   did not cover, twice. Nothing was committed without explicit instruction after
   that.

---

## Part 4 — Measurements from today that stand

Recorded because several are new instrument readings the report should carry.

* **Footprint-dependent ceilings, both modes** — the table in §1.2.
* **SSVE FP issue rate:** FMLA 0.97 G instr/s (31.0 GFLOPS); FMUL 0.97 G instr/s.
  Ratio **1.00×** — FMUL costs exactly what FMLA costs.
* **SME FMOPA issue rate:** 3.87 G instr/s (1983.9 GFLOPS), **3.99×** the FMLA
  *issue* rate and 64× its flop rate. Reproduces the Jena "Hello SME" figure.
* **FMOPA and SSVE FP are disjoint resources.** 8 FMLA = 8.26 ns, 8 FMOPA =
  2.07 ns, 8 of each interleaved = **8.26 ns** = exactly `max`, not `sum`. They
  overlap completely.
* **`mova` is issue-bound:** 4-vector form is 4.00× the single-vector form.
* **Multi-vector MOVA round-trips exactly** (two groups, all Z clobbered between
  write and read-back, 0/128 elements wrong).
* **Our position, against the correct footprint-matched ceiling:**

| | useful GiB/s | moved | ceiling | % of ceiling | what binds |
|---|---|---|---|---|---|
| RMS V7, 16 MiB  | 38.47 | 57.7 | 115.5 | 50 % | SSVE FP issue (100 %) |
| RMS V7, 256 MiB | 24.73 | 37.1 | 59.5  | 62 % | memory |
| LN V7, 16 MiB   | 16.49 | 33.0 | 115.5 | 29 % | SSVE FP issue |
| LN V7, 256 MiB  | 13.60 | 27.2 | 59.5  | 46 % | memory + 3R+1W structure |

* **PyTorch 2.8.0 on this M4**, our shapes and byte convention, its own best
  (contiguous) layout: `torch.compile` 6.27 GiB/s, `torch.nn.RMSNorm` 2.76,
  eager 2.78 (all at 4096×8192, 1 thread). Slow enough that it should be treated
  as "PyTorch is not optimised for this op here", not as a state-of-the-art bar.

---

## What the sprint is actually worth

Two kernels got faster (V7 tail-free; ZA rebuilt on SME2 for both norms, +62 %
and +114 %). But the durable output is this file: **four statements in the
existing report are wrong or overstated** (§1.1–§1.4), and **nine claims made
during the sprint were refuted by the next measurement** (§2). Every one of them
was refuted by an experiment that took minutes to run.

The recurring pattern is worth stating once: every wrong claim here came from
reasoning about a mechanism instead of measuring it — the tail, the alignment,
the P/E cores, the MLP model, the GFLOPS comparison, the vendor baseline. The
ones that survived contact were the ones written down as a prediction *before*
the measurement, which is the practice the earlier sprints established and the
only reason the errors were caught at all.
