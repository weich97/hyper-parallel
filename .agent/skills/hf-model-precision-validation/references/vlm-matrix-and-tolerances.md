# VLM Precision Matrix and Tolerance Policy

Use this reference when the final model consumes real image/video/audio inputs or
when a modality encoder/projector is parallelized. A text matrix and a single
Encoder-Parallel A/B are insufficient VLM coverage. Define the scope of each
matrix explicitly: the compact encoder overlay below covers the interaction of
SP, encoder degree, and feature exchange; the broader decoder/recompute/resume
matrix remains a separate obligation. "Complete" for a declared scope means
every row ran or has a specific, evidenced exclusion, not a Cartesian product.

## Freeze the experiment before launching

Record the final model and replacement tree, retained decoder and modality-tower
depth, parameter crop, real-media dataset, tokenizer/processor revisions, input
replay rule, global batch, optimizer, precision policy, 10 measured updates,
warm-start step, and source revision. Use one common full-state checkpoint for
all comparable cases. From a step-1 warm-start, run through global step 11 and
compare steps 2–11. Keep the data cursor policy explicit: same-input replay is
not a K-to-K+N dataloader-continuation test.

Create `matrix-ledger.json` before execution. For every row record `case_id`,
`category`, full decoder topology (`TP,CP,EP,FSDP,PP,SP`), modality topology
(`encoder enabled, encoder CP/DP, feature-exchange mode`), workload ID,
recompute selection, resume mode, comparison parent, acceptance profile, expected
steps, and `planned`/`excluded` disposition. Exclusions need the failing
constraint and a preflight artifact. Resolve launcher arguments from this ledger;
the executed config must match the ledger exactly. An old-revision result or a
unit test cannot be marked `executed` for the current full-model matrix.

When the generic matrix launcher cannot express the modality overlay, use the
skill's thin launcher and comparator with the frozen ledger:

```bash
python .agent/skills/hf-model-precision-validation/scripts/run_vlm_matrix.py \
    --ledger <evidence-dir>/matrix-ledger.json --dry-run
python .agent/skills/hf-model-precision-validation/scripts/run_vlm_matrix.py \
    --ledger <evidence-dir>/matrix-ledger.json
python .agent/skills/hf-model-precision-validation/scripts/compare_vlm_matrix.py \
    --ledger <evidence-dir>/matrix-ledger.json
```

The ledger schema used by these scripts contains `devices`, `device_ids`, `source_checkout`,
`source_sha256`, `asset_paths`, `asset_sha256`,
`acceptance.strict`, `acceptance.bf16_cross_layout`,
`acceptance.parameters`, and `cases`. Each case declares `case_id`, `status`
(`planned`, `running`, or `existing`), `config`, `evidence_dir`,
`measured_steps`, and `compare_to` (case ID or null); keep the full topology,
global batch, category, and checkpoint path alongside them for audit. The
launcher skips an existing case only when all expected structured steps exist.
The launcher also requires the previous execution record to carry matching
source and ledger fingerprints. Freeze the source fingerprint from all
`hyper_parallel/**/*.py`, `scripts/train_vl.py`, and the ten generated YAML
files before the first run. Record paths and SHA256 hashes for model config,
training records, and cropped model assets. The runner rechecks source and
assets before each case; the comparator requires matching fingerprints. A
changed acceptance policy requires a separately labeled comparison artifact.
The same raw evidence may be rescored when source, inputs, and scalar limits
are unchanged; never present that result as the original frozen verdict. Do not
reuse old evidence after a refactor. Declare exactly eight distinct
physical NPU IDs in `device_ids`. The launcher masks visibility with
`ASCEND_RT_VISIBLE_DEVICES`, checks `npu-smi info` for selected-device
processes before every case, and holds a local reservation lock while running
the sequential matrix. If another process owns any selected card, the case is
`BLOCKED` and must be retried when those cards are free.
Use `pair_checks` for additional comparisons that reuse the ten runs, such as
ECP1 versus ECP2 under both SP settings. The comparator preserves incomplete
rows and writes a per-case and per-pair JSON verdict; it cannot promote an
incomplete matrix to PASS.

## Compact encoder/SP covering matrix

For eight ranks, use ten formal VLM runs rather than every degree combination.
All cases use the same cropped model, real image plus text, optimizer, fixed
input replay, and ten measured updates. `off` means encoder redistribution is
disabled; `ECP1` activates encoder DP without vision sequence sharding; `ECP2`
activates vision sequence sharding and encoder DP. `EDP=(TP×CP)/ECP`.

| Plane | Decoder topology | SP | Encoder settings | Route and purpose |
| --- | --- | --- | --- | --- |
| Main A | TP4 CP2 EP8 FSDP2, batch 1 | true | off, ECP1 (EDP8), ECP2 (EDP4) | Parent A2A, SP token owners; isolates encoder DP from CP. |
| Main B | Same as A | false | off, ECP1 (EDP8), ECP2 (EDP4) | Parent A2A, non-SP TP fan-out; crosses each encoder mode with SP. |
| TP A | TP4 CP1 EP8 FSDP2, batch 2 | true | off, ECP2 (EDP2) | TP variable A2A. |
| TP B | Same as TP A | false | off, ECP2 (EDP2) | TP variable all-gather. |

Compare each enabled case to the `off` case **in its own plane**; compare ECP1
with ECP2 within each main SP setting. Compare SP true/false for the same
encoder setting only when checkpoint, global batch, image records, and input
hashes match. The CP1 pair uses batch 2 because the eight-rank decoder DP
degree is 2; never compare its loss curve directly with batch-1 main cases.
Express the SP and ECP cross comparisons as `pair_checks`; a table of ten run
rows alone does not prove that the crossings were compared.
The main plane covers the 2×3 SP/encoder cross. The TP plane covers the two
different CP1 feature-exchange implementations. ECP4/ECP8 are degree stress
tests: add them when the stated question is degree scaling, but they do not
replace an SP cross. EP1, TP1/CP1 LOCAL, text-only, uneven multi-image,
recompute, and continuation belong in separately declared matrices; report
them as outside this encoder/SP scope if they were not run. This ten-case
matrix alone must never be described as a complete VLM acceptance matrix.

For a broader VLM acceptance claim, record separate dispositions for native
ViT/projector/merge parity, decoder DP/FSDP/EP/CP/TP and supported hybrids,
real image and text-only/uneven multi-image workloads, recomputation depths,
uninterrupted versus dataloader-cursor continuation, and Production/Validate
where supported. A case may cover several properties, but do not count it as
several independent strategy comparisons. Use explicit evidence for unsupported
PP>1, vision checkpointing, or encoder Validate mode.

On eight ranks, the dense data-parallel degree is `8/(TP×CP×PP)`; the global
batch must divide into that many single-sample microbatches. A `TP4×CP2`
experiment with global batch 1 therefore cannot be re-labelled as a comparable
`TP1×CP1` DP8 axis case. Run a separately declared fixed-global-batch decoder
axis matrix (for example global batch 8 throughout), with its own common
checkpoint and baseline, or mark those axis rows `BLOCKED` with this exact
feasibility evidence. Never compare loss curves from different global batches
as though they were a topology-only A/B.

## Pairing and evidence gates

1. Compare encoder on/off only with identical decoder TP/CP/EP/FSDP/SP, source
   checkpoint, optimizer/scheduler state, LR, global input hashes, and image
   records. Separately compare ECP degrees to the same disabled reference and
   to ECP1 to isolate visual sharding from visual work redistribution.
2. Compare decoder topology variants to the declared decoder baseline only
   when global batch and input records are identical. Do not treat an ECP change
   as a decoder strategy case, or a recompute/resume case as additional strategy
   coverage.
3. Require 10 aligned globally aggregated loss and pre-clip global norm records
   per formal case, with rank-indexed FP32 main-gradient/update probes,
   vision/projector gradient presence, image-feature/merge shape evidence,
   input hashes, LR, finite status,
   optimizer layout, and checkpoint metadata. Missing steps or files are
   `INCOMPLETE`; a present run with invalid provenance is `BLOCKED`, and a
   numeric bound violation is `FAIL`.
4. Gate numerical self-consistency on each step's loss and pre-clip global
   norm, after input, checkpoint, LR, and finite-state checks. Parameter and
   main-gradient probes diagnose scalar divergence; they do not veto an aligned
   ten-step curve. Report each step's absolute and reference-relative loss/norm
   delta, maxima, first failing step, and trend after optimizer updates. A
   matching first forward does not establish ten-step self-consistency. Report both strict
   scalar and BF16 cross-layout verdicts when both were predeclared.
5. Do not aggregate a matrix to `PASS` if one planned row is unexecuted,
   `BLOCKED`, or `FAIL`. Publish per-category coverage and exclusions beside
   the overall result. Preserve failed cases rather than silently replacing
   them with easier topologies or shorter runs.

## Setting numerical bounds

There is no universal BF16 percentage that proves model equivalence. Fix limits
before candidate results are inspected, using an independent calibration:

1. Run at least three identical-state, identical-topology reference repeats on
   the target accelerator and workload. Measure max per-step loss and pre-clip
   norm spread, plus FP32 main-gradient probe spread. Exact deterministic
   repeats should normally be exact; explain any nonzero floor.
2. Run native vs optimized module forward/backward parity in FP32 and BF16.
   Compare complete tensors with elementwise max-abs and relative L2; for
   summary-only probes compare named summary metrics only. Also check discrete
   routing/top-k, token ownership, image counts, and shape identity exactly.
3. Declare separate profiles: same-topology repeat/resume, BF16 cross-decoder
   topology, and encoder on/off or ECP layout change. For each scalar use both
   an absolute bound and a relative bound, with `any` (either bound passes)
   **per scalar** when justified for near-zero values. Loss and norm must each
   pass on every measured step. Record the combination rule explicitly.
4. For this cropped DeepSeek-V4.1 workload, the user-selected *next-run*
   8-rank BF16 profile is: strict loss `abs<=5e-4` or `rel<=1e-4`;
   strict pre-clip norm `abs<=0.002` or `rel<=5e-3`; cross-layout loss
   `abs<=0.005` or `rel<=5e-3`; cross-layout pre-clip norm `abs<=0.125`
   or `rel<=1e-2` (1%). The preceding matrix used `rel<=5e-3` (0.5%)
   for cross-layout norm and its original verdict must remain available.
   These are engineering limits for loss/norm curve agreement, not BF16
   mathematical guarantees. The same-topology norm relative fallback of
   0.5% is permissive: require repeat calibration and report absolute
   deviation so it cannot hide drift. A policy revision after a run must
   use a separately named verdict artifact and state whether it rescored
   old evidence or came from a fresh matrix; never overwrite the original.
5. Keep authoritative module parity checks strict. For full-model cross-layout
   parameter diagnostics, use a scale-aware profile or report raw deltas when
   the reference noise floor is unknown; do not turn a probe threshold into a
   hidden numerical acceptance gate. For tensor probes, report max-abs and
   relative L2 separately; for summary-only probes, report scalar norm bounds.
   Do not call summary statistics an elementwise error. Exact
   checkpoint/input/LR/finite/ownership gates never receive numeric slack.

For the current visual-gradient reduction, `sum_r BF16(sum_rows g_r)` and
`BF16(sum_rows sum_r g_r)` are different algorithms in BF16. A candidate can
have fewer casts and still be farther from an FP64 local reference. Diagnose
the first differing intermediate and state which reference the tolerance
means: native BF16 parity, FP32/FP64 local arithmetic, or training-curve
agreement. Do not infer one from another or loosen the VLM curve limit to
rename a scalar FAIL as PASS.
