# Port notes — smaulbrain-core

The runtime is pure NumPy plus the Python standard library. PyTorch remains only
under `tests/_reference/` as a development oracle for parity checks.

## Numeric deviations

- **float32 runtime.** Parameters, activations, gradients, and optimizer state
  are `float32`. Reductions that the original computed in `float64` (RMSNorm
  statistics, gradient norms, loss normalization, optimizer norm tracking) are
  still accumulated in `float64`. BLAS and NumPy may sum in a different order
  than the original kernels, so results can differ in the last few bits.
- **Autograd implementation.** `smaulbrain.npgrad` records a lightweight tape
  with explicit backward closures. Broadcasting gradients are reduced over all
  leading dimensions; floating-point dtype casts remain differentiable; Python
  scalar constants are kept at float32. Primitive parity tests cover linear,
  recurrent scan, and RMSNorm forward/backward at `rtol` 1e-5 to 5e-5.
- **Reduced-precision activation storage.** `--activation-storage bf16`
  stores cached activation values as raw IEEE bfloat16 bit patterns in
  `uint16` (2 bytes/value, RNE, specials preserved) and widens on access;
  `int8` uses a dynamic symmetric per-tensor scale (1 byte/value plus a
  float32 scale). A computed scale guarantees rounded codes stay in
  `[-127, 127]`, so that path skips a redundant clipping pass; explicit scales
  are clipped. Values that affect routing, normalization statistics,
  recurrent state history, losses, or final logits remain float32. This is a
  storage quantization applied to the tape, not a change to the model
  definition, so forward/backward arithmetic and gradients differ slightly in
  the last bits. `--gradient-storage bf16` compresses completed parameter
  gradients after float32 accumulation; optimizer state stays float32.
  `fp32/fp32` remains the exact reference path.
- **Reduced-mode nonlinear math.** SiLU, GELU, sigmoid, and softplus use
  float32 transcendentals while activation storage is BF16/INT8; the FP32
  reference path retains float64. The difference is far below the storage
  quantization step and is deterministic. Reduced modes also avoid computing
  and then discarding full derivative arrays in forward: the derivative is
  recomputed in backward, preserving the memory contract.
- **Codec fast paths.** BF16 encoding branches on finiteness: finite narrow
  tensors use a `uint32` RNE path, wider tensors use a `uint16` half-word path
  with rare-tie handling, and non-finite tensors take a payload-preserving
  path. INT8 quantization avoids a tensor-sized boolean mask and the redundant
  clip for computed scales, and dequantization widens then scales in place.
  These changes do not alter the encoded values; they remove temporary arrays
  and passes. On dense-tiny, sequence 512, quantized activations now run
  12-16% faster than the FP32 cache in the benchmark environment, while the
  FP32 path is unchanged.
- **Dust (experimental).** `smaulbrain/dust.py` is an independent NumPy
  activation-perturbation implementation of the Dust estimator. It does not
  call the autograd tape, `Model.loss_and_backward`, or any backward closure.
  It mirrors the paper's direct-loss sites, attention-output targets, local
  attention credit, head slabs, and SGD-with-momentum update. The clean
  forward and from-block reruns are parity-tested against `Model.forward`.
  - **Architecture mapping.** Dense models only. The writer group jitters the
    attention and FFN output projections together (the paper's two `c_proj`
    sites); the hidden group jitters the two FFN input projections together
    (one `c_fc` in the paper); the hub group jitters the attention output
    before `Wo`; embeddings are a scatter-add; the tied embedding/head
    gradients are summed. The paper's untied head and `15*tanh` logit cap are
    absent because SmaulBRAIN has neither, so the head perturbation targets
    raw logits exactly.
  - **Credit.** Direct writer/hub/embedding sites use the current token's
    reward (gamma 0); attention R uses its own token; K and V use future-token
    credit. The future-credit recursion is evaluated in reverse (a
    `gamma**-t` prefix sum is numerically unstable for small gamma and long
    sequences). The head reward is centered over draws because SmaulBRAIN has
    no batch axis; the official code centers over the batch when one exists.
  - **Hyperparameter deviations.** The paper uses sigma 0.2 for
    writers/hidden/embeddings, 0.2/0.4 for hubs, 0.05 for local/head, gamma
    0.98, and populations from 256 to 16k. The CPU default here is sigma 0.05
    for direct sites, 0.05 for local/head, gamma 0.5, and 2-4 draws. The
    smaller sigma was selected by direct-gradient cosine checks at
    SmaulBRAIN's activation scale; gamma 0.5 matches the mean recurrence decay
    (`alpha` ~0.5). The paper's values remain available through the CLI for
    controlled comparisons. Head draws default to 1024, much larger than the
    paper's `4*population` at these tiny populations, because the slab
    evaluation is cheap.
  - **Scalar parameters.** The paper trains only the `2L` residual mixing
    scalars with weight-space ES. SmaulBRAIN's `attn.w`/`attn.u` are per-channel
    vectors and its RMSNorm gains are not residual scalars. All non-matrix
    parameters are frozen by default; `--dust-scalar-draws K` enables a
    two-sided ES fallback over every non-matrix array. That fallback is a
    generalization, not a faithful port, and was not beneficial in the CPU
    tests.
  - **Performance.** Perturbed forwards use float32 nonlinearities while the
    clean forward retains the reference float64 intermediates. Hub/writer/
    hidden groups at each layer are evaluated as one batch with zeros outside
    the current group's sites; this is algebraically identical to three
    separate forwards and is tested bitwise. The head uses an adaptive float32
    row batch (near a 4 MiB transient budget per array) and, on machines with
    more than one CPU, runs concurrently with the recurrent loop. Its cloned
    generator is advanced through the current serial pre-head draw order,
    preserving the stream position and draw budget; a test requires
    byte-identical serial and concurrent gradients. This refactor cut a sequence-64 step from about 1.21 s to
    0.39 s and a sequence-512 step from 10.05 s to 3.35 s on the benchmark
    machine.
  - **Memory.** Dust caches one clean forward, one block's direct/local error
    arrays at a time, the current draw batch, and one gradient dict for the
    trained matrix parameters. It also keeps SGD momentum buffers. It removes
    the autograd tape and activation-gradient storage, but it is not a
    constant-memory trainer. The benchmark reports clean-cache, estimator, and
    temporary bytes separately. In the CPU memory points the clean cache was
    smaller than backprop's activation cache, but peak RSS did not improve; at
    sequence 512 Dust peaked at 118.6 MiB versus 103.8 MiB for backprop because
    the concurrent head pass, draw scratch, and allocator retention dominated.
    The `temp_bytes` metric is not a complete RSS model.
  - **Failure handling.** Clean and perturbed forwards, the head partition,
    each estimated gradient, and the optimizer input are checked for
    non-finite values. `DustSGD` validates every gradient shape and value
    before mutating any parameter. Scalar ES restores the clean parameters in
    a `finally` block. Unsupported configurations raise instead of falling
    back to another algorithm. `--grad-clip` is backprop-only; a non-default
    value in Dust mode is rejected in favor of `--dust-grad-clip` while the
    latter is left at its default 0.
  - **Checkpointing.** The manifest records `algorithm` and a `dust` config
    block. Backprop checkpoints without an `algorithm` field are treated as
    backprop. Loading across Dust/backprop modes is rejected. The perturbation
    RNG is derived from `(dust_seed, global_step)`, so a checkpoint resumes
    bit-exactly without storing generator state. Resume also restores the
    Dust learning rate, momentum, and weight decay from the checkpoint rather
    than from the launching command line. Ctrl+C during a Dust step is blocked
    until the update commits, then the trainer saves.
- **FP8 backward fix.** `_weight_grad` previously called `.data` on both
  operands, which failed when an FP8 backward passed an already-quantized
  ndarray. It now accepts either a `Tensor` or an ndarray; this makes the
  documented `fp8/fp8` mode usable but does not change fp32 math.
- **Backward graph reclamation.** The reverse-mode traversal now uses an
  explicit stack instead of a self-referential nested helper. The old helper
  formed a reference cycle that kept its topological list, and therefore every
  cached activation in the step, alive until a generational GC pass. Peak RSS
  consequently grew with step count. Traversal order is unchanged, so fp32
  losses and gradients are identical; memory is just released at the end of
  each backward call.
- **No auxiliary quantization loss.** Weight quantization (ternary/FP8) is
  applied inside the forward operators. Training optimizes the model
  cross-entropy plus optional MTP cross-entropy; there is no reconstruction or
  weight-quantization loss term in this repository to preserve, reweight, or
  benchmark.
- **Recurrent scan.** The original time loop / native scan is replaced by a
  chunked closed-form scan (`npgrad.scan`). The chunk size adapts to the decay
  range, so summation order differs from a token-by-token loop. The math is the
  same, and the scan is parity-tested against PyTorch.
- **RNG streams.** NumPy `Generator` (PCG64) replaces the PyTorch RNG. Existing
  checkpoints load weights exactly, but freshly seeded runs do not reproduce
  PyTorch's random stream. Checkpoints record and restore the NumPy bit
  generator state, so NumPy runs resume exactly.
- **FP8.** E4M3 is software-emulated with explicit scaling, subnormals,
  saturation at 448, and NaN/Inf sentinels. It is not hardware FP8 arithmetic.
- **Ternary.** Weights use the original rule: threshold
  `0.7 * mean(abs(W))`, codes in {-1, 0, +1}, scale `mean(abs(W))` over
  nonzeros, packed four codes per byte. Packed matmul avoids reconstructing a
  dense weight matrix. The optional C++ kernel and the portable NumPy path use
  the same layout; the fallback is used when no compiler is available.
- **SmaulOPT.** The factored optimizer keeps the original first moment and
  row/column second-moment structure. The optional fused C++ update is reached
  through NumPy buffers; without a compiler or for non-qualifying tensors the
  NumPy reference update runs instead. Rounding can differ from the original
  fused kernel.
- **Interpolation.** `F.interpolate(..., mode="linear", align_corners=False)`
  and `area` / `adaptive_avg_pool1d` are reimplemented and parity-tested; edge
  sampling follows PyTorch's half-pixel coordinates.

## Checkpoint and file-format changes

- New checkpoints are directories containing `manifest.json`,
  `params.safetensors`, and `optim.safetensors`. The version string is
  `smaulbrain-numpy-safetensors-v1:1`. The manifest keeps checksums, model and
  optimizer metadata, data cursor, and NumPy RNG state.
- Activation/gradient storage choices are recorded in the manifest as runtime
  configuration. They do not change the parameter or optimizer payload, so any
  checkpoint loads under any storage setting; resume keeps the storage selected
  on the current command line.
- Dust checkpoints add `algorithm: dust` and a `dust` configuration block
  inside the manifest's `config` object. The optimizer payload is an SGD
  momentum dict keyed by parameter name. A checkpoint from one trainer is
  rejected by the other instead of being partially interpreted.
- Legacy `smaulbrain-v2:1` `params.npz` / `optim.npz` checkpoints remain
  readable; the next save writes the safetensors format.
- Legacy PyTorch `.pt` checkpoints are not read at runtime. Convert them with
  `tests/_reference/convert_torch_checkpoint.py` (development-only; it imports
  PyTorch).
- Paged expert files are `expert_<id>.safetensors`. Legacy per-expert `.npz`
  files are still read; legacy `.pt` expert files must be converted.

## Device and performance behavior

- The build is CPU-only. `--device cuda` is accepted for CLI compatibility but
  executes on CPU; there is no CUDA backend. `paging.accelerator_available()`
  always reports no accelerator, so `V`, `D2V`, and `R2V` paging raise
  `PagingError`.
- Optional C++ kernels are compiled on first use with SSE2/AVX flags appropriate
  for the host. No AVX2-only path is required. If compilation fails, the NumPy
  paths remain functional.
- Full presets require substantial RAM. Use the tiny presets for CPU
  development and benchmark the actual machine before selecting larger models.

## Unsupported items

- `--pagingmthd`, adaptive halting depth, and `--trunk-lr-mult` are not present
  in this repository; they were not invented during the port.
- CUDA execution, CUDA paging, and hardware FP8 are not supported.
- PyTorch RNG continuation is not reproduced; see the RNG note above.
- Dust does not support MoE routing, MTP heads, paged experts, quantized
  forward/backward modes, or non-FP32 activation/gradient storage. These
  combinations raise `ValueError`; the trainer never silently falls back to
  backpropagation.
