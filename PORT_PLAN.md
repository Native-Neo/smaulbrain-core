# Torch → NumPy port plan (smaulbrain-core)

Inventory produced by grepping `torch`, `nn.`, `F.`, `autograd`, `.cuda`, `.to(`,
`DataLoader`, and `torch.save/load` in this repository. Every runtime reference
below is removed; PyTorch survives only in `tests/_reference/` as a dev oracle.

## Replacement layer

| Concept | NumPy replacement |
| --- | --- |
| `torch.Tensor` | `smaulbrain.npgrad.Tensor` (float32, explicit backward closure) |
| `nn.Module` / `nn.Parameter` | `smaulbrain.npgrad.Module` / `Parameter` |
| `torch.Generator` | `numpy.random.Generator` (`npgrad.default_rng`) |
| `torch.save/load` | safetensors writer/reader in `smaulbrain/safetensors_np.py` |
| `torch.cuda` | CPU only; `paging.accelerator_available()` reports no accelerator |
| `DataLoader` | none used; `data.ByteStream` is already streaming |

## File / symbol inventory

### `smaulbrain/ops.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `_tensor` | `torch.as_tensor` | `npgrad.as_tensor` |
| `rmsnorm_fwd` | `square/mean/rsqrt` | `npgrad` ops |
| `rmsnorm_bwd` | tensor reductions | explicit NumPy formula (unchanged math) |
| `sigmoid/silu/gelu_tanh/softplus` | `torch.*`, `F.*` | `npgrad.*` |
| `scatter_add_rows` | `index_add_` | `npgrad.index_add_rows` |
| `RecurrentAttention` | torch tensors + time loop | `npgrad.Tensor` + `npgrad.scan` |

### `smaulbrain/moe.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `Router.__init__` | `torch.Generator`, `randn` | `numpy.random.Generator` |
| `Router.route` | `argsort/gather/softmax/bincount/index_add` | NumPy equivalents; returns NumPy idx/weights/info/cache |
| `Router.backward` | tensor scatter/gemm | NumPy `np.add.at` + matmul |
| `softmax_rows/balance_aux_loss/expert_output_stats` | torch reductions | NumPy reductions |

### `smaulbrain/model.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `build_params` | `torch.randn`, `ParameterList` | `npgrad.Parameter` + `numpy.random.Generator` |
| `PackedCache/FP8Cache` | `.detach()`, `_version`, `weakref` | arrays + explicit invalidate |
| `_QuantLinear` | `torch.autograd.Function` | `npgrad.make_op` custom forward/backward |
| `Model` | `nn.Module`, `F.*`, `.to`, `.cuda`, `.detach()` | `npgrad.Module`, `npgrad` functional |
| `_attn` | Python time loop | `npgrad.scan` (closed-form chunked scan + explicit backward) |
| `_moe` | `argsort/bincount/index_add_` | NumPy routing + `index_add_rows` |
| `loss_and_backward` | `loss.backward()` | `Tensor.backward()` |
| `serialize/restore_states` | `.tolist`, `as_tensor` | lists / `np.asarray` |

### `smaulbrain/quant.py`, `_quant_native.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `_tensor/_np` | torch conversion | array validation / `np.ascontiguousarray` |
| `pack_ternary*`, `unpack*`, `packed_matmat*` | `torch.from_numpy` | return `np.ndarray` / `TiledPacked` |
| `fp8_e4m3_quantize` | torch searchsorted | NumPy searchsorted (already present in `_quant_native`) |
| `sample/apply_ternary_updates` | torch tensors | NumPy int32/int8 arrays |
| `recurrent_scan/backward` | torch bridge | direct NumPy buffers |

### `smaulbrain/paging.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `accelerator_available` | `torch.cuda` | CPU-only capability report |
| `ExpertStore` | `torch.save/load`, `.numel/.element_size`, `.copy_` | safetensors expert files + NumPy copies (legacy `.npz` readable) |

### `smaulbrain/inference.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| RNG | `torch.Generator`, `manual_seed` | `numpy.random.Generator` |
| sampling | `softmax/multinomial/argmax/topk/sort` | NumPy equivalents |
| tensors | `torch.stack/tensor/as_tensor` | `np.stack/asarray` |
| speculation | `.detach().clone()` | `.copy()` |

### `smaulbrain/train_loop.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| RNG | `torch.Generator.get_state/set_state` | NumPy bit-generator state |
| checkpointing | `torch.save` mirrors | safetensors mirrors |
| grad norm/clip | `torch.float64`, `mul_` | NumPy float64 reductions, in-place scale |
| optimizer state resize | `torch.zeros_like`, `.clone` | `np.zeros_like`, `.copy` |

### `smaulbrain/smaulopt.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `_init_state` | `torch.zeros_like` | `np.zeros_like` |
| `step` | in-place tensor ops, `.item()` | in-place NumPy ops (same float order) |
| `state_dict/load_state_dict` | `.clone`, `torch.as_tensor` | `.copy`, `np.asarray` |

### `smaulbrain/checkpoint.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `_cpu_tree` | `.detach().cpu()` | `np.ascontiguousarray` |
| `_write_tree` | `torch.save` | `params.safetensors` + `optim.safetensors` |
| `load_checkpoint` | `torch.load` | safetensors reader; legacy `.npz` still accepted |
| `_json_default` | torch tensor | NumPy array |

### `smaulbrain/growth.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `eval_loss_on_tokens*` | `F.cross_entropy` | `npgrad.cross_entropy` |
| `clone_with_mutation` | `torch.Generator/randn` | NumPy generator / `standard_normal` |
| `grow_candidate` | `torch.empty`, tensor slicing | `np.empty`, array slicing |
| `validate_candidate` | `torch.isfinite` | `np.isfinite` |

### `smaulbrain/data.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `synth_bytes` | `torch.Generator/randint` | `numpy.random.Generator.integers` |

### `smaulbrain/_native_opt.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `factored_step` | Tensor view bridge | directly call `_native_opt_numpy.factored_step` |

### `smaulbrain/prune_cmd.py`
| Symbol | Torch usage | NumPy replacement |
| --- | --- | --- |
| `prune_checkpoint` | `torch.isfinite`, `.clone()` | `np.isfinite`, `.copy()` |

### Entry points / benchmarks
`tests/*` (all torch-based) move to `tests/_reference/`. Runtime benchmarks are
ported to NumPy. No `DataLoader` usage exists.

## Checkpoint compatibility

* New format: directory with `manifest.json`, `params.safetensors`,
  `optim.safetensors`; checksums stay in the manifest.
* Legacy `smaulbrain-v2:1` `.npz` checkpoints remain readable.
* Legacy PyTorch `.pt` checkpoints are converted by
  `tests/_reference/convert_torch_checkpoint.py` (dev-only, imports torch).

## Numerical notes

* fp32 everywhere; reductions use float64 where the original used
  `dtype=torch.float64` / `.double()`.
* PyTorch RNG streams are not reproduced. Weight *values* are bit-identical
  when loaded from an existing checkpoint; freshly seeded runs differ.
* `torch.nn.functional.interpolate` corner handling is reimplemented explicitly
  and covered by parity tests.

## Open questions

1. `--pagingmthd`, adaptive halting depth, and `--trunk-lr-mult` are not present
   in either repository; no implementation is invented here.
2. `D2V`/`R2V`/`V` paging modes require CUDA in the original. CPU-only behavior
   remains a raised `PagingError`; only `R`, `D2R` (and `D2V` rejected) exist.
