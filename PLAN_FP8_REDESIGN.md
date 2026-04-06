# FlatQuant + Trainium2 FP8 Architecture Redesign

> Saved 2026-04-03. Start a new Claude Code session and say "implement the FP8 redesign plan" — context is in memory.

## Decision

- **Approach B**: Reparameterize weights offline + NxDI native FP8 quantization
- **Calibration**: Keep existing INT4/INT8 targets (no changes needed)
- No runtime activation transforms. No custom NxDI model code.

## Pipeline

```
Calibrate (learn transforms T, offline)
  -> Reparameterize (W' = W @ T^{-1}, save float16 HF checkpoint)
    -> NxDI FP8 quantize (float16 -> FP8 E4M3 per-channel, via save_quantized_state_dict)
      -> NxDI compile (FP8 kernels on Trainium2, tp_degree=2)
        -> Inference
```

## Implementation Steps (4 steps)

### 1. Fix v_proj reparameterization bug
File: `agent-workflow/tools/flatquant_trainium_unified.py`, `convert_flatquant_checkpoint()` ~line 486
- v_proj needs vcache_trans applied to output dimension (currently missing)
- This fixes "TABLE TABLE TABLE" output

### 2. Update NeuronConfig for FP8
File: `agent-workflow/tools/flatquant_trainium_unified.py`, `compile_with_nxdi()` ~line 600
- Set `quantized=True`, `quantization_dtype="f8e4m3"`, `quantized_mlp_kernel_enabled=True`
- Set env vars: `XLA_HANDLE_SPECIAL_SCALAR=1`, `UNSAFE_FP8FNCAST=1`

### 3. Add FP8 quantization step
Use `NeuronLlamaForCausalLM.save_quantized_state_dict(reparam_path, config)` after reparameterization
- NxDI handles FP8 E4M3 per-channel quantization internally

### 4. Update pipeline CLI flow
Add `--fp8` flag, wire up the new quantization step

## Full details in Claude Code memory
All code snippets, file paths, NxDI API details, and verification steps are saved in:
`~/.claude/projects/-home-ubuntu-flatquant-trainium/memory/project_fp8_redesign_plan.md`
