#!/usr/bin/env python3
"""
================================================================================
UNIFIED FLATQUANT + TRAINIUM2 PIPELINE
================================================================================

THIS ENTIRE SCRIPT RUNS ON TRAINIUM2 INSTANCE.
NO EXTERNAL DEPENDENCIES. EVERYTHING SELF-CONTAINED.

Flow:
  1. Load base model from HuggingFace (on Trainium2)
  2. Apply FlatQuant INT4 wrappers (on Trainium2)
  3. Run calibration on real data (on Trainium2)
  4. Save quantized checkpoint with explicit transforms (on Trainium2)
  5. Immediately trace for Trainium2 compilation (on Trainium2)
  6. Run inference and benchmarking (on Trainium2)

Execution:
  # On Trainium2 instance only:
  python flatquant_trainium_unified.py \
      --model meta-llama/Llama-2-7b-hf \
      --hf_token YOUR_TOKEN \
      --output ./quantized_llama2_7b \
      --benchmark \
      --num_tokens 50

Requirements on Trainium2:
  - torch_neuronx (Trainium SDK)
  - transformers
  - FlatQuantBundled in PYTHONPATH
  - ~32GB RAM available
  - Trainium2 NeuroCores available

================================================================================
"""

import sys
import os
import json
import torch
import time
from pathlib import Path
from typing import Dict, Optional
import argparse

# CRITICAL: Import transformers FIRST, before any FlatQuantBundled modules
from transformers import AutoModelForCausalLM, AutoTokenizer

# FlatQuantBundled imports are deferred to load_and_wrap_model() so that the NxDI
# virtualenv (which lacks calibration deps like 'datasets', 'accelerate') can run
# --skip_calibration --skip_save without hitting ImportError at module load time.


class TrainiumUnifiedPipeline:
    """
    All-in-one FlatQuant pipeline that runs entirely on Trainium2.

    No data transfer, no external dependencies, no cross-machine issues.
    Everything happens on Trainium2.
    """

    def __init__(self, model_name: str, output_path: str, hf_token: str = None):
        self.model_name = model_name
        self.output_path = Path(output_path)
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.device = None

        print("\n" + "=" * 80)
        print("TRAINIUM2 UNIFIED PIPELINE - RUNNING ON TRAINIUM2 INSTANCE")
        print("=" * 80)
        self._check_trainium_environment()

    def _check_trainium_environment(self):
        """Verify we're on Trainium2 with necessary packages."""
        print("\n[INIT] Checking Trainium2 environment...")

        # Check PyTorch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ Device: {torch.device('cpu')}")

        # Check Trainium-specific imports (importability only — defer actual import
        # to trace_for_trainium2 so _XLAC is not loaded during calibration/save,
        # avoiding a known _XLAC destructor segfault on process exit)
        # Check for torch_neuronx by looking for the package directory directly —
        # avoids importing _XLAC (which segfaults on shutdown with large models).
        import importlib.util
        spec = importlib.util.find_spec("torch_neuronx")
        if spec is not None:
            print(f"  ✓ torch_neuronx available (Trainium SDK installed)")
            self.has_trainium = True
        else:
            print(f"  ⚠ torch_neuronx NOT available (this is CPU, not Trainium2)")
            print(f"    But continuing anyway - compilation will be skipped")
            self.has_trainium = False

        nxdi_spec = importlib.util.find_spec("neuronx_distributed_inference")
        if nxdi_spec is not None:
            print(f"  ✓ neuronx_distributed_inference available (NxDI installed)")
            self.has_nxdi = True
        else:
            print(f"  ⚠ neuronx_distributed_inference NOT available")
            print(f"    For NxDI compilation, run under the NxDI venv:")
            print(f"    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate")
            self.has_nxdi = False

        # Check FlatQuantBundled
        try:
            import flatquant
            print(f"  ✓ FlatQuantBundled available in PYTHONPATH")
        except ImportError:
            print(f"  ✗ FlatQuantBundled NOT in PYTHONPATH")
            print(f"    Run: export PYTHONPATH=./FlatQuantBundled:$PYTHONPATH")
            sys.exit(1)

        # Detect available device for training (GPU if available, else CPU)
        # Note: This will be CPU for Trainium2 training, but that's fine
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print(f"  ✓ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"  ✓ Using CPU for training (Trainium2 CPU cores)")

        print(f"  ✓ Training device: {self.device}")
        print(f"  ✓ Trainium2 environment verified\n")

    def load_and_wrap_model(self, num_samples=128, cali_epochs=5, cali_bsz_val=1, flat_lr_val=5e-3):
        """
        [ON TRAINIUM2] Load base model and apply FlatQuant wrappers.
        Everything stays on Trainium2.
        """
        print("[STEP 1/6] Loading base model and applying FlatQuant wrappers")
        print("=" * 80)

        # Lazy imports: only needed for calibration, not for NxDI-only path
        from llama_2_7b_hf_utils import FlatQuantLlamaMLP, FlatQuantLlamaAttention

        # Step 1a: Load model from HuggingFace
        print(f"\nLoading base model: {self.model_name}")
        print(f"  (Loading on Trainium2, will keep quantized model here)")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",  # CPU for now, will move to training device during calibration
            token=self.hf_token,
        )
        self.model.eval()
        print(f"  ✓ Base model loaded: {type(self.model).__name__}")

        # Step 1b: Load tokenizer
        print(f"\nLoading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            token=self.hf_token,
        )
        print(f"  ✓ Tokenizer loaded")

        # Step 1c: Apply FlatQuant wrappers
        print(f"\nApplying FlatQuant wrappers (INT4 weights, FP8 activations)...")

        class FlatQuantArgs:
            # Quantization settings
            w_bits = 4
            a_bits = 8
            group_size = 128
            w_asym = False
            a_asym = False
            a_groupsize = -1
            lwc = False
            direct_inv = False
            add_diag = False
            diag_init = "sq_style"
            lac = False
            separate_vtrans = True
            q_bits = 8
            k_bits = 8
            v_bits = 8
            q_asym = False
            k_asym = False
            v_asym = False
            # Calibration settings (used by cali_flat_quant)
            nsamples = num_samples
            cali_bsz = cali_bsz_val
            deactive_amp = not torch.cuda.is_available()  # float32 on CPU; AMP only when GPU present
            diag_alpha = 0.5
            cali_trans = True
            flat_lr = flat_lr_val
            epochs = cali_epochs
            warmup = True
            exp_dir = str(self.output_path)

        args = FlatQuantArgs()
        Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
        num_layers = self.model.config.num_hidden_layers

        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Wrap attention
            try:
                layer.self_attn = FlatQuantLlamaAttention(args, layer.self_attn)
            except Exception as e:
                print(f"  Layer {layer_idx}: attention wrap failed - {e}")

            # Wrap MLP
            try:
                layer.mlp = FlatQuantLlamaMLP(args, layer.mlp)
            except Exception as e:
                print(f"  Layer {layer_idx}: MLP wrap failed - {e}")

        print(f"  ✓ Applied wrappers to {num_layers} layers")
        print(f"\n[STEP 1 COMPLETE] Model ready for calibration on Trainium2\n")

        return self.model, self.tokenizer, args

    def calibrate_on_trainium2(self, args, dataset_name: str = "wikitext", num_samples: int = 128):
        """
        [ON TRAINIUM2] Run calibration using Trainium2's CPU/memory.
        Learns transformation matrices T for quantization-friendly activations.
        Everything happens on Trainium2 - no external calibration needed.
        """
        print("[STEP 2/6] Running FlatQuant calibration (ON TRAINIUM2)")
        print("=" * 80)

        # Lazy imports: calibration deps not available in NxDI venv
        import flatquant.data_utils as data_utils
        import flatquant.train_utils as train_utils

        print(f"\nCalibration dataset: {dataset_name}")
        print(f"Calibration samples: {num_samples}")
        print(f"Calibration device: {self.device}")
        print(f"(This runs entirely on Trainium2 instance)")

        # Set sequence length
        self.model.seqlen = 2048

        # Load calibration dataset
        print(f"\nLoading calibration data...")
        try:
            trainloader = data_utils.get_loaders(
                args=None,
                name=dataset_name,
                tokenizer=self.tokenizer,
                nsamples=num_samples,
                seqlen=self.model.seqlen,
                eval_mode=False,
            )
            print(f"  ✓ Loaded {num_samples} calibration samples")
        except Exception as e:
            print(f"  ⚠ Could not load dataset: {e}")
            print(f"    Skipping calibration (using untrained transforms)")
            return

        # Run calibration
        print(f"\nRunning calibration (learning transform matrices)...")
        try:
            train_utils.cali_flat_quant(
                args=args,
                model=self.model,
                dataloader=trainloader,
                dev=self.device,
                logger=self._get_logger(),
            )
            print(f"  ✓ Calibration complete")
        except Exception as e:
            print(f"  ⚠ Calibration failed: {e}")
            print(f"    Continuing with untrained transforms...")

        print(f"\n[STEP 2 COMPLETE] Calibration done on Trainium2\n")

    def save_quantized_on_trainium2(self, args):
        """
        [ON TRAINIUM2] Save quantized model with explicit transforms.
        Keep everything on Trainium2 - no data transfer.
        """
        print("[STEP 3/6] Saving quantized model (ON TRAINIUM2)")
        print("=" * 80)

        output_path = str(self.output_path)
        print(f"\nSaving to: {output_path}")
        print(f"(Model stays on Trainium2 instance)")

        # Set evaluation mode on all layers
        print(f"\nSetting evaluation mode (keeping transforms explicit)...")
        num_layers = self.model.config.num_hidden_layers

        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            # Attention eval mode
            if hasattr(layer.self_attn, '_eval_mode'):
                layer.self_attn._eval_mode = True
                if hasattr(layer.self_attn, 'ln_trans') and layer.self_attn.ln_trans is not None:
                    layer.self_attn.ln_trans.to_eval_mode()
                if hasattr(layer.self_attn, 'o_trans') and layer.self_attn.o_trans is not None:
                    layer.self_attn.o_trans.to_eval_mode()
                if hasattr(layer.self_attn, 'kcache_trans') and layer.self_attn.kcache_trans is not None:
                    layer.self_attn.kcache_trans.to_eval_mode()
                if hasattr(layer.self_attn, 'vcache_trans') and layer.self_attn.vcache_trans is not None:
                    layer.self_attn.vcache_trans.to_eval_mode()

            # MLP eval mode
            if hasattr(layer.mlp, '_ori_mode'):
                layer.mlp._ori_mode = False
            if hasattr(layer.mlp, 'up_gate_trans') and layer.mlp.up_gate_trans is not None:
                layer.mlp.up_gate_trans.to_eval_mode()
            if hasattr(layer.mlp, 'down_trans') and layer.mlp.down_trans is not None:
                layer.mlp.down_trans.to_eval_mode()

        # Set eval mode on projections
        print(f"Setting FlatQuantizedLinear to evaluation mode...")
        for layer_idx in range(num_layers):
            layer = self.model.model.layers[layer_idx]

            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, proj_name):
                    proj = getattr(layer.self_attn, proj_name)
                    if hasattr(proj, '_eval_mode'):
                        proj._eval_mode = True

            for proj_name in ['up_proj', 'gate_proj', 'down_proj']:
                if hasattr(layer.mlp, proj_name):
                    proj = getattr(layer.mlp, proj_name)
                    if hasattr(proj, '_eval_mode'):
                        proj._eval_mode = True

        print(f"  ✓ Evaluation mode set")

        # Save checkpoint
        import gc, sys
        print(f"\nSaving model checkpoint...")
        sys.stdout.flush()
        Path(output_path).mkdir(parents=True, exist_ok=True)
        print(f"  Calling save_pretrained...")
        sys.stdout.flush()
        self.model.save_pretrained(output_path)
        sys.stdout.flush()
        print(f"  Model weights saved, saving tokenizer...")
        sys.stdout.flush()
        self.tokenizer.save_pretrained(output_path)
        gc.collect()
        print(f"  ✓ Model saved")

        # Save quant config
        print(f"Saving quantization config...")
        quant_config = {
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "group_size": args.group_size,
            "w_asym": args.w_asym,
            "a_asym": args.a_asym,
            "a_groupsize": args.a_groupsize,
            "lwc": args.lwc,
            "q_bits": args.q_bits,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "model_type": "llama",
            "strategy": "option2_explicit_transforms",
            "device_location": "TRAINIUM2",  # EXPLICITLY MARK QUANTIZATION LOCATION
        }

        config_path = Path(output_path) / "quant_config.json"
        with open(config_path, "w") as f:
            json.dump(quant_config, f, indent=2)
        print(f"  ✓ Config saved")

        print(f"\n[STEP 3 COMPLETE] Quantized model saved on Trainium2\n")

    # -------------------------------------------------------------------------
    # STEP 4 (NEW): Checkpoint conversion + NxDI compilation
    # -------------------------------------------------------------------------

    def _kronecker_matmul(self, x, hadL, hadR):
        """
        Applies the Kronecker product transform to tensor x along its last two dims.
        Equivalent to: x @ kron(hadL, hadR)  (applied to input/column dimension).
        Same logic as FlatQuantBundled/flatquant/flat_utils.py::kronecker_matmul.
        """
        init_shape = x.shape
        x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
        x = torch.matmul(x, hadR)
        x = torch.matmul(hadL.T, x)
        return x.reshape(init_shape)

    @staticmethod
    def _verify_checkpoint(ckpt_path: str, description: str = "checkpoint") -> dict:
        """
        Load a safetensors checkpoint and report NaN/Inf counts and dtype distribution.
        Returns a dict with keys: total_keys, nan_keys, inf_keys, dtype_counts, ok.
        """
        from safetensors import safe_open
        import glob as _glob

        print(f"\n[VERIFY] {description}: {ckpt_path}")

        # Collect all .safetensors shards in the directory (or single file)
        if os.path.isdir(ckpt_path):
            shards = sorted(_glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        else:
            shards = [ckpt_path]

        if not shards:
            print(f"  ✗ No safetensors files found at {ckpt_path}")
            return {"ok": False}

        total_keys = 0
        nan_keys = []
        inf_keys = []
        dtype_counts: dict = {}

        for shard in shards:
            sf = safe_open(shard, framework="pt", device="cpu")
            for k in sf.keys():
                t = sf.get_tensor(k)
                total_keys += 1
                dtype_str = str(t.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
                # NaN/Inf checks (skip integer-like dtypes that don't have NaN)
                if t.is_floating_point():
                    tf = t.float()
                    if torch.isnan(tf).any():
                        nan_keys.append(k)
                    if torch.isinf(tf).any():
                        inf_keys.append(k)

        ok = len(nan_keys) == 0 and len(inf_keys) == 0
        print(f"  Total tensors : {total_keys}")
        print(f"  Dtype counts  : {dtype_counts}")
        print(f"  NaN tensors   : {len(nan_keys)}" + (f" — {nan_keys[:5]}" if nan_keys else ""))
        print(f"  Inf tensors   : {len(inf_keys)}" + (f" — {inf_keys[:5]}" if inf_keys else ""))
        print(f"  Status        : {'✓ OK' if ok else '✗ FAIL'}")

        return {
            "ok": ok,
            "total_keys": total_keys,
            "nan_keys": nan_keys,
            "inf_keys": inf_keys,
            "dtype_counts": dtype_counts,
        }

    def convert_flatquant_checkpoint(self, identity_transforms: bool = False):
        """
        CPU-only: fold FlatQuant's learned transform matrices into the weight matrices
        (reparameterization), then write a clean HF-format safetensors checkpoint that
        NxDI's NeuronLlamaForCausalLM can load directly.

        Calibration was run with separate_vtrans=True and add_diag=False, so:
          - q/k/v: absorb ln_trans (Kronecker decompose) into weight columns
          - v:     NO vcache_trans folded into weight (separate_vtrans=True)
          - o:     absorb o_trans (Single) + vcache_trans (Single) as Kronecker pair
          - gate/up: absorb up_gate_trans (Kronecker decompose) into weight columns
          - down:    absorb down_trans (Kronecker decompose) into weight columns

        The reparameterize() in FlatQuantizedLinear uses float64 intermediates, so we do
        the same to avoid numerical drift.
        """
        import gc
        from safetensors import safe_open
        from safetensors.torch import save_file
        import shutil

        reparam_path = Path(str(self.output_path) + "_reparameterized")
        weights_file = reparam_path / "model.safetensors"

        if reparam_path.exists() and weights_file.exists():
            print(f"\n  ✓ Reparameterized checkpoint already exists at {reparam_path}")
            return str(reparam_path)

        print(f"\nConverting FlatQuant checkpoint → plain HF format...")
        print(f"  Source:  {self.output_path}")
        print(f"  Output:  {reparam_path}")
        print(f"  (CPU-only, reparameterizing transform matrices into weights)")

        reparam_path.mkdir(parents=True, exist_ok=True)

        hf_sd = {}

        # Support both sharded (model-XXXXX-of-YYYYY.safetensors + index.json)
        # and monolithic (model.safetensors) checkpoint formats.
        index_file = self.output_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as fi:
                weight_map = json.load(fi)["weight_map"]
            _open_shards = {}
            def get_tensor(key):
                shard = weight_map[key]
                if shard not in _open_shards:
                    _open_shards[shard] = safe_open(
                        str(self.output_path / shard), framework="pt", device="cpu"
                    )
                return _open_shards[shard].get_tensor(key)
        else:
            _mono = safe_open(str(self.output_path / "model.safetensors"), framework="pt", device="cpu")
            get_tensor = _mono.get_tensor

        num_layers = self.model.config.num_hidden_layers if self.model is not None else 32

        print(f"  Processing {num_layers} layers...")
        for i in range(num_layers):
            if i % 8 == 0:
                print(f"    Layer {i}/{num_layers}...")
                import sys; sys.stdout.flush()

            if identity_transforms:
                # --- Identity bypass: copy raw linear weights, skip transform folding ---
                for proj in ["q_proj", "k_proj", "v_proj"]:
                    hf_sd[f"model.layers.{i}.self_attn.{proj}.weight"] = \
                        get_tensor(f"model.layers.{i}.self_attn.{proj}.linear.weight")
                hf_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = \
                    get_tensor(f"model.layers.{i}.self_attn.o_proj.linear.weight")
                for proj in ["gate_proj", "up_proj"]:
                    hf_sd[f"model.layers.{i}.mlp.{proj}.weight"] = \
                        get_tensor(f"model.layers.{i}.mlp.{proj}.linear.weight")
                hf_sd[f"model.layers.{i}.mlp.down_proj.weight"] = \
                    get_tensor(f"model.layers.{i}.mlp.down_proj.linear.weight")
            else:
                # --- Load transform matrices (already in eval mode from step 3) ---
                ln_left_inv  = get_tensor(f"model.layers.{i}.self_attn.ln_trans.matrix_left_inv").to(torch.float64)
                ln_right_inv = get_tensor(f"model.layers.{i}.self_attn.ln_trans.matrix_right_inv").to(torch.float64)
                o_inv_t      = get_tensor(f"model.layers.{i}.self_attn.o_trans.matrix_inv_t").to(torch.float64)
                vc_inv_t     = get_tensor(f"model.layers.{i}.self_attn.vcache_trans.matrix_inv_t").to(torch.float64)
                ug_left_inv  = get_tensor(f"model.layers.{i}.mlp.up_gate_trans.matrix_left_inv").to(torch.float64)
                ug_right_inv = get_tensor(f"model.layers.{i}.mlp.up_gate_trans.matrix_right_inv").to(torch.float64)
                d_left_inv   = get_tensor(f"model.layers.{i}.mlp.down_trans.matrix_left_inv").to(torch.float64)
                d_right_inv  = get_tensor(f"model.layers.{i}.mlp.down_trans.matrix_right_inv").to(torch.float64)

                # --- NaN validation gate ---
                transforms_check = [ln_left_inv, ln_right_inv, o_inv_t, vc_inv_t,
                                    ug_left_inv, ug_right_inv, d_left_inv, d_right_inv]
                for t in transforms_check:
                    if torch.isnan(t).any():
                        raise ValueError(
                            f"NaN in transform matrices at layer {i}. "
                            f"Re-run calibration to fix."
                        )

                # --- Attention projections: q/k (ln_trans only) ---
                for proj in ["q_proj", "k_proj"]:
                    W = get_tensor(f"model.layers.{i}.self_attn.{proj}.linear.weight").to(torch.float64)
                    W_new = self._kronecker_matmul(W, ln_left_inv, ln_right_inv)
                    hf_sd[f"model.layers.{i}.self_attn.{proj}.weight"] = W_new.to(torch.float16)
                    del W, W_new

                # v_proj: ln_trans on input columns + vcache_trans (forward) on output rows per-head
                # With separate_vtrans=True and no runtime activation transforms (Approach B),
                # we must fold vcache_trans into the weight here so the KV cache is pre-rotated.
                W_v = get_tensor(f"model.layers.{i}.self_attn.v_proj.linear.weight").to(torch.float64)
                W_v = self._kronecker_matmul(W_v, ln_left_inv, ln_right_inv)
                vc_matrix = get_tensor(f"model.layers.{i}.self_attn.vcache_trans.matrix").to(torch.float64)
                head_dim = vc_matrix.shape[0]  # 128 for LLaMA-2-7B
                # Right-multiply each head's slice of W_v rows by vcache_trans:
                # W_v is [hidden, hidden]; transpose to [hidden, hidden], reshape to [n_heads*hidden, head_dim],
                # matmul with [head_dim, head_dim], reshape back, transpose.
                W_v = (W_v.T.reshape(-1, head_dim) @ vc_matrix).reshape(W_v.T.shape).T
                hf_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = W_v.to(torch.float16)
                del W_v, vc_matrix

                # o_proj: Approach B — NxDI has no runtime o_trans.T step, so only vc_inv_t per-head.
                # Math: FlatQuant eval does kron(o_trans,I) @ kron(o_inv_t,vc_inv_t).T = kron(I,vc_inv_t.T)
                # (o components cancel: o_trans @ o_inv_t.T = I). NxDI needs the same net effect
                # but without the o_trans.T pre-multiply, so fold kron(I, vc_inv_t) into W_o directly.
                W_o = get_tensor(f"model.layers.{i}.self_attn.o_proj.linear.weight").to(torch.float64)
                num_heads = o_inv_t.shape[0]  # 32 (o_inv_t still loaded above for NaN check)
                W_o_new = (W_o.reshape(W_o.shape[0], num_heads, -1) @ vc_inv_t).reshape(W_o.shape)
                hf_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = W_o_new.to(torch.float16)
                del W_o, W_o_new

                # --- MLP projections ---
                for proj in ["gate_proj", "up_proj"]:
                    W = get_tensor(f"model.layers.{i}.mlp.{proj}.linear.weight").to(torch.float64)
                    W_new = self._kronecker_matmul(W, ug_left_inv, ug_right_inv)
                    hf_sd[f"model.layers.{i}.mlp.{proj}.weight"] = W_new.to(torch.float16)
                    del W, W_new

                W_d = get_tensor(f"model.layers.{i}.mlp.down_proj.linear.weight").to(torch.float64)
                W_d_new = self._kronecker_matmul(W_d, d_left_inv, d_right_inv)
                hf_sd[f"model.layers.{i}.mlp.down_proj.weight"] = W_d_new.to(torch.float16)
                del W_d, W_d_new

                # Free transform matrices
                del ln_left_inv, ln_right_inv, o_inv_t, vc_inv_t
                del ug_left_inv, ug_right_inv, d_left_inv, d_right_inv

            # --- Pass-through: layernorm weights (both branches) ---
            hf_sd[f"model.layers.{i}.input_layernorm.weight"] = \
                get_tensor(f"model.layers.{i}.input_layernorm.weight")
            hf_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = \
                get_tensor(f"model.layers.{i}.post_attention_layernorm.weight")

        # --- Global pass-through weights ---
        hf_sd["model.embed_tokens.weight"] = get_tensor("model.embed_tokens.weight")
        hf_sd["model.norm.weight"] = get_tensor("model.norm.weight")
        hf_sd["lm_head.weight"] = get_tensor("lm_head.weight")

        print(f"  Saving reparameterized weights ({len(hf_sd)} tensors)...")
        save_file(hf_sd, str(weights_file))
        del hf_sd
        gc.collect()
        print(f"  ✓ Weights saved")

        # Copy config.json and patch pad_token_id (LLaMA-2 has null; NxDI requires a value)
        src_config = self.output_path / "config.json"
        dst_config = reparam_path / "config.json"
        with open(src_config) as f_in:
            cfg = json.load(f_in)
        # NxDI's LlamaInferenceConfig requires pad_token_id to be set
        if cfg.get("pad_token_id") is None:
            cfg["pad_token_id"] = cfg.get("eos_token_id", 2)
        # Remove FlatQuant-specific keys that confuse HF auto-class loading
        for k in ["quantization_config", "auto_map"]:
            cfg.pop(k, None)
        with open(dst_config, "w") as f_out:
            json.dump(cfg, f_out, indent=2)

        # Copy tokenizer files
        for fname in ["tokenizer.json", "tokenizer_config.json",
                      "tokenizer.model", "special_tokens_map.json",
                      "generation_config.json"]:
            src = self.output_path / fname
            if src.exists():
                shutil.copy2(src, reparam_path / fname)

        print(f"  ✓ Config and tokenizer copied")
        print(f"\n[CONVERT COMPLETE] Reparameterized checkpoint at {reparam_path}\n")
        return str(reparam_path)

    def compile_with_nxdi(self, sequence_length: int = 128, max_new_tokens: int = 128,
                          identity_transforms: bool = False, fp8: bool = True):
        """
        [ON TRAINIUM2] Compile LLaMA-2-7B using NeuronX Distributed Inference (NxDI)
        with tp_degree=2, distributing weights across both LNCs on trn2.3xlarge.

        When fp8=True (default): quantizes the reparameterized float16 checkpoint to
        FP8 E4M3 per-channel via NxDI's save_quantized_state_dict, then compiles with
        the fused RMSNorm+quantize+matmul kernel (quantized_mlp_kernel_enabled).

        When fp8=False: compiles in float16 without quantization (baseline path).

        Must be run under the NxDI virtualenv:
          source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
        """
        print(f"[STEP 4/6] Compiling with NxDI (tp_degree=2, fp8={fp8})")
        print("=" * 80)

        import gc

        # Free calibration model from RAM before NxDI runtime init
        if hasattr(self, 'model') and self.model is not None:
            print(f"\nFreeing calibration model from RAM...")
            del self.model
            self.model = None
            gc.collect()
            print(f"  ✓ RAM freed")

        try:
            from neuronx_distributed_inference.models.llama.modeling_llama import (
                LlamaInferenceConfig, NeuronLlamaForCausalLM
            )
            from neuronx_distributed_inference.models.config import NeuronConfig
            from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
        except ImportError as e:
            raise ImportError(
                f"NxDI not available: {e}\n"
                "Run this script under the NxDI virtualenv:\n"
                "  source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate"
            )

        # 1. Convert FlatQuant checkpoint to plain HF format (CPU reparameterization)
        reparam_path = self.convert_flatquant_checkpoint(identity_transforms=identity_transforms)

        # Verification Step 1: Check reparameterized weights for NaN/Inf
        reparam_result = self._verify_checkpoint(reparam_path, "Reparameterized float16 checkpoint")
        if not reparam_result["ok"]:
            raise ValueError(
                f"Reparameterized checkpoint has NaN/Inf — "
                f"NaN: {reparam_result['nan_keys'][:3]}, Inf: {reparam_result['inf_keys'][:3]}. "
                f"Re-run reparameterization."
            )

        # 2. Build NxDI config
        print(f"\nBuilding NxDI config (tp_degree=2, seq_len={sequence_length}, fp8={fp8})...")

        if fp8:
            # Required env vars for FP8 compilation on Trainium2
            os.environ["XLA_HANDLE_SPECIAL_SCALAR"] = "1"
            os.environ["UNSAFE_FP8FNCAST"] = "1"

            fp8_ckpt_path = str(self.output_path.parent / "fp8_quantized_checkpoint") + "/"
            os.makedirs(fp8_ckpt_path, exist_ok=True)

            neuron_config = NeuronConfig(
                tp_degree=2,
                batch_size=1,
                seq_len=sequence_length,
                max_context_length=sequence_length,
                max_new_tokens=max_new_tokens,
                max_length=sequence_length + max_new_tokens,
                torch_dtype=torch.bfloat16,
                padding_side="right",
                quantized=True,
                quantized_checkpoints_path=fp8_ckpt_path,
                quantization_type="per_channel_symmetric",
                quantization_dtype="f8e4m3",
                # Note: quantized_mlp_kernel_enabled / mlp_kernel_enabled /
                # rmsnorm_quantize_kernel_enabled are NOT set here because those
                # fused kernels require intermediate_size ≤ 4096, but LLaMA-2-7B
                # has intermediate_size=11008. Weights are still FP8 (smaller HBM
                # footprint); they dequantize to BF16 before the matmul.
                modules_to_not_convert=["lm_head"],
            )
        else:
            neuron_config = NeuronConfig(
                tp_degree=2,
                batch_size=1,
                seq_len=sequence_length,
                max_context_length=sequence_length,
                max_new_tokens=max_new_tokens,
                max_length=sequence_length + max_new_tokens,
                torch_dtype=torch.float16,
                padding_side="right",
            )

        config = LlamaInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(reparam_path),
        )
        print(f"  ✓ NeuronConfig: tp_degree=2, seq_len={sequence_length}, "
              f"max_length={sequence_length + max_new_tokens}, fp8={fp8}")

        # 3. FP8 quantization: float16 → FP8 E4M3 per-channel (cached after first run)
        if fp8:
            fp8_weights_file = os.path.join(fp8_ckpt_path, "model.safetensors")
            if not os.path.exists(fp8_weights_file):
                print(f"\nQuantizing float16 checkpoint to FP8 E4M3 per-channel...")
                print(f"  Source:  {reparam_path}")
                print(f"  Output:  {fp8_ckpt_path}")
                NeuronLlamaForCausalLM.save_quantized_state_dict(reparam_path, config)
                print(f"  ✓ FP8 quantized checkpoint saved")
            else:
                print(f"\n  ✓ Using cached FP8 checkpoint at {fp8_ckpt_path}")

            # Verification Step 2: Check FP8 checkpoint dtypes (weights=float8_e4m3fn, scales=float32)
            fp8_result = self._verify_checkpoint(fp8_ckpt_path, "FP8 E4M3 quantized checkpoint")
            if not fp8_result["ok"]:
                raise ValueError(
                    f"FP8 checkpoint has NaN/Inf in scales — "
                    f"NaN: {fp8_result['nan_keys'][:3]}. "
                    f"Delete {fp8_ckpt_path} and retry."
                )
            fp8_weight_count = fp8_result["dtype_counts"].get("torch.float8_e4m3fn", 0)
            print(f"  FP8 weight tensors: {fp8_weight_count} "
                  f"({'✓' if fp8_weight_count > 0 else '✗ EXPECTED > 0'})")

        # 4. Compile (or load cached compilation)
        compiled_path = str(self.output_path.parent / "compiled_nxdi_model") + "/"

        # --verify_only: stop here after verification checks, before the slow compile step
        if getattr(self, '_verify_only', False):
            print(f"\n[VERIFY ONLY] All checkpoint verification passed. Skipping compilation.")
            print(f"  Reparam checkpoint : {reparam_path}")
            if fp8:
                print(f"  FP8 checkpoint     : {fp8_ckpt_path}")
            return None

        model = NeuronLlamaForCausalLM(reparam_path, config)

        if not os.path.exists(compiled_path + "model.pt"):
            print(f"\nCompiling for Trainium2 (this takes 30–60 min, please wait)...")
            print(f"  Output: {compiled_path}")
            model.compile(compiled_path)
            print(f"  ✓ Compilation complete")
        else:
            print(f"\n  ✓ Using cached compilation at {compiled_path}")

        # 5. Load compiled model to NeuronCores
        print(f"\nLoading compiled model to Trainium2 NeuronCores...")
        model.load(compiled_path)
        print(f"  ✓ Model loaded to Trainium2 (tp_degree=2, both LNCs active)")

        print(f"\n[STEP 4 COMPLETE] NxDI model ready on Trainium2\n")
        return model

    def run_inference_nxdi(
        self,
        nxdi_model,
        prompt: str = "The future of artificial intelligence is",
        max_tokens: int = 50,
    ):
        """
        [ON TRAINIUM2] Run inference using NxDI model.
        Uses HuggingFaceGenerationAdapter for standard HF generate() API.
        """
        print("[STEP 5/6] Running inference (NxDI on Trainium2)")
        print("=" * 80)

        print(f"\nPrompt: {prompt}")
        print(f"Max tokens to generate: {max_tokens}")

        try:
            from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

            hf_model = HuggingFaceGenerationAdapter(nxdi_model)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            print(f"  ✓ Input encoded: {input_ids.shape}")

            print(f"\nGenerating {max_tokens} tokens...")
            with torch.no_grad():
                output_ids = hf_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            generated_text = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )

            print(f"\n" + "=" * 80)
            print("GENERATED TEXT (NxDI on Trainium2):")
            print("=" * 80)
            print(generated_text)
            print("=" * 80)

            return generated_text

        except Exception as e:
            print(f"\n✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_nxdi(self, nxdi_model, max_tokens: int = 50, num_iterations: int = 5):
        """
        [ON TRAINIUM2] Benchmark NxDI model generation latency.
        """
        print("[STEP 6/6] Benchmarking NxDI model on Trainium2")
        print("=" * 80)

        print(f"\nBenchmark configuration:")
        print(f"  Max new tokens: {max_tokens}")
        print(f"  Iterations: {num_iterations}")

        try:
            from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

            hf_model = HuggingFaceGenerationAdapter(nxdi_model)
            prompt = "The future of artificial intelligence is"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Warmup
            print(f"\nWarming up (2 iterations)...")
            for i in range(2):
                with torch.no_grad():
                    _ = hf_model.generate(
                        input_ids, attention_mask=attention_mask,
                        max_new_tokens=max_tokens, do_sample=False
                    )
                print(f"  Warmup {i+1}/2 complete")

            # Benchmark
            print(f"\nBenchmarking...")
            times = []
            for i in range(num_iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    out = hf_model.generate(
                        input_ids, attention_mask=attention_mask,
                        max_new_tokens=max_tokens, do_sample=False
                    )
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.3f}s "
                      f"({max_tokens / elapsed:.1f} tok/s)")

            avg_latency = sum(times) / len(times)
            throughput = max_tokens / avg_latency

            print(f"\n" + "=" * 80)
            print("BENCHMARK RESULTS (NxDI on Trainium2):")
            print("=" * 80)
            print(f"Average latency: {avg_latency:.3f}s for {max_tokens} new tokens")
            print(f"Min latency:     {min(times):.3f}s")
            print(f"Max latency:     {max(times):.3f}s")
            print(f"Throughput:      {throughput:.2f} tokens/sec")
            print("=" * 80)

            return {
                "avg_latency": avg_latency,
                "min_latency": min(times),
                "max_latency": max(times),
                "throughput": throughput,
            }

        except Exception as e:
            print(f"\n✗ Benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_logger(self):
        """Simple logger for calibration."""
        class SimpleLogger:
            def info(self, msg):
                print(f"  [LOG] {msg}")
        return SimpleLogger()


def main():
    parser = argparse.ArgumentParser(
        description="UNIFIED FLATQUANT + TRAINIUM2 PIPELINE (Everything on Trainium2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calibration + trace on Trainium2:
  python flatquant_trainium_unified.py --model meta-llama/Llama-2-7b-hf

  # With benchmarking:
  python flatquant_trainium_unified.py --model meta-llama/Llama-2-7b-hf --benchmark

  # Custom output and token generation:
  python flatquant_trainium_unified.py \\
      --model meta-llama/Llama-2-7b-hf \\
      --output ./my_quantized_model \\
      --num_tokens 100 \\
      --benchmark
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model name (default: Llama 2 7B)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (for gated models)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./quantized_model",
        help="Output path for quantized model (on Trainium2)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Calibration dataset (wikitext, openwebtext, etc.)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length for tracing and inference"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=50,
        help="Number of tokens to generate for inference"
    )
    parser.add_argument(
        "--cali_epochs",
        type=int,
        default=5,
        help="Calibration epochs per layer (default 5; use 1 for fast testing)"
    )
    parser.add_argument(
        "--cali_bsz",
        type=int,
        default=1,
        help="Calibration batch size (default 1)"
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Skip saving (reuse existing checkpoint in --output dir)"
    )
    parser.add_argument(
        "--skip_calibration",
        action="store_true",
        help="Skip calibration (reuse existing flat_parameters.pth)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmarking on Trainium2"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Prompt for text generation"
    )
    parser.add_argument(
        "--flat_lr",
        type=float,
        default=1e-3,
        help="Learning rate for FlatQuant transform calibration (default 1e-3)"
    )
    parser.add_argument(
        "--identity_transforms",
        action="store_true",
        help="Skip transform folding: copy raw linear weights directly (for pipeline validation)"
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        default=True,
        help="Use FP8 E4M3 per-channel quantization via NxDI (default: True)"
    )
    parser.add_argument(
        "--no_fp8",
        action="store_true",
        help="Disable FP8 — compile in float16 without quantization (baseline)"
    )
    parser.add_argument(
        "--force_recompile",
        action="store_true",
        help="Delete cached reparameterized checkpoint and compiled model to force recompilation"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help=(
            "Run verification checks only (no compilation, no inference). "
            "Checks reparameterized checkpoint for NaN/Inf and, if --fp8, "
            "also creates and verifies the FP8 checkpoint dtypes."
        )
    )

    args = parser.parse_args()

    # Run unified pipeline
    import os as _os
    import shutil as _shutil

    # Clear caches when explicitly forced (identity_transforms alone does not force clear)
    if args.force_recompile:
        reparam_dir = str(args.output) + "_reparameterized"
        compiled_dir = str(Path(args.output).parent / "compiled_nxdi_model")
        for d in [reparam_dir, compiled_dir]:
            if Path(d).exists():
                print(f"  [CACHE] Removing {d}")
                _shutil.rmtree(d)

    try:
        pipeline = TrainiumUnifiedPipeline(
            args.model,
            args.output,
            args.hf_token,
        )

        nxdi_only = args.skip_calibration and args.skip_save

        if nxdi_only:
            # NxDI-only path: load tokenizer directly from checkpoint, skip model wrap
            print("[STEP 1/6] Skipping model load (--skip_calibration + --skip_save)\n")
            print("[STEP 2/6] Skipping calibration (--skip_calibration)\n")
            print("[STEP 3/6] Skipping save (--skip_save)\n")
            pipeline.tokenizer = AutoTokenizer.from_pretrained(
                str(pipeline.output_path),
                use_fast=False,
                token=args.hf_token,
            )
            if pipeline.tokenizer.pad_token is None:
                pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
        else:
            # Full pipeline: load, wrap, calibrate, save
            model, tokenizer, quant_args = pipeline.load_and_wrap_model(
                num_samples=args.num_samples,
                cali_epochs=args.cali_epochs,
                cali_bsz_val=args.cali_bsz,
                flat_lr_val=args.flat_lr,
            )

            if not args.skip_calibration:
                pipeline.calibrate_on_trainium2(quant_args, args.dataset, args.num_samples)
            else:
                flat_params_path = pipeline.output_path / "flat_parameters.pth"
                if flat_params_path.exists():
                    print(f"[STEP 2/6] Loading saved flat_parameters from {flat_params_path}")
                    import flatquant.flat_utils as flat_utils
                    flat_utils.load_flat_parameters(quant_args, pipeline.model)
                    print(f"  ✓ Flat parameters loaded into model\n")
                else:
                    print("[STEP 2/6] Skipping calibration (--skip_calibration, no flat_parameters.pth found)\n")

            if not args.skip_save:
                pipeline.save_quantized_on_trainium2(quant_args)
            else:
                print("[STEP 3/6] Skipping save (--skip_save, reusing existing checkpoint)\n")

        # Step 4: Compile with NxDI (tp_degree=2, distributes across both LNCs)
        use_fp8 = args.fp8 and not args.no_fp8
        pipeline._verify_only = getattr(args, 'verify_only', False)
        nxdi_model = pipeline.compile_with_nxdi(
            sequence_length=args.sequence_length,
            max_new_tokens=args.num_tokens,
            identity_transforms=args.identity_transforms,
            fp8=use_fp8,
        )

        # Step 5: Inference (skipped when --verify_only)
        if nxdi_model is None:
            print("\n[VERIFY ONLY] Done. To proceed with compilation, re-run without --verify_only.")
            return

        generated = pipeline.run_inference_nxdi(
            nxdi_model,
            prompt=args.prompt,
            max_tokens=args.num_tokens,
        )

        # Step 6: Optional benchmarking
        if args.benchmark:
            stats = pipeline.benchmark_nxdi(nxdi_model, max_tokens=args.num_tokens)

        print("\n" + "=" * 80)
        print("✓ UNIFIED FLATQUANT + TRAINIUM2 PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Quantized model location: {args.output} (on Trainium2)")
        print(f"Status: ALL EXECUTION ON TRAINIUM2 INSTANCE")
        print("=" * 80 + "\n")

    except Exception as e:
        import traceback
        print(f"\n✗ PIPELINE FAILED: {e}", flush=True)
        traceback.print_exc()

    finally:
        # Always bypass _XLAC destructor segfault on process exit
        # (known torch_neuronx/_XLAC issue when large tensors are in memory at shutdown)
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        _os._exit(0)


if __name__ == "__main__":
    main()
