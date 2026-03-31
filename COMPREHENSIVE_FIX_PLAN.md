# Comprehensive Fix Plan: FlatQuant + Trainium2 Agent

## Current State Analysis

### ✅ What's Working
1. **Agent successfully generated:**
   - FlatQuant quantization configuration (quant_config_llama_2_7b_hf.py)
   - FlatQuant wrapper classes (llama_2_7b_hf_utils.py) - partially
   - Calibration script (calibrate_llama_2_7b_hf.py)
   - Patching logic (patch_llama_2_7b_hf.py)
   - Runner script structure (run_llama_2_7b_hf.py)

2. **Dependencies installed:**
   - neuronx-distributed-inference ✅
   - torch-2.9.1 ✅
   - transformers ✅

3. **Base infrastructure:**
   - Model config loads correctly ✅
   - HuggingFace authentication works ✅

### ❌ Critical Issues

#### Issue 1: Broken FlatQuant Dependencies
**Status:** Generated code imports from `flatquant` module which doesn't exist in path
```python
from flatquant.quant_utils import ActivationQuantizer  # ❌ NOT FOUND
from flatquant.utils import DEV, skip_initialization   # ❌ NOT FOUND
from flatquant.flat_linear import FlatQuantizedLinear  # ❌ NOT FOUND
```

**Impact:** Cannot run calibration or apply FlatQuant without this module
**Root Cause:** Agent assumed FlatQuant module is available; it's in FlatQuantBundled but needs proper setup

#### Issue 2: Incomplete NxDI Model Implementation
**Status:** Generated classes are stubs requiring distributed initialization
```python
# nxdi/neuron_llama_2_7b_hf_nxdi.py has:
class NeuronLlama27bHfModel(NeuronBaseModel):
    # Only defines init_model() but __init__ fails
    # Requires torch.distributed setup
    # Requires SPMDRank initialization
```

**Impact:** Cannot instantiate or run inference on Trainium
**Root Cause:** Agent generated incomplete Neuron SDK integration

#### Issue 3: Missing deploy.nn Module
**Status:** modeling_llama_2_7b_hf.py imports non-existent deploy module
```python
from deploy.nn import Linear4bit, OnlineTrans, Quantizer  # ❌ NOT FOUND
from deploy.kernels.pytorch.kron_matmul_pytorch import ...  # ❌ NOT FOUND
```

**Impact:** Modified model file cannot be imported
**Root Cause:** Agent assumes custom deploy module exists

#### Issue 4: Disconnected Test Suite
**Status:** Generated tests don't match the actual generated code structure
- Tests reference classes that don't exist or have wrong signatures
- Block tests expect classes with incompatible initialization

---

## End Goal Recap

**User Input:** Model name (e.g., "meta-llama/Llama-2-7b-hf")

**Pipeline:**
1. Agent fetches and analyzes model architecture
2. Agent applies **FlatQuant INT4 quantization** to the model
3. Agent adapts quantized model to **Trainium2** hardware using NxDI
4. User gets a **quantized model optimized for Trainium2** that can run inference

**Output:** Quantized model runnable on Trainium2 with FlatQuant optimization

---

## Comprehensive Fix Plan

### Phase 1: Fix FlatQuant Integration (Critical - Must Work First)

#### 1.1 Setup FlatQuant Module Path
**Action:** Make FlatQuantBundled accessible to generated code

```bash
# Option A: Add to Python path in runner script
export PYTHONPATH=/home/ubuntu/flatquant-trainium/FlatQuantBundled:$PYTHONPATH

# Option B: Create symlink in outputs directory
ln -s /home/ubuntu/flatquant-trainium/FlatQuantBundled/flatquant ./flatquant
```

**Files to Update:**
- `run_llama_2_7b_hf.py` - Add path setup at top
- `calibrate_llama_2_7b_hf.py` - Add path setup at top
- `patch_llama_2_7b_hf.py` - Verify imports work

#### 1.2 Replace deploy.nn Module with FlatQuant Equivalents
**Problem:** modeling_llama_2_7b_hf.py imports from non-existent `deploy` module

**Solution:** Rewrite to use actual FlatQuant classes

**Current (broken):**
```python
from deploy.nn import Linear4bit, OnlineTrans, Quantizer
```

**Fix:** Use FlatQuantBundled classes:
```python
from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.quant_utils import ActivationQuantizer
# No custom kernels needed - use PyTorch
```

**Action Items:**
- Modify agent's modeling_llama_2_7b_hf.py generation to output correct imports
- Simplify FlatQuantLlamaMLP/Attention to use FlatQuantizedLinear directly
- Remove references to custom kernels (kron_matmul, block_matmul)

#### 1.3 Unify Quantization Code
**Problem:** Two different quantization approaches in the codebase:
1. `llama_2_7b_hf_utils.py` - Uses FlatQuantizedLinear
2. `modeling_llama_2_7b_hf.py` - Uses Linear4bit (doesn't exist)

**Solution:** Use ONE approach consistently

**Recommendation:** Use llama_2_7b_hf_utils.py approach:
- FlatQuantLlamaMLP (FlatQuant wrapper)
- FlatQuantLlamaAttention (FlatQuant wrapper)
- Apply via patch_llama_2_7b_hf.py

**Action Items:**
- Remove or replace modeling_llama_2_7b_hf.py
- Keep patch_llama_2_7b_hf.py approach
- Verify calibrate_llama_2_7b_hf.py uses correct flow

#### 1.4 Fix Calibration Script
**Problem:** calibrate_llama_2_7b_hf.py has incomplete implementation
- References `apply_flatquant_to_llama_2_7b_hf` function that may not exist in utils
- Missing proper configuration merging
- Unclear data loading

**Solution:** Rewrite calibration to be self-contained

**Action Items:**
- Simplify: Load model → Apply FlatQuant → Run calibration → Save
- Use established FlatQuantBundled calibration patterns
- Test with small dataset first (wikitext-2 sample)

---

### Phase 2: Trainium2 Integration (After FlatQuant Works)

#### ⚠️ CRITICAL: XLA Tracing vs FlatQuant INT4 Incompatibility

**The Hard Problem:**
FlatQuant uses INT4 weights with Kronecker-transformed activations. Trainium compiles via XLA (torch_neuronx), which means:

1. **Custom INT4 packing ops won't trace cleanly through XLA**
   - FlatQuantBundled's INT4 operations assume PyTorch CPU/GPU execution
   - XLA compilation doesn't understand custom FlatQuant kernels
   - torch.neuron.trace() will fail on unpacking/dequantization ops

2. **Activation quantization (FP8) requires special handling**
   - FlatQuant uses online activation quantization (FP8)
   - These custom quantizers won't trace through XLA
   - Need either dequantization or NKI custom kernels

3. **Weight format mismatch**
   - FlatQuant stores weights in quantized format with transform matrices
   - Trainium expects weights in standard format (optionally sharded by TP degree)
   - Simple key remapping is insufficient

#### 2.1 Phase 2 Option A: Dequantization Pass (Simpler, Slower)

**Approach:** INT4 → BF16 before Trainium tracing

```python
# nxdi/dequant_for_trainium.py
def dequantize_flatquant_model(model):
    """
    Convert FlatQuant INT4 model to BF16 for Trainium tracing.

    This:
    1. Loads quantized model with FlatQuant wrappers
    2. Performs forward pass with dequantization
    3. Extracts full-precision weights
    4. Saves as standard BF16 model (no quantization)

    Trade-off: Loses quantization benefits but guarantees Trainium compatibility
    """
    # Load quantized model
    model = load_quantized_model(model_path)

    # Extract dequantized weights
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, FlatQuantizedLinear):
            # Get dequantized weight
            weight = module.get_dequantized_weight()  # INT4 → BF16
            state_dict[name + '.weight'] = weight.to(torch.bfloat16)
            if module.bias is not None:
                state_dict[name + '.bias'] = module.bias.to(torch.bfloat16)

    # Create new model with standard Linear layers (no quantization)
    model_deq = create_standard_model(config)
    model_deq.load_state_dict(state_dict)

    return model_deq

# Save dequantized model for Trainium
dequant_model = dequantize_flatquant_model(quantized_model)
dequant_model.save_pretrained("./llama2_bf16_for_trainium/")
```

**Pros:**
- Simple to implement
- Guaranteed Trainium compatibility
- Standard PyTorch operations

**Cons:**
- Loses INT4 quantization benefits
- Model is now BF16 (larger, slower than INT4)
- No memory savings

**When to use:** If you need guaranteed Trainium support immediately

#### 2.2 Phase 2 Option B: NKI Custom Operators (Complex, Fast)

**Approach:** Use Neuron Kernel Interface (NKI) for INT4 matmul

```python
# nxdi/int4_nki_kernels.py
# This requires NKI expertise (reference Kevin's NxDI work)

class NKIInt4Linear(nn.Module):
    """
    INT4 linear layer using NKI custom operators for Trainium.

    This replaces FlatQuantizedLinear with Trainium-compatible equivalent.
    """
    def __init__(self, in_features, out_features, quantization_config):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store INT4 packed weights
        self.weight_int4 = nn.Parameter(torch.randint(0, 2, (out_features, in_features // 2)))

        # Store scale factors
        self.scale = nn.Parameter(torch.ones(out_features, in_features // 128))

        # Store transformation matrices for activation quantization
        self.act_scale = nn.Parameter(torch.ones(in_features // 128))

    def forward(self, x):
        # This will use nki.jit compiled INT4 matmul kernel
        # Requires NKI implementation
        raise NotImplementedError("Requires NKI kernel implementation")
```

**Pros:**
- Keeps INT4 quantization benefits
- Optimized for Trainium hardware
- Better performance than dequantized

**Cons:**
- Requires NKI expertise (complex)
- Limited documentation
- Non-portable (Trainium-specific)

**When to use:** If you want maximum performance and have NKI expertise

#### 2.3 Phase 2 Option C: Hybrid (Recommended for Now)

**Approach:**
1. Use Option A (dequantization) for initial Trainium deployment
2. Plan Option B (NKI kernels) as future optimization
3. In the meantime, get a working system

```python
# nxdi/phase2_hybrid.py

# Step 1: Dequantize quantized model from Phase 1
dequant_model = dequantize_flatquant_model(
    quantized_model_path="./quantized_llama2_7b"
)
dequant_model.save_pretrained("./llama2_bf16_for_trainium/")

# Step 2: Standard weight conversion (no special sharding for now)
weights = dequant_model.state_dict()
save_weights_for_trainium(weights, "./weights_for_trainium/")

# Step 3: Trace with torch_neuronx
model_neuron = torch.neuron.trace(
    dequant_model,
    example_input,
    compiler_workdir="./compiler_workdir/",
)

# Step 4: Run inference on Trainium
output = model_neuron(input_ids)
```

---

#### 2.4 Weight Conversion for Trainium (Updated)

**Real requirements:**

1. **Format conversion** - Weights must be in Trainium-compatible format:
```python
# For single-device (tp_degree=1): just convert to BF16
weights_bf16 = dequant_model.state_dict()

# For multi-device (tp_degree>1): shard by dimension
# Example: linear weight [out, in] → shard along out dimension
def shard_weights(state_dict, tp_degree):
    sharded = {}
    for key, tensor in state_dict.items():
        if 'weight' in key and len(tensor.shape) == 2:
            # Shard along first dimension (output features)
            sharded[key] = tensor.split(tensor.shape[0] // tp_degree)
        else:
            sharded[key] = tensor
    return sharded
```

2. **Validation** - Ensure weights match model architecture:
```python
def validate_weights_for_trainium(model, weights):
    """Validate weight shapes match model expectations"""
    for name, param in model.named_parameters():
        if name not in weights:
            raise ValueError(f"Missing weight: {name}")
        if weights[name].shape != param.shape:
            raise ValueError(f"Shape mismatch for {name}: "
                           f"expected {param.shape}, got {weights[name].shape}")
    return True
```

3. **Serialization** - Save in format Trainium can load:
```python
# Option 1: PyTorch format (simple)
torch.save(weights, "weights.pt")

# Option 2: Safetensors (recommended)
from safetensors.torch import save_file
save_file(weights, "weights.safetensors")

# Option 3: HuggingFace format
model.save_pretrained("./weights_dir/")
```

---

#### 2.5 Actual Phase 2 Implementation (Given Current Constraints)

**For now, do this:**

```python
# nxdi/phase2_trainium_deployment.py
class TrainiumDeploymentPipeline:
    """
    Realistic Phase 2: Dequant → Save → Trace → Run
    """

    def __init__(self, quantized_model_path):
        self.quantized_model_path = quantized_model_path

    def step1_dequantize(self):
        """Convert INT4 quantized model to BF16"""
        print("Step 1: Dequantizing FlatQuant model...")
        model = load_flatquant_model(self.quantized_model_path)

        # Extract dequantized weights
        state_dict = extract_dequantized_weights(model)

        # Create standard BF16 model
        model_bf16 = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16
        )
        model_bf16.load_state_dict(state_dict)

        return model_bf16

    def step2_trace_for_trainium(self, model_bf16):
        """Trace model with torch_neuronx for Trainium compilation"""
        print("Step 2: Tracing for Trainium...")

        # Create example input
        example_input = torch.randint(0, 32000, (1, 128))

        # Trace
        model_neuron = torch.neuron.trace(
            model_bf16,
            example_input,
            compiler_workdir="./compiler_workdir/",
            compiler_args=[
                "--model-type=transformer",
                "--num-neuroncores=1",  # or 2, 8, etc.
            ]
        )

        return model_neuron

    def step3_benchmark(self, model_neuron):
        """Benchmark inference latency"""
        print("Step 3: Benchmarking...")

        # Warmup
        for _ in range(3):
            output = model_neuron(torch.randint(0, 32000, (1, 128)))

        # Measure
        import time
        times = []
        for _ in range(10):
            start = time.time()
            output = model_neuron(torch.randint(0, 32000, (1, 128)))
            times.append(time.time() - start)

        avg_latency = sum(times) / len(times)
        print(f"Average latency: {avg_latency:.2f}s")

        return avg_latency

    def deploy(self):
        """Full deployment pipeline"""
        model_bf16 = self.step1_dequantize()
        model_neuron = self.step2_trace_for_trainium(model_bf16)
        latency = self.step3_benchmark(model_neuron)
        return model_neuron, latency
```

**Expected behavior:**
- Step 1: 2-5 minutes (dequantization)
- Step 2: 5-15 minutes (tracing and compilation)
- Step 3: 1-2 minutes (benchmarking)
- **Total: ~15-25 minutes**

**Expected performance:**
- Dequantized BF16 on Trainium: ~3-5x speedup vs CPU
- *Note: Not 5-10x speedup because we lost INT4 quantization*
- *But it's working and stable*

---

### Phase 3: Agent Improvements (For Future Runs)

#### 3.1 Fix Agent Prompts
**Issue:** Agent generates incomplete/incompatible code

**Changes to agent/prompts.py:**
- Remove assumptions about `deploy` module
- Generate only FlatQuantBundled-compatible code
- Make NxDI generation conditional (skip if incomplete)
- Add validation checks for imports

#### 3.2 Fix Agent Node Logic
**Issue:** Trainium integration nodes don't validate dependencies

**Changes to agent/nodes/:**
- Add import validation before code generation
- Verify FlatQuantBundled is available
- Skip NxDI generation if dependencies missing
- Add human-readable error messages

#### 3.3 Restructure Output
**Issue:** Mixed concerns in output directory

**Better structure:**
```
outputs/meta-llama__Llama-2-7b-hf/
├── flatquant/              # Phase 1: Quantization
│   ├── calibrate.py
│   ├── patch.py
│   ├── run.py
│   └── config.py
├── trainium/               # Phase 2: Trainium deployment
│   ├── convert_weights.py
│   ├── inference.py
│   └── run.py
├── tests/
└── README.md
```

---

## Implementation Priority

### Must Have (Critical Path)
1. ✅ Fix FlatQuant module path setup
2. ✅ Fix imports in modeling_llama_2_7b_hf.py
3. ✅ Unify quantization approach
4. ✅ Test calibration script works
5. ✅ Create executable weight conversion
6. ✅ Create executable Trainium inference

### Should Have (Quality)
1. Test suite that actually works
2. Better error messages
3. Performance benchmarking

### Nice to Have (Future)
1. Multi-model support
2. Distributed training setup
3. Model compilation optimization

---

## Testing Strategy

### Test 1: FlatQuant Quantization Works
```bash
# After Phase 1 fixes
cd outputs/meta-llama__Llama-2-7b-hf
python calibrate_llama_2_7b_hf.py \
    --model meta-llama/Llama-2-7b-hf \
    --w_bits 4 \
    --a_bits 8 \
    --nsamples 128
# Expected: Produces quantized model checkpoint
```

### Test 2: Trainium Model Can Load
```bash
# After Phase 2 fixes
python nxdi/convert_weights_to_neuron.py \
    --quantized-model ./quantized_checkpoint \
    --output ./weights_neuron
# Expected: Converts weights without errors
```

### Test 3: Inference Runs on Trainium
```bash
# After Phase 2 fixes
python nxdi/run_on_trainium.py \
    --model ./quantized_checkpoint \
    --weights ./weights_neuron \
    --benchmark
# Expected: Runs forward pass with latency measurement
```

---

## Success Criteria

### Phase 1 Success
- [ ] calibrate_llama_2_7b_hf.py runs without import errors
- [ ] FlatQuant wrapper classes instantiate correctly
- [ ] Calibration produces quantized model
- [ ] Quantized model inference works on CPU

### Phase 2 Success
- [ ] Weight conversion script runs without errors
- [ ] Model loads on Trainium2 without distributed errors
- [ ] Forward pass executes on Trainium2 hardware
- [ ] Performance is better than unquantized CPU baseline

### Overall Success
- [ ] User can: `python agent/main.py "meta-llama/Llama-2-7b-hf"`
- [ ] Get back: Quantized model optimized for Trainium2
- [ ] Can immediately: Run inference on Trainium2 hardware

---

## Root Cause Analysis

### Why Did Agent Generation Fail?

1. **Agent assumed external modules exist:**
   - Assumed `flatquant` module is in path (it's in FlatQuantBundled)
   - Assumed `deploy` module exists (doesn't exist anywhere)
   - No validation of dependencies before code generation

2. **NxDI integration was incomplete:**
   - Agent generated class stubs without full implementation
   - Distributed initialization requirements not addressed
   - No error handling for missing dependencies

3. **Testing was insufficient:**
   - Generated tests don't match generated code
   - No integration tests for full pipeline
   - No validation that generated code is executable

### Lesson for Next Iteration

Agent should:
1. Validate all external dependencies BEFORE generating code
2. Generate testable, self-contained code
3. Include minimal example that actually runs
4. Skip complex integrations if dependencies missing
5. Generate actionable error messages for users

