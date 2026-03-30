from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    # Input
    model_name: str

    # arch_agent populates
    model_config: dict          # HF config as dict (hidden_size, num_layers, etc.)
    linears: dict               # {layer_name: {in_features, out_features}}
    modeling_source: str        # full text of the model's modeling .py file
    modeling_source_path: str   # path to the modeling .py on disk
    model_type: str             # e.g. "gemma2", "mistral", "llama"
    has_moe: bool               # whether MoE routing blocks were detected

    # ref_reader populates
    ref_patterns: dict          # extracted FlatQuant patterns ready for codegen

    # codegen_agent populates
    generated_files: dict       # {filename: source_code_string}

    # reg_agent populates (appended into generated_files)
    registration_code: str      # apply_flatquant_to_{model} source

    # validation_agent populates
    validation_result: dict     # {passed: bool, errors: list[str]}

    # After validation (optional subprocess)
    flatquant_calibrate_result: dict  # smoke/full calibrate or skip

    # nxdi_port node (after validation if passed; TRAINIUM_SKILL_MODE=fast)
    nxdi_result: dict           # {skipped, reason?, written_files?, filenames?}

    # Trainium full skill chain (TRAINIUM_SKILL_MODE=full, default)
    trainium_plan: dict         # Phase 1 JSON (architecture inventory, partitions, etc.)
    trainium_skill_setup_result: dict  # block_testing_utils copy + package inits
    trainium_block_files: dict  # {relative_path: content} from Phase 2 LLM
    trainium_test_audit: dict   # anti-cheat static check (skill Phase 2 audit)
    trainium_test_report: dict  # pytest rc / stdout or skip reason
    trainium_integrate_result: dict  # {written_files, filenames, skipped?}
    trainium_weight_result: dict     # {written_files, filenames, skipped?}
    trainium_nxdi_verify: dict       # import neuronx_distributed + exec neuron_*_nxdi.py
    trainium_compile_smoke_result: dict  # optional TRAINIUM_COMPILE_CMD / SMOKE_CMD

    # Shared conversation/debug log
    messages: list
