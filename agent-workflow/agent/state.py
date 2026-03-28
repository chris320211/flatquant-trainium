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

    # Shared conversation/debug log
    messages: list
