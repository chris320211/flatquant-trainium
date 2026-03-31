#!/usr/bin/env python3
"""
Trainium2 inference wrapper for quantized Llama-2-7b-hf.
Wraps the quantized model for deployment on Trainium2 hardware.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any


class TrainiumQuantizedLlamaModel(nn.Module):
    """
    Wrapper for quantized Llama-2-7b-hf model on Trainium2.

    This class:
    1. Loads a quantized model checkpoint
    2. Loads Neuron-converted weights
    3. Provides inference interface for Trainium2
    """

    def __init__(
        self,
        quantized_model_path: str,
        weights_path: Optional[str] = None,
        tp_degree: int = 1,
        device: str = "cpu",
    ):
        """
        Args:
            quantized_model_path: Path to quantized model checkpoint
            weights_path: Path to Neuron-converted weights (optional)
            tp_degree: Tensor parallelism degree (1 = single device)
            device: Device to load model on ("cpu" or "neuron")
        """
        super().__init__()

        self.quantized_model_path = Path(quantized_model_path)
        self.weights_path = Path(weights_path) if weights_path else None
        self.tp_degree = tp_degree
        self.device_name = device

        self.model = None
        self.tokenizer = None
        self.config = None

    def load_quantized_model(self):
        """Load quantized model from checkpoint"""
        print(f"Loading quantized model from {self.quantized_model_path}")

        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

        # Load config
        self.config = AutoConfig.from_pretrained(str(self.quantized_model_path))
        print(f"✓ Config loaded: {self.config.model_type}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.quantized_model_path),
            torch_dtype=torch.float16,
            device_map="auto" if self.device_name == "cpu" else self.device_name,
        )
        self.model.eval()
        print(f"✓ Model loaded: {type(self.model).__name__}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.quantized_model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✓ Tokenizer loaded")

        return self.model

    def load_neuron_weights(self):
        """Load Neuron-converted weights (if available)"""
        if not self.weights_path:
            print("No Neuron weights path provided, using quantized model weights")
            return

        if not self.weights_path.exists():
            print(f"Warning: Weights path does not exist: {self.weights_path}")
            return

        print(f"Loading Neuron weights from {self.weights_path}")

        try:
            # Load Neuron state dict
            weights_file = self.weights_path / "model.safetensors"
            if not weights_file.exists():
                weights_file = self.weights_path / "pytorch_model.bin"

            if weights_file.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_file))
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✓ Neuron weights loaded")
            else:
                print(f"Warning: Weights file not found in {self.weights_path}")
        except Exception as e:
            print(f"Warning: Could not load Neuron weights: {e}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments passed to model

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

        return outputs.logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        return outputs

    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Token IDs tensor [1, seq_len]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        return inputs.input_ids.to(self.model.device)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids[0] if token_ids.ndim > 1 else token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def benchmark(self, prompt: str, num_tokens: int = 128) -> Dict[str, Any]:
        """
        Benchmark model inference performance.

        Args:
            prompt: Input prompt
            num_tokens: Number of tokens to generate

        Returns:
            Dictionary with timing information
        """
        import time

        print(f"\nBenchmarking with prompt: '{prompt[:50]}...'")
        print(f"Generating {num_tokens} tokens")

        # Encode prompt
        input_ids = self.encode(prompt)

        # Warmup
        print("Warmup (1 iteration)...")
        with torch.no_grad():
            _ = self.generate(input_ids, max_new_tokens=10)

        # Measure
        print(f"Measuring (5 iterations)...")
        times = []
        for i in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                outputs = self.generate(input_ids, max_new_tokens=num_tokens)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f}s")

        avg_time = sum(times) / len(times)
        tokens_per_sec = num_tokens / avg_time

        results = {
            "prompt": prompt,
            "num_tokens_generated": num_tokens,
            "average_latency_seconds": avg_time,
            "tokens_per_second": tokens_per_sec,
            "individual_times": times,
        }

        print(f"\nResults:")
        print(f"  Avg latency: {avg_time:.2f}s")
        print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trainium2 Quantized Llama Inference")
    parser.add_argument("--quantized_model", type=str, required=True,
                       help="Path to quantized model checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                       help="Path to Neuron-converted weights")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run on (cpu or neuron)")
    parser.add_argument("--prompt", type=str,
                       default="The future of AI is",
                       help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark")

    args = parser.parse_args()

    print("=" * 60)
    print("Trainium2 Quantized Llama-2-7b-hf Inference")
    print("=" * 60)

    # Create wrapper
    wrapper = TrainiumQuantizedLlamaModel(
        quantized_model_path=args.quantized_model,
        weights_path=args.weights,
        device=args.device,
    )

    # Load models
    wrapper.load_quantized_model()
    wrapper.load_neuron_weights()

    # Run inference or benchmark
    if args.benchmark:
        wrapper.benchmark(args.prompt, num_tokens=args.max_tokens)
    else:
        print(f"\nPrompt: {args.prompt}")
        input_ids = wrapper.encode(args.prompt)
        outputs = wrapper.generate(input_ids, max_new_tokens=args.max_tokens)
        generated_text = wrapper.decode(outputs)
        print(f"Generated: {generated_text}")

    print("=" * 60)
