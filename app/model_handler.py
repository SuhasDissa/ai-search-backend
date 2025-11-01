import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from .config import settings
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = settings.DEVICE
        self.model_name = settings.MODEL_NAME

    def load_model(self):
        """Load the Qwen model and tokenizer"""
        logger.info(f"Loading model {self.model_name} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("Tokenizer loaded successfully")

            # Load model with appropriate settings based on device
            if self.device == "cuda":
                logger.info("Loading model with CUDA support (float16, device_map=auto)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU mode - don't use device_map
                logger.info("Loading model in CPU mode (float32)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded successfully! Total parameters: {num_params:,}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = None,
        top_p: float = None
    ) -> str:
        """Generate a response using the loaded model"""
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded when generate_response was called")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        temperature = temperature or settings.TEMPERATURE
        top_p = top_p or settings.TOP_P

        logger.debug(f"Generating response with max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Format prompt for Qwen model
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8000  # Qwen supports ~8k tokens
        )

        input_token_count = inputs['input_ids'].shape[1]
        logger.debug(f"Input tokenized: {input_token_count} tokens")

        if self.device == "cuda":
            inputs = inputs.to("cuda")

        # Generate response
        import time
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generation_time = time.time() - start_time
        output_token_count = outputs.shape[1] - input_token_count
        tokens_per_sec = output_token_count / generation_time if generation_time > 0 else 0

        logger.info(f"Generated {output_token_count} tokens in {generation_time:.2f}s ({tokens_per_sec:.1f} tokens/s)")

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        logger.debug(f"Response length: {len(response)} characters")
        return response.strip()

# Global model handler instance
model_handler = ModelHandler()
