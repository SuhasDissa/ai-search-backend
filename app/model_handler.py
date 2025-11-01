import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from .config import settings

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = settings.DEVICE
        self.model_name = settings.MODEL_NAME

    def load_model(self):
        """Load the Qwen model and tokenizer"""
        print(f"Loading model {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model with appropriate settings based on device
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # CPU mode - don't use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        print("Model loaded successfully!")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = None,
        top_p: float = None
    ) -> str:
        """Generate a response using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        temperature = temperature or settings.TEMPERATURE
        top_p = top_p or settings.TOP_P

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

        if self.device == "cuda":
            inputs = inputs.to("cuda")

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

# Global model handler instance
model_handler = ModelHandler()
