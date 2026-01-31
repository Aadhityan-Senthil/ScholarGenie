"""Model management and loading utilities."""

import os
import logging
from typing import Optional, Dict, Any
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of ML models."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model manager.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}

        # Determine device
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")

    def _get_device(self) -> str:
        """Determine the compute device to use."""
        # Check environment variable first
        env_device = os.getenv("DEVICE")
        if env_device:
            return env_device

        # Check config
        config_device = self.config.get("summarization", {}).get("device", "cpu")

        # Validate CUDA availability
        if config_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        return config_device

    def get_summarization_model(self):
        """Load and return summarization model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = self.config["summarization"]["model_name"]

        if model_name not in self._models:
            logger.info(f"Loading summarization model: {model_name}")

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                model = model.to(self.device)
                model.eval()

                self._models[model_name] = model
                self._tokenizers[model_name] = tokenizer

                logger.info(f"Successfully loaded {model_name}")

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Fallback to a smaller model
                fallback_model = "google/long-t5-local-base"
                logger.info(f"Attempting fallback to {fallback_model}")

                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
                model = model.to(self.device)
                model.eval()

                self._models[model_name] = model
                self._tokenizers[model_name] = tokenizer

        return self._models[model_name], self._tokenizers[model_name]

    def get_summarization_pipeline(self):
        """Get a summarization pipeline.

        Returns:
            Hugging Face summarization pipeline
        """
        model_name = self.config["summarization"]["model_name"]
        cache_key = f"pipeline_{model_name}"

        if cache_key not in self._models:
            logger.info(f"Creating summarization pipeline for {model_name}")

            model, tokenizer = self.get_summarization_model()

            sum_pipeline = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            self._models[cache_key] = sum_pipeline

        return self._models[cache_key]

    def get_bertscore_model(self):
        """Get model for BERTScore evaluation.

        Returns:
            Model name for BERTScore
        """
        return self.config.get("evaluation", {}).get(
            "bertscore_model",
            "microsoft/deberta-base-mnli"
        )

    def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> str:
        """Summarize text using the loaded model.

        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search

        Returns:
            Summary text
        """
        model, tokenizer = self.get_summarization_model()

        # Get config values
        config = self.config["summarization"]
        max_length = max_length or config.get("max_output_length", 512)
        min_length = min_length or config.get("min_output_length", 100)
        num_beams = num_beams or config.get("num_beams", 4)
        max_input = config.get("max_input_length", 4096)

        # Tokenize input
        inputs = tokenizer(
            text,
            max_length=max_input,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        # Decode
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> list:
        """Split text into chunks for processing.

        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        model_name = self.config["summarization"]["model_name"]
        tokenizer = self._tokenizers.get(model_name)

        if not tokenizer:
            # Load tokenizer if not already loaded
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._tokenizers[model_name] = tokenizer

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            start = end - overlap

        return chunks

    def clear_cache(self):
        """Clear model cache to free memory."""
        self._models.clear()
        self._tokenizers.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models.

        Returns:
            Dictionary with model information
        """
        return {
            "device": self.device,
            "summarization_model": self.config["summarization"]["model_name"],
            "loaded_models": list(self._models.keys()),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
