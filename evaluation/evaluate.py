# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import pipeline

from kvpress import ComposedPress, DuoAttentionPress, FinchPress, ObservedAttentionPress, ThinKPress

logger = logging.getLogger(__name__)


def load_config(config_path: Path, cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file and overrides with command-line arguments.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    cli_args : Dict[str, Any]
        Dictionary of command-line arguments passed.

    Returns
    -------
    Dict[str, Any]
        The merged configuration dictionary.
    """
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using only command-line arguments and defaults.")
        base_config = {}
    else:
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f) or {}

    # Override config values with non-None command-line arguments
    for key, value in cli_args.items():
        if value is not None:
            base_config[key] = value

    return base_config


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration dictionary for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    predictions_prefix = "predictions"
    metrics_prefix = "metrics"

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary for the evaluation run.
        """
        self.config = config
        self.pipeline = None
        self.press = None
        self.df = None
        self._setup_logging()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(self.config, indent=2)}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.get("log_level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _load_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config["dataset"]
        data_dir = str(self.config["data_dir"]) if self.config["data_dir"] else None
        fraction = self.config["fraction"]

        assert dataset_name in DATASET_REGISTRY, f"No dataset found for {dataset_name}"
        assert dataset_name in SCORER_REGISTRY, f"No scorer found for {dataset_name}"

        logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
        df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()

        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=42)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        self.df = df
        logger.info(f"Dataset loaded with {len(self.df)} entries.")

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config["press_name"]
        compression_ratio = self.config["compression_ratio"]
        key_channel_compression_ratio = self.config["key_channel_compression_ratio"]

        assert press_name in PRESS_REGISTRY, f"Press '{press_name}' not found in PRESS_REGISTRY"
        press = PRESS_REGISTRY[press_name]

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            press.head_compression_ratio = compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {compression_ratio}")
        elif issubclass(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(
                        f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}"
                    )
                else:
                    # type:ignore[attr-defined] is handled by checking if attr exists or using try-except
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(
                            f"Set ComposedPress compression_ratio to {compression_ratio}"
                        )
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(f"Press {press.__class__.__name__} has no 'compression_ratio' attribute.")

        self.press = press
        logger.info(f"KV Press '{press_name}' setup.")

    def _setup_model_pipeline(self):

        model_name = self.config["model"]
        device = self.config["device"]

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config["model_kwargs"] or {}
        if isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        else:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        if device == "auto":
            self.pipeline = pipeline(
                "kv-press-text-generation", model=model_name, device_map="auto", model_kwargs=model_kwargs
            )
        else:
            self.pipeline = pipeline(
                "kv-press-text-generation", model=model_name, device=device, model_kwargs=model_kwargs
            )
        logger.info("Model pipeline loaded.")

    def _prepare_data_for_inference(self):
        """
        Prepares the dataset for inference, handling `compress_questions` and `FinchPress` specifics.
        """
        compress_questions = self.config["compress_questions"]

        if isinstance(self.press, FinchPress):
            if not compress_questions:
                logger.error("FinchPress requires 'compress_questions' to be set to True.")
                raise ValueError("FinchPress requires compress_questions to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)
            self.df["context"] = self.df["context"] + self.press.delimiter_token

        if compress_questions:
            logger.info("Compressing questions into context.")
            self.df["context"] = self.df["context"] + self.df["question"]
            self.df["question"] = ""

    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """
        max_new_tokens_cfg = self.config["max_new_tokens"]
        max_context_length_cfg = self.config["max_context_length"]

        self.df["predicted_answer"] = None
        # Group by context to avoid reprocessing the same context multiple times if questions differ
        df_context_grouped = self.df.groupby("context")
        assert all(
            df_context_grouped["answer_prefix"].nunique() == 1
        ), "Inconsistent 'answer_prefix' within the same context group detected."

        logger.info("Starting inference...")
        for context, df_group in tqdm(df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"):
            questions = df_group["question"].to_list()
            # Use max_new_tokens from config, or fallback to dataset's default for the task
            max_new_tokens = (
                max_new_tokens_cfg if max_new_tokens_cfg is not None else df_group["max_new_tokens"].iloc[0]
            )
            answer_prefix = df_group["answer_prefix"].iloc[0]

            output = self.pipeline(
                context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=self.press,
                max_new_tokens=max_new_tokens,
                max_context_length=max_context_length_cfg,
            )
            self.df.loc[df_group.index, "predicted_answer"] = output["answers"]
            # Store the actual compression ratio used (if the press has one)
            self.df.loc[df_group.index, "compression_ratio"] = self.press.compression_ratio
            torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        logger.info("Inference completed.")

    def _get_save_filenames(self, output_dir: Path) -> Tuple[Path, Path]:
        """
        Generates the unique save filename based on configuration parameters.
        """
        dataset_name = self.config["dataset"]
        data_dir = str(self.config["data_dir"]) if self.config["data_dir"] else ""
        model_name = self.config["model"]
        press_name = self.config["press_name"]
        compression_ratio = self.config["compression_ratio"]
        fraction = self.config["fraction"]
        max_context_length = self.config["max_context_length"]
        compress_questions = self.config["compress_questions"]
        key_channel_compression_ratio = self.config["key_channel_compression_ratio"]

        # Build filename components
        components = [
            dataset_name,
            data_dir,
            model_name.replace("/", "--"),
            press_name,
            f"{compression_ratio:.2f}",
        ]

        if fraction < 1.0:
            components.append(f"fraction{fraction:.2f}".replace(".", "_"))
        if max_context_length is not None:
            components.append(f"max_context{max_context_length}")
        if compress_questions:
            components.append("compressed_questions")

        # Specific handling for ThinKPress or composed presses containing it
        if isinstance(self.press, ThinKPress) or (
            isinstance(self.press, ComposedPress) and any(isinstance(p, ThinKPress) for p in self.press.presses)
        ):
            components.append(f"channel{key_channel_compression_ratio:.2f}".replace(".", "_"))

        filename_stem = "__".join(filter(None, components))  # Filter None/empty strings
        return (
            output_dir / f"{self.predictions_prefix}_{filename_stem}.csv",
            output_dir / f"{self.metrics_prefix}_{filename_stem}.json",
        )

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[["predicted_answer", "compression_ratio"]].to_csv(str(save_filename), index=False)
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config["dataset"]
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = scorer(self.df)

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Average compression ratio: {self.df['compression_ratio'].mean():.2f}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()

        predictions_filename, metrics_filename = self._get_save_filenames(output_dir)
        if predictions_filename.exists() and metrics_filename.exists():
            logger.info(
                f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
            )
            return

        self._load_dataset()
        self._setup_press()
        self._setup_model_pipeline()
        self._prepare_data_for_inference()

        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---


def evaluate_cli(
    config_path: str = "./evaluate_config.yaml",
    dataset: Optional[str] = None,
    data_dir: Optional[str] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    press_name: Optional[str] = None,
    compression_ratio: Optional[float] = None,
    fraction: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    max_context_length: Optional[int] = None,
    compress_questions: Optional[bool] = None,
    key_channel_compression_ratio: Optional[float] = None,
    output_dir: Optional[str] = None,
    log_level: Optional[str] = None,
):
    """
    Evaluate a model on a dataset using a KV-Press and save the results.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file, by default "config/default_config.yaml"
    dataset : str, optional
        Dataset to evaluate (overrides config)
    data_dir : str, optional
        Subdirectory of the dataset to evaluate (overrides config)
    model : str, optional
        Model to use (overrides config)
    device : str, optional
        Model device (overrides config). For multi-GPU use "auto"
    press_name : str, optional
        Press to use (see PRESS_REGISTRY) (overrides config)
    compression_ratio : float, optional
        Compression ratio for the press (overrides config)
    fraction : float, optional
        Fraction of the dataset to evaluate (overrides config)
    max_new_tokens : int, optional
        Maximum number of new tokens to generate (overrides config).
        By default, uses the default for the task (recommended).
    max_context_length : int, optional
        Maximum number of tokens to use in the context (overrides config).
        By default, will use the maximum length supported by the model.
    compress_questions : bool, optional
        Whether to compress the questions as well (overrides config).
        Required for FinchPress.
    key_channel_compression_ratio : float, optional
        Key Channel Compression ratio for the channel press (overrides config).
        Specific to ThinKPress or ComposedPress with ThinKPress.
    output_dir : str, optional
        Directory to save evaluation results (overrides config).
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (overrides config).
    """
    # Collect all CLI arguments that are not None
    cli_args = {k: v for k, v in locals().items() if v is not None and k != "config_path"}
    full_config = load_config(Path(config_path), cli_args)

    runner = EvaluationRunner(full_config)
    runner.run_evaluation()


if __name__ == "__main__":
    Fire(evaluate_cli)
