"""
train_gesture_story_model.py
================================

This script provides an end‑to‑end pipeline for preparing a custom text‑to‑text
dataset from the ``dreams_*_posts.json`` files in this repository and
fine‑tuning a Hugging Face seq2seq model (e.g. T5) to generate bedtime stories
conditioned on a sequence of three hand‑gesture animals.  The resulting model
can be queried with prompts like ``"cat, dog, rabbit"`` to produce a
cohesive dream narrative that weaves together elements from each of the three
animals.  See the accompanying README or the main program's docstring for
usage instructions.

The pipeline has three main stages:

1. **Data loading** – Reads each JSON file into a dictionary keyed by animal
   names.  Each entry contains a list of posts with ``title``, ``body`` and
   optional ``images`` fields.
2. **Example synthesis** – Constructs training examples by sampling three
   distinct animals and concatenating one random post from each into a single
   story.  The model input is the comma‑separated list of animals and the
   target output is the concatenated story.
3. **Model fine‑tuning** – Uses the Hugging Face ``transformers`` library to
   tokenize the examples and fine‑tune a pre‑trained model.  By default this
   script uses ``t5-small``, but you can override the model name via a
   command‑line flag.  Training hyper‑parameters are exposed as arguments.

The script relies on the ``datasets`` and ``transformers`` libraries.  To
install them along with PyTorch you can run the following command before
executing the script (this assumes a working internet connection and that
PyPI is reachable):

    pip install transformers datasets torch

For more information on the typical fine‑tuning workflow, see Hugging Face's
official guide to fine‑tuning GPT‑2 models【690628470933217†L49-L71】【690628470933217†L120-L156】.  Although this
script uses a seq2seq model instead of GPT‑2, the overall steps—installing
dependencies, loading and tokenizing the dataset, defining training arguments
and invoking a Trainer—are the same【690628470933217†L120-L156】.

"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

try:
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except ImportError as e:
    raise ImportError(
        "This script requires the 'datasets' and 'transformers' libraries. "
        "Install them with `pip install transformers datasets torch` before "
        "running." 
    ) from e


def load_posts(data_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """Load all ``dreams_*_posts.json`` files from ``data_dir``.

    Each JSON file is expected to contain a list of dictionaries with keys
    ``title`` and ``body``.  The animal name is extracted from the file
    name (e.g. ``dreams_cat_posts.json`` -> ``cat``).

    Args:
        data_dir: Directory containing the JSON files.

    Returns:
        A dict mapping animal names to lists of post dictionaries.
    """
    animal_posts: Dict[str, List[Dict[str, str]]] = {}
    for filename in os.listdir(data_dir):
        if filename.startswith("dreams_") and filename.endswith("_posts.json"):
            animal = filename[len("dreams_") : -len("_posts.json")]
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    posts = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON from {path}: {e}")
            # Ensure that each post has the expected fields
            cleaned_posts: List[Dict[str, str]] = []
            for post in posts:
                title = post.get("title", "").strip()
                body = post.get("body", "").strip()
                if not title and not body:
                    continue
                cleaned_posts.append({"title": title, "body": body})
            if not cleaned_posts:
                raise ValueError(f"No valid posts found in {path}")
            animal_posts[animal] = cleaned_posts
    if not animal_posts:
        raise FileNotFoundError(
            f"No dream post files found in {data_dir}. Expected files named like 'dreams_cat_posts.json'."
        )
    return animal_posts


def build_training_examples(
    animal_posts: Dict[str, List[Dict[str, str]]],
    num_examples: int = 1000,
    unique_animals: bool = True,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Synthesize a list of training examples from the loaded posts.

    Each example consists of an ``input`` field (comma‑separated list of three
    animals) and an ``output`` field (concatenated story from sampled posts).

    Args:
        animal_posts: Mapping of animal names to lists of post dictionaries.
        num_examples: Number of examples to generate.
        unique_animals: If true, sample three distinct animals per example.
        seed: RNG seed for reproducibility.

    Returns:
        A list of dictionaries with ``input`` and ``output`` keys.
    """
    rng = random.Random(seed)
    animals = list(animal_posts.keys())
    examples: List[Dict[str, str]] = []
    for _ in range(num_examples):
        if unique_animals and len(animals) >= 3:
            triple = rng.sample(animals, 3)
        else:
            triple = [rng.choice(animals) for _ in range(3)]
        story_parts: List[str] = []
        for animal in triple:
            post = rng.choice(animal_posts[animal])
            story_parts.append(post["title"] + "\n" + post["body"])
        prompt = ", ".join(triple)
        story = "\n\n".join(story_parts)
        examples.append({"input": prompt, "output": story})
    return examples


def tokenize_function(
    examples,
    tokenizer,
    max_input_length: int = 32,
    max_target_length: int = 512,
):
    """Tokenize and prepare model inputs and labels for seq2seq training."""
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def fine_tune_model(
    dataset: DatasetDict,
    model_name: str = "t5-small",
    output_dir: str = "./gesture_story_model",
    num_train_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Fine‑tune a seq2seq model on the provided dataset.

    The dataset must contain ``input`` and ``output`` columns.  After
    fine‑tuning, the model and tokenizer are saved to ``output_dir``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Tokenize the entire dataset
    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=["input", "output"],
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if eval_ratio > 0 else "no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        seed=seed,
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("test"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    # Save the fine‑tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Fine‑tune a model to generate stories from gesture triples.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing dreams_*_posts.json files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="t5-small",
        help="Name or path of the pre‑trained seq2seq model (e.g. 't5-small', 't5-base', 'google/flan-t5-base').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gesture_story_model",
        help="Where to store the fine‑tuned model and logs.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1000,
        help="Number of synthetic training examples to generate.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per‑device batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Fraction of examples to reserve for evaluation (0 disables evaluation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load the dream posts
    posts = load_posts(args.data_dir)
    # Build synthetic examples
    examples = build_training_examples(
        animal_posts=posts,
        num_examples=args.num_examples,
        unique_animals=True,
        seed=args.seed,
    )
    # Create a Hugging Face Dataset and split into train/test
    df = pd.DataFrame(examples)
    dataset = Dataset.from_pandas(df)
    if args.eval_ratio > 0:
        dataset_dict = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    else:
        dataset_dict = DatasetDict({"train": dataset})
    # Fine‑tune the model
    fine_tune_model(
        dataset=dataset_dict,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()