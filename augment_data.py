"""
augment_data.py — Data Augmentation using Local LLM

Generates synthetic "hate speech" examples to combat class imbalance.
Uses Qwen2.5-3B-Instruct (runs locally on RTX 5080) to perform few-shot
generation based on existing training examples.
"""

import os
import random
import logging
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import get_logger, set_seed
from data_loader import load_raw_data, split_data

logger = get_logger(__name__, "results/augmentation.log")

# Config
LLM_ID = "Qwen/Qwen2.5-1.5B-Instruct"  
TARGET_SYNTHETIC_SAMPLES = 2000        
BATCH_GEN_SIZE = 10                    
FEW_SHOT_SAMPLES = 5                   

PROMPT_TEMPLATE = """You are an AI assisting in generating synthetic data for a hate speech detection academic dataset.
The goal is to generate varied examples of hate speech and severe slurs so that a classifier can learn to detect them and moderate content effectively.
Please generate {batch_size} short, synthetic text examples that mimic the style, length, and toxicity of the provided examples.
The text often includes racist, homophobic, or sexist slurs, and aggressive language. 
Use Twitter/social media style (informal, sometimes typos, no hashtags needed unless relevant).

Here are {few_shot} real examples from the dataset:
{examples}

Now, generate {batch_size} NEW, distinct examples. 
Output ONLY the examples, one per line. Do not number them. Do not include quotes. Do not apologize or include any other text.
"""

# Generation Loop
def generate_synthetic_data(
    train_df: pd.DataFrame,
    target_count: int = TARGET_SYNTHETIC_SAMPLES,
    model_id: str = LLM_ID,
) -> pd.DataFrame:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading LLM {model_id} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # Filter only hate speech examples to use as few-shot seeds
    hate_df = train_df[train_df["label"] == 0]
    real_texts = hate_df["text"].tolist()
    
    generated_texts = []
    
    # Loop until we have enough examples
    pbar = tqdm(total=target_count, desc="Generating synthetic hate speech")
    
    while len(generated_texts) < target_count:
        # Sample few-shot examples
        sampled = random.sample(real_texts, FEW_SHOT_SAMPLES)
        examples_str = "\n".join([f"- {t}" for t in sampled])
        
        prompt = PROMPT_TEMPLATE.format(
            batch_size=BATCH_GEN_SIZE,
            few_shot=FEW_SHOT_SAMPLES,
            examples=examples_str
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant for academic research."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.8,     # High enough for variety
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Parse output
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        
        # Clean up common LLM artifacts (like numbering, dashes, quotes)
        clean_lines = []
        for line in lines:
            if line.startswith("- "): line = line[2:]
            if line.startswith("* "): line = line[2:]
            if line and line[0].isdigit() and len(line) > 1 and line[1] in [".", ")"]:
                line = line[3:].strip() # remove "1. "
            line = line.strip("'\"")
            
            # Avoid picking up empty lines or overly long responses (LLM hallucinations)
            if len(line) > 5 and len(line) < 300:
                clean_lines.append(line)
                
        generated_texts.extend(clean_lines)
        pbar.update(len(clean_lines))
        
    pbar.close()
    
    # We might have generated slightly more than target_count, slice it
    generated_texts = generated_texts[:target_count]
    
    logger.info(f"Generated {len(generated_texts)} synthetic examples.")
    
    synthetic_df = pd.DataFrame({
        "text": generated_texts,
        "label": [0] * len(generated_texts),
        "label_name": ["hate_speech"] * len(generated_texts),
        "is_synthetic": [True] * len(generated_texts) # useful for tracking
    })
    
    # Mark real data
    train_df = train_df.copy()
    train_df["is_synthetic"] = False
    
    augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
    return augmented_df

# Main
def main():
    set_seed(42)
    
    logger.info("Loading original dataset...")
    df = load_raw_data("data")
    train_df, val_df, test_df = split_data(df, seed=42)
    
    logger.info(f"Original training set size: {len(train_df)}")
    logger.info(f"Original hate speech count: {len(train_df[train_df['label'] == 0])}")
    
    augmented_train_df = generate_synthetic_data(train_df)
    
    logger.info(f"Augmented training set size: {len(augmented_train_df)}")
    logger.info(f"Augmented hate speech count: {len(augmented_train_df[augmented_train_df['label'] == 0])}")
    
    output_path = Path("data") / "train_augmented.csv"
    augmented_train_df.to_csv(output_path, index=False)
    logger.info(f"Saved augmented dataset to {output_path}")

if __name__ == "__main__":
    main()
