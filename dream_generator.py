#!/usr/bin/env python3
"""
Dream Generator with Gesture Combination
=======================================

This module provides a complete pipeline for generating new dreams based on
combinations of two gestures detected by the hand pose system. It uses a
fine-tuned Hugging Face model to generate coherent dream narratives that
combine elements from two different animals.

The system works as follows:
1. Load and preprocess dream data from JSON files
2. Train a text generation model on the dream corpus
3. Provide an API endpoint for generating dreams from gesture combinations
4. Integrate with the existing gesture detection frontend

Usage:
    python dream_generator.py --train  # Train the model
    python dream_generator.py --serve  # Start the API server
"""

import argparse
import json
import os
import random
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset
import pandas as pd

# Animal mapping from gesture detection
ANIMALS = ["deer", "cat", "dog", "snake", "elephant", "flower", "rabbit", "horse", "fish", "bird"]

@dataclass
class DreamPost:
    """Represents a single dream post with metadata."""
    title: str
    body: str
    animal: str
    images: List[str]

class DreamDataProcessor:
    """Handles loading and preprocessing of dream data."""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.dreams: List[DreamPost] = []
        
    def load_dreams(self) -> List[DreamPost]:
        """Load all dream posts from JSON files."""
        print("Loading dream data...")
        
        for animal in ANIMALS:
            json_file = self.data_dir / f"dreams_{animal}_posts.json"
            if not json_file.exists():
                print(f"Warning: {json_file} not found, skipping {animal}")
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
                posts = json.load(f)
                
            for post in posts:
                dream = DreamPost(
                    title=post.get('title', '').strip(),
                    body=post.get('body', '').strip(),
                    animal=animal,
                    images=post.get('images', [])
                )
                if dream.title or dream.body:  # Only include non-empty posts
                    self.dreams.append(dream)
        
        print(f"Loaded {len(self.dreams)} dream posts")
        return self.dreams
    
    def create_training_data(self) -> List[Dict[str, str]]:
        """Create training examples for the model."""
        print("Creating training data...")
        
        training_examples = []
        
        # Single animal dreams (for baseline learning)
        for dream in self.dreams:
            if len(dream.title) > 10 and len(dream.body) > 50:
                text = f"{dream.title}\n\n{dream.body}"
                training_examples.append({
                    'text': text,
                    'animals': [dream.animal],
                    'type': 'single'
                })
        
        # Combined animal dreams (for combination learning)
        animal_dreams = {animal: [d for d in self.dreams if d.animal == animal] 
                        for animal in ANIMALS}
        
        for _ in range(2000):  # Generate 2000 combination examples
            # Sample two different animals
            animal1, animal2 = random.sample(ANIMALS, 2)
            dream1 = random.choice(animal_dreams[animal1])
            dream2 = random.choice(animal_dreams[animal2])
            
            # Create a combined dream narrative
            combined_text = self._create_combined_dream(dream1, dream2, animal1, animal2)
            training_examples.append({
                'text': combined_text,
                'animals': [animal1, animal2],
                'type': 'combined'
            })
        
        print(f"Created {len(training_examples)} training examples")
        return training_examples
    
    def _create_combined_dream(self, dream1: DreamPost, dream2: DreamPost, 
                             animal1: str, animal2: str) -> str:
        """Create a coherent combined dream from two different animal dreams."""
        
        # Extract key elements from each dream
        elements1 = self._extract_dream_elements(dream1.body, animal1)
        elements2 = self._extract_dream_elements(dream2.body, animal2)
        
        # Create transition phrases
        transitions = [
            "Then suddenly, the dream shifted...",
            "As the dream continued, everything changed...",
            "But then something unexpected happened...",
            "The dream took an unexpected turn...",
            "Suddenly, the scene transformed..."
        ]
        
        # Combine the dreams with a transition
        combined = f"In my dream, {elements1}\n\n{random.choice(transitions)} {elements2}"
        
        # Add a title that reflects both animals
        title = f"Dream about {animal1} and {animal2}"
        return f"{title}\n\n{combined}"

    def _extract_dream_elements(self, text: str, animal: str) -> str:
        """Extract and clean dream elements, focusing on the core narrative."""
        # Remove Reddit-specific content
        text = re.sub(r'ANNOUNCEMENT:.*?(?=\n\n|\n[A-Z])', '', text, flags=re.DOTALL)
        text = re.sub(r'I am a bot.*', '', text, flags=re.DOTALL)
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        
        # Clean up the text
        sentences = text.split('. ')
        # Take the first 3-5 meaningful sentences
        meaningful_sentences = []
        for sentence in sentences[:5]:
            if len(sentence.strip()) > 20 and animal.lower() in sentence.lower():
                meaningful_sentences.append(sentence.strip())
        
        if meaningful_sentences:
            return '. '.join(meaningful_sentences) + '.'
        else:
            # Fallback: return first part of the text
            return text[:200] + '...' if len(text) > 200 else text

class DreamModelTrainer:
    """Handles training of the dream generation model."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def prepare_model(self):
        """Load and prepare the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Resize model embeddings if necessary
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def train(self, training_data: List[Dict[str, str]], 
              output_dir: str = "./dream_model",
              epochs: int = 3,
              batch_size: int = 4):
        """Train the model on the dream data."""
        print("Training the dream generation model...")
        
        # Prepare the dataset
        texts = [example['text'] for example in training_data]
        dataset = Dataset.from_dict({'text': texts})
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            save_steps=500,
            eval_strategy="no",
            save_total_limit=2,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model trained and saved to {output_dir}")

class DreamGenerator:
    """Generates new dreams based on animal combinations."""
    
    def __init__(self, model_path: str = "./dream_model"):
        self.model_path = model_path
        self.generator = None
        self.data_processor = DreamDataProcessor()
        
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
        
        print("Loading trained dream generation model...")
        self.generator = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.model_path,
            max_length=300,
            temperature=0.8,
            do_sample=True,
            pad_token_id=50256
        )
        
        # Load the original dream data for reference
        self.data_processor.load_dreams()
        
    def generate_dream(self, animal1: str, animal2: str) -> str:
        """Generate a new dream combining two animals."""
        if not self.generator:
            self.load_model()
        
        # Create a prompt that combines both animals
        prompt = f"In my dream, I encountered a {animal1} and a {animal2}. "
        
        # Generate the dream
        result = self.generator(prompt, max_length=250, num_return_sequences=1)
        generated_text = result[0]['generated_text']
        
        # Clean up the generated text
        dream_text = self._clean_generated_text(generated_text, prompt)
        
        return {
            'title': f"Dream of {animal1} and {animal2}",
            'body': dream_text,
            'animals': [animal1, animal2]
        }
    
    def generate_random_dream(self) -> str:
        """Generate a dream from two random animals."""
        animal1, animal2 = random.sample(ANIMALS, 2)
        return self.generate_dream(animal1, animal2)
    
    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """Clean and format the generated dream text."""
        # Remove the prompt from the beginning
        if text.startswith(prompt):
            text = text[len(prompt):]
        
        # Remove incomplete sentences at the end
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Remove the last sentence if it's incomplete
            last_sentence = sentences[-1]
            if len(last_sentence) < 20 or not last_sentence.endswith('.'):
                sentences = sentences[:-1]
        
        # Join sentences and clean up
        cleaned_text = '. '.join(sentences)
        if cleaned_text and not cleaned_text.endswith('.'):
            cleaned_text += '.'
        
        return cleaned_text.strip()

def main():
    """Main function for training and serving the dream generator."""
    parser = argparse.ArgumentParser(description="Dream Generator with Gesture Combination")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--serve", action="store_true", help="Start the API server")
    parser.add_argument("--generate", nargs=2, metavar=('ANIMAL1', 'ANIMAL2'), 
                       help="Generate a dream for two animals")
    parser.add_argument("--model-path", default="./dream_model", 
                       help="Path to save/load the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    
    args = parser.parse_args()
    
    if args.train:
        # Train the model
        print("Starting model training...")
        
        # Load and process data
        processor = DreamDataProcessor()
        dreams = processor.load_dreams()
        training_data = processor.create_training_data()
        
        # Train the model
        trainer = DreamModelTrainer()
        trainer.prepare_model()
        trainer.train(training_data, args.model_path, args.epochs, args.batch_size)
        
        print("Training completed!")
        
    elif args.generate:
        # Generate a specific dream
        animal1, animal2 = args.generate
        if animal1 not in ANIMALS or animal2 not in ANIMALS:
            print(f"Error: Animals must be one of {ANIMALS}")
            return
        
        generator = DreamGenerator(args.model_path)
        dream = generator.generate_dream(animal1, animal2)
        
        print(f"\n=== {dream['title']} ===")
        print(dream['body'])
        
    elif args.serve:
        # Start the API server
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)
        
        # Initialize the generator
        generator = DreamGenerator(args.model_path)
        
        @app.route('/generate_dream', methods=['POST'])
        def generate_dream_api():
            """API endpoint for generating dreams from gesture combinations."""
            data = request.get_json()
            animal1 = data.get('animal1')
            animal2 = data.get('animal2')
            
            if not animal1 or not animal2:
                return jsonify({'error': 'Both animal1 and animal2 are required'}), 400
            
            if animal1 not in ANIMALS or animal2 not in ANIMALS:
                return jsonify({'error': f'Animals must be one of {ANIMALS}'}), 400
            
            try:
                dream = generator.generate_dream(animal1, animal2)
                return jsonify(dream)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/generate_random_dream', methods=['GET'])
        def generate_random_dream_api():
            """API endpoint for generating random dreams."""
            try:
                dream = generator.generate_random_dream()
                return jsonify(dream)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'animals': ANIMALS})
        
        print("Starting Dream Generator API server on http://localhost:5000")
        print("Available endpoints:")
        print("  POST /generate_dream - Generate dream for specific animals")
        print("  GET /generate_random_dream - Generate random dream")
        print("  GET /health - Health check")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()



