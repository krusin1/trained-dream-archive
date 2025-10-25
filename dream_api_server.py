#!/usr/bin/env python3
"""
Dream API Server
================

A Flask-based API server that provides endpoints for the dream generation system.
This server handles requests from the enhanced gesture detection interface and
generates new dreams based on combinations of two animals.

Endpoints:
- POST /generate_dream - Generate dream for specific animals
- GET /generate_random_dream - Generate random dream
- GET /health - Health check
- GET /animals - List available animals
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
import random
from typing import Dict, List, Optional
import logging

# Import our dream generator
from dream_generator import DreamGenerator, ANIMALS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the dream generator
dream_generator = None

def initialize_dream_generator():
    """Initialize the dream generator with fallback options."""
    global dream_generator
    
    try:
        # Try to load the trained model
        dream_generator = DreamGenerator("./dream_model")
        dream_generator.load_model()
        logger.info("Successfully loaded trained dream generation model")
    except FileNotFoundError:
        logger.warning("Trained model not found, using fallback dream generation")
        dream_generator = FallbackDreamGenerator()
    except Exception as e:
        logger.error(f"Error loading model: {e}, using fallback")
        dream_generator = FallbackDreamGenerator()

class FallbackDreamGenerator:
    """Fallback dream generator that creates dreams from existing data when the AI model isn't available."""
    
    def __init__(self):
        self.dream_data = {}
        self.load_dream_data()
    
    def load_dream_data(self):
        """Load existing dream data from JSON files."""
        for animal in ANIMALS:
            json_file = f"dreams_{animal}_posts.json"
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.dream_data[animal] = json.load(f)
        
        logger.info(f"Loaded dream data for {len(self.dream_data)} animals")
    
    def generate_dream(self, animal1: str, animal2: str) -> Dict[str, str]:
        """Generate a combined dream using existing data."""
        # Get random posts for each animal
        post1 = random.choice(self.dream_data.get(animal1, [{}]))
        post2 = random.choice(self.dream_data.get(animal2, [{}]))
        
        # Extract key elements from each post
        elements1 = self._extract_dream_elements(post1.get('body', ''), animal1)
        elements2 = self._extract_dream_elements(post2.get('body', ''), animal2)
        
        # Create a combined narrative
        combined_dream = self._create_combined_narrative(elements1, elements2, animal1, animal2)
        
        return {
            'title': f"Dream of {animal1.title()} and {animal2.title()}",
            'body': combined_dream,
            'animals': [animal1, animal2],
            'type': 'fallback'
        }
    
    def _extract_dream_elements(self, text: str, animal: str) -> str:
        """Extract key elements from dream text."""
        if not text:
            return f"In my dream, I saw a {animal}."
        
        # Take the first meaningful sentence or two
        sentences = text.split('. ')
        meaningful_sentences = []
        
        for sentence in sentences[:3]:
            if len(sentence.strip()) > 20:
                meaningful_sentences.append(sentence.strip())
        
        if meaningful_sentences:
            return '. '.join(meaningful_sentences) + '.'
        else:
            return text[:200] + '...' if len(text) > 200 else text
    
    def _create_combined_narrative(self, elements1: str, elements2: str, animal1: str, animal2: str) -> str:
        """Create a combined narrative from two dream elements."""
        transitions = [
            "Then suddenly, the dream shifted...",
            "As the dream continued, everything changed...",
            "But then something unexpected happened...",
            "The dream took an unexpected turn...",
            "Suddenly, the scene transformed...",
            "In the next moment, the dream evolved...",
            "Then, as if by magic, everything changed..."
        ]
        
        transition = random.choice(transitions)
        
        return f"{elements1}\n\n{transition} {elements2}"

@app.route('/')
def index():
    """Serve the enhanced HTML interface."""
    return render_template_string(open('index_enhanced.html').read())

@app.route('/generate_dream', methods=['POST'])
def generate_dream_api():
    """API endpoint for generating dreams from gesture combinations."""
    try:
        data = request.get_json()
        animal1 = data.get('animal1')
        animal2 = data.get('animal2')
        
        if not animal1 or not animal2:
            return jsonify({'error': 'Both animal1 and animal2 are required'}), 400
        
        if animal1 not in ANIMALS or animal2 not in ANIMALS:
            return jsonify({'error': f'Animals must be one of {ANIMALS}'}), 400
        
        if animal1 == animal2:
            return jsonify({'error': 'Animals must be different'}), 400
        
        logger.info(f"Generating dream for {animal1} and {animal2}")
        
        dream = dream_generator.generate_dream(animal1, animal2)
        
        logger.info(f"Successfully generated dream: {dream['title']}")
        return jsonify(dream)
        
    except Exception as e:
        logger.error(f"Error generating dream: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_random_dream', methods=['GET'])
def generate_random_dream_api():
    """API endpoint for generating random dreams."""
    try:
        animal1, animal2 = random.sample(ANIMALS, 2)
        logger.info(f"Generating random dream for {animal1} and {animal2}")
        
        dream = dream_generator.generate_dream(animal1, animal2)
        
        logger.info(f"Successfully generated random dream: {dream['title']}")
        return jsonify(dream)
        
    except Exception as e:
        logger.error(f"Error generating random dream: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'animals': ANIMALS,
        'generator_type': type(dream_generator).__name__
    })

@app.route('/animals', methods=['GET'])
def get_animals():
    """Get list of available animals."""
    return jsonify({
        'animals': ANIMALS,
        'count': len(ANIMALS)
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    stats = {
        'total_animals': len(ANIMALS),
        'available_combinations': len(ANIMALS) * (len(ANIMALS) - 1),
        'generator_type': type(dream_generator).__name__
    }
    
    if hasattr(dream_generator, 'dream_data'):
        stats['loaded_datasets'] = len(dream_generator.dream_data)
    
    return jsonify(stats)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Initializing Dream API Server...")
    initialize_dream_generator()
    
    print("Starting Dream Generator API server on http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /                    - Enhanced HTML interface")
    print("  POST /generate_dream      - Generate dream for specific animals")
    print("  GET  /generate_random_dream - Generate random dream")
    print("  GET  /health              - Health check")
    print("  GET  /animals             - List available animals")
    print("  GET  /stats               - System statistics")
    print()
    print("Open http://localhost:5000 in your browser to start generating dreams!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)



