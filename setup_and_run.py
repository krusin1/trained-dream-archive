#!/usr/bin/env python3
"""
Setup and Run Script for AI Dream Generator
===========================================

This script provides an easy way to set up and run the AI Dream Generator system.
It handles dependency installation, model training, and server startup.

Usage:
    python setup_and_run.py --setup     # Install dependencies
    python setup_and_run.py --train     # Train the AI model
    python setup_and_run.py --serve     # Start the server
    python setup_and_run.py --all       # Do everything (setup + train + serve)
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if uv is available
    if run_command("uv --version", "Checking for uv package manager"):
        # Use uv for faster installation
        return run_command("uv pip install -e .", "Installing dependencies with uv")
    else:
        # Fallback to pip
        return run_command("pip install -e .", "Installing dependencies with pip")

def check_data_files():
    """Check if dream data files exist."""
    print("\n📁 Checking for dream data files...")
    
    required_files = [
        "dreams_bird_posts.json",
        "dreams_cat_posts.json", 
        "dreams_dog_posts.json",
        "dreams_snake_posts.json",
        "dreams_elephant_posts.json",
        "dreams_flower_posts.json",
        "dreams_rabbit_posts.json",
        "dreams_horse_posts.json",
        "dreams_fish_posts.json",
        "dreams_deer_posts.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        print("Please ensure all dream data JSON files are present in the current directory")
        return False
    
    print(f"✅ Found {len(required_files) - len(missing_files)}/{len(required_files)} data files")
    return True

def train_model():
    """Train the dream generation model."""
    print("\n🤖 Training the dream generation model...")
    
    if not Path("dream_generator.py").exists():
        print("❌ dream_generator.py not found")
        return False
    
    # Train with smaller parameters for faster training
    return run_command(
        "python dream_generator.py --train --epochs 2 --batch-size 2", 
        "Training dream generation model"
    )

def start_server():
    """Start the dream generation server."""
    print("\n🚀 Starting the dream generation server...")
    
    if not Path("dream_api_server.py").exists():
        print("❌ dream_api_server.py not found")
        return False
    
    print("🌐 Starting server on http://localhost:5000")
    print("📱 Open the URL in your browser to start generating dreams!")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run("python dream_api_server.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False

def main():
    """Main setup and run function."""
    parser = argparse.ArgumentParser(description="Setup and run the AI Dream Generator")
    parser.add_argument("--setup", action="store_true", help="Install dependencies")
    parser.add_argument("--train", action="store_true", help="Train the AI model")
    parser.add_argument("--serve", action="store_true", help="Start the server")
    parser.add_argument("--all", action="store_true", help="Do everything (setup + train + serve)")
    parser.add_argument("--check", action="store_true", help="Check system requirements")
    
    args = parser.parse_args()
    
    print("🌟 AI Dream Generator Setup and Run Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check data files
    if not check_data_files():
        return 1
    
    if args.check or args.all or args.setup or args.train or args.serve:
        pass  # Continue with operations
    else:
        parser.print_help()
        return 0
    
    success = True
    
    # Setup dependencies
    if args.setup or args.all:
        if not install_dependencies():
            success = False
    
    # Train model
    if (args.train or args.all) and success:
        if not train_model():
            success = False
    
    # Start server
    if (args.serve or args.all) and success:
        if not start_server():
            success = False
    
    if success:
        print("\n🎉 All operations completed successfully!")
        return 0
    else:
        print("\n❌ Some operations failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
