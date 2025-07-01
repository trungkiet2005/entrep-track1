#!/usr/bin/env python3
"""
Quick runner script for ENT Classification
This script will install requirements and run the main classification
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("Warning: Could not install some requirements. Please install manually.")

def run_classification():
    """Run the main classification script"""
    print("Running ENT classification...")
    try:
        # Import and run the main function
        from ent_classifier import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all requirements are installed.")
    except Exception as e:
        print(f"Error during classification: {e}")

def main():
    """Main runner function"""
    print("="*50)
    print("ENT Image Classification Challenge")
    print("="*50)
    
    # Check if requirements file exists
    if os.path.exists("requirements.txt"):
        install_requirements()
    
    # Run classification
    run_classification()
    
    print("\nDone! Check 'submission.json' for results.")

if __name__ == "__main__":
    main() 