#!/usr/bin/env python3
"""
Test script to validate requirements.txt compatibility
Run this before deploying to catch dependency conflicts early
"""

import subprocess
import sys
import tempfile
import os

def test_requirements_install():
    """Test if requirements.txt can be installed cleanly"""
    print("🧪 Testing requirements.txt installation...")
    
    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = os.path.join(temp_dir, "test_venv")
        
        try:
            # Create virtual environment
            print("📦 Creating temporary virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
            
            # Determine python executable path
            if os.name == 'nt':  # Windows
                python_exe = os.path.join(venv_path, "Scripts", "python.exe")
                pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
            else:  # Unix/Linux/Mac
                python_exe = os.path.join(venv_path, "bin", "python")
                pip_exe = os.path.join(venv_path, "bin", "pip")
            
            # Upgrade pip
            print("⬆️  Upgrading pip...")
            subprocess.run([pip_exe, "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
            
            # Install requirements
            print("📥 Installing requirements...")
            result = subprocess.run([pip_exe, "install", "-r", "requirements.txt"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Requirements installed successfully!")
                
                # Test imports
                print("🔍 Testing critical imports...")
                test_imports = [
                    "import fastapi",
                    "import uvicorn", 
                    "import torch",
                    "import transformers",
                    "import sentence_transformers",
                    "import pinecone",
                    "import groq",
                    "import google.generativeai as genai",
                    "import PyPDF2",
                    "import numpy",
                ]
                
                for import_test in test_imports:
                    try:
                        subprocess.run([python_exe, "-c", import_test], check=True, capture_output=True)
                        print(f"  ✅ {import_test}")
                    except subprocess.CalledProcessError as e:
                        print(f"  ❌ {import_test} - FAILED")
                        return False
                
                print("🎉 All imports successful!")
                return True
            else:
                print("❌ Requirements installation failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during testing: {e}")
            return False

def check_render_config():
    """Check if render.yaml is properly configured"""
    print("\n🔧 Checking render.yaml configuration...")
    
    if not os.path.exists("render.yaml"):
        print("❌ render.yaml not found!")
        return False
    
    with open("render.yaml", "r") as f:
        content = f.read()
        
    required_fields = [
        "services:",
        "type: web",
        "env: python", 
        "buildCommand:",
        "startCommand:",
        "GROQ_API_KEY",
        "PINECONE_API_KEY",
        "GEMINI_API_KEY"
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in content:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Missing required fields in render.yaml: {missing_fields}")
        return False
    else:
        print("✅ render.yaml looks good!")
        return True

def main():
    print("🚀 Pre-deployment validation for Render")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = True
    
    # Test requirements
    if not test_requirements_install():
        success = False
    
    # Check render config
    if not check_render_config():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All checks passed! Ready for Render deployment.")
        print("\n📋 Next steps:")
        print("1. Push your code to GitHub")
        print("2. Connect your repo to Render")
        print("3. Set environment variables in Render dashboard")
        print("4. Deploy!")
    else:
        print("❌ Some checks failed. Please fix the issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()
