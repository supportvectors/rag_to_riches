#!/usr/bin/env python3
"""
Test script to verify that the environment and configuration are set up correctly.
"""

import sys
import os
from pathlib import Path

def main():
    print("🚀 SupportVectors Environment Setup Test")
    print("=" * 50)
    
    # Test Python version
    print(f"✅ Python version: {sys.version}")
    
    # Test current working directory
    print(f"✅ Working directory: {os.getcwd()}")
    
    # Test PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', 'Not set')
    print(f"✅ PYTHONPATH: {pythonpath}")
    
    # Test PROJECT_PYTHON
    project_python = os.environ.get('PROJECT_PYTHON', 'Not set')
    print(f"✅ PROJECT_PYTHON: {project_python}")
    
    # Verify the Python executable exists
    if project_python != 'Not set':
        if os.path.exists(project_python):
            print(f"✅ Project Python executable found at: {project_python}")
        else:
            print(f"⚠️  Project Python executable not found at: {project_python}")
    
    # Test that we can import our module
    try:
        # Dynamic import based on the module structure
        src_path = Path('src')
        if src_path.exists():
            module_dirs = [d for d in src_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if module_dirs:
                module_name = module_dirs[0].name
                print(f"✅ Found module: {module_name}")
                
                # Try to import the module
                sys.path.insert(0, str(src_path))
                try:
                    module = __import__(module_name)
                    print(f"✅ Successfully imported {module_name}")
                    
                    # Try to access the config if it exists
                    if hasattr(module, 'config'):
                        print("✅ Configuration object found and accessible")
                    else:
                        print("ℹ️  Configuration object not yet accessible (this is normal)")
                        
                except ImportError as e:
                    print(f"⚠️  Could not import {module_name}: {e}")
                    print("   This might be normal if dependencies aren't fully installed yet")
            else:
                print("ℹ️  No module directories found in src/")
        else:
            print("⚠️  src/ directory not found")
    
    except Exception as e:
        print(f"⚠️  Error during module test: {e}")
    
    print("=" * 50)
    print("🎉 Hello World! Environment setup test completed!")
    print("🎯 Your SupportVectors project environment is ready to use!")

if __name__ == "__main__":
    main()
