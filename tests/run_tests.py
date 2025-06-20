# =============================================================================
#  Filename: run_tests.py
#
#  Short Description: Test runner script for rag_to_riches tests.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

"""Test runner script for executing tests with different configurations."""

import sys
import subprocess
from pathlib import Path

def run_tests(test_type: str = "all", verbose: bool = False) -> int:
    """Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run ('all', 'unit', 'integration', 'fast')
        verbose: Whether to run in verbose mode
        
    Returns:
        Exit code from pytest
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Configure based on test type
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend(["--cov=rag_to_riches", "--cov-report=term-missing"])
    except ImportError:
        print("Coverage not available. Install with: pip install pytest-cov")
    
    # Add colored output
    cmd.append("--color=yes")
    
    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run rag_to_riches tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "fast"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run in verbose mode"
    )
    
    args = parser.parse_args()
    
    # Check Python version
    if sys.version_info < (3, 12):
        print("Error: Tests require Python 3.12 or higher")
        print(f"Current version: {sys.version}")
        return 1
    
    # Run tests
    return run_tests(args.type, args.verbose)

if __name__ == "__main__":
    sys.exit(main())

#============================================================================================ 