# =============================================================================
#  Filename: conftest.py
#
#  Short Description: Pytest configuration and shared fixtures for all tests.
#
#  Creation date: 2025-01-06
#  Author: Asif Qamar
# =============================================================================

import pytest
import sys
import warnings
from pathlib import Path

# Ensure Python 3.12+ compatibility
if sys.version_info < (3, 12):
    pytest.skip("Tests require Python 3.12 or higher", allow_module_level=True)

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Filter warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure pytest
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

# Shared fixtures
@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Get the data directory."""
    return project_root / "data"

@pytest.fixture(scope="session")
def test_images_dir(data_dir: Path) -> Path:
    """Get the test images directory."""
    return data_dir / "small_image_collection"

# Test configuration
pytest_plugins = []

#============================================================================================ 