#!/bin/bash

# Script to build documentation with suppressed warnings
# This suppresses Jupyter deprecation warnings and other noise

# Set environment variables to suppress Jupyter warnings
export JUPYTER_PLATFORM_DIRS=1

# Suppress Python warnings related to jupyter and nbconvert
export PYTHONWARNINGS="ignore::DeprecationWarning:jupyter_core.utils,ignore::DeprecationWarning:traitlets.traitlets,ignore::DeprecationWarning:nbconvert.exporters.templateexporter,ignore::SyntaxWarning:mknotebooks"

# Build the documentation
mkdocs build 