# =============================================================================
#  Filename: logging_config.py
#
#  Short Description: Utility functions for configuring loguru logging levels
#
#  Creation date: 2025-01-21
#  Author: Asif Qamar
# =============================================================================

from loguru import logger
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

def configure_logging(level: LogLevel = "INFO", 
                     format_string: str = None,
                     colorize: bool = True) -> None:
    """
    Configure loguru logging for cleaner output in notebooks and applications.
    
    Args:
        level: Logging level to set. Defaults to "INFO" for cleaner output.
        format_string: Custom format string for log messages.
        colorize: Whether to use colored output. Defaults to True.
    
    Examples:
        >>> configure_logging("INFO")  # Clean output, no DEBUG messages
        >>> configure_logging("DEBUG")  # Show all messages including DEBUG
        >>> configure_logging("WARNING")  # Only show warnings and errors
    """
    # Remove existing handlers
    logger.remove()
    
    # Set default format if not provided
    if format_string is None:
        if level == "DEBUG":
            # More detailed format for debug level
            format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        else:
            # Cleaner format for other levels
            format_string = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
    
    # Add new handler with specified configuration
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=colorize
    )
    
    # Log the configuration
    logger.info(f"Logging configured with level: {level}")

def set_info_level() -> None:
    """Quick function to set logging to INFO level for clean notebook output."""
    configure_logging("INFO")

def set_debug_level() -> None:
    """Quick function to set logging to DEBUG level for detailed output."""
    configure_logging("DEBUG")

def set_warning_level() -> None:
    """Quick function to set logging to WARNING level for minimal output."""
    configure_logging("WARNING")

# Auto-configure with INFO level when module is imported
configure_logging("INFO") 