"""Standardized error handling for yQi evaluation platform."""

import logging
import traceback
from typing import Optional, Any, Callable
from functools import wraps
import streamlit as st

logger = logging.getLogger(__name__)


class YQiError(Exception):
    """Base exception for yQi platform errors."""
    pass


class APIError(YQiError):
    """Error related to API operations."""
    pass


class FileError(YQiError):
    """Error related to file operations."""
    pass


class ValidationError(YQiError):
    """Error related to data validation."""
    pass


def handle_errors(error_type: str = "general", show_user: bool = True):
    """Decorator for standardized error handling.
    
    Args:
        error_type: Type of operation for context
        show_user: Whether to show error to user via Streamlit
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {error_type} operation: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
                if show_user:
                    st.error(error_msg)
                
                raise
        return wrapper
    return decorator


def log_and_display_error(error: Exception, context: str, show_user: bool = True) -> None:
    """Log error and optionally display to user.
    
    Args:
        error: Exception that occurred
        context: Context where error occurred
        show_user: Whether to show error to user
    """
    error_msg = f"Error in {context}: {str(error)}"
    logger.error(f"{error_msg}\n{traceback.format_exc()}")
    
    if show_user:
        st.error(error_msg)


def safe_execute(func: Callable, context: str, default_return: Any = None, 
                show_user: bool = True) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        context: Context description for error messages
        default_return: Value to return on error
        show_user: Whether to show errors to user
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        log_and_display_error(e, context, show_user)
        return default_return


def validate_required_keys(data: dict, required_keys: list, context: str = "data") -> None:
    """Validate that dictionary contains required keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required keys
        context: Context for error messages
        
    Raises:
        ValidationError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValidationError(f"Missing required keys in {context}: {missing_keys}")


def validate_file_exists(filepath: str) -> None:
    """Validate that file exists.
    
    Args:
        filepath: Path to file
        
    Raises:
        FileError: If file doesn't exist
    """
    import os
    if not os.path.exists(filepath):
        raise FileError(f"File not found: {filepath}")
