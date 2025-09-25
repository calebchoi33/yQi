"""Custom exceptions for yQi evaluation platform."""


class YQiException(Exception):
    """Base exception for yQi platform."""
    pass


class DatabaseError(YQiException):
    """Database-related errors."""
    pass


class APIError(YQiException):
    """API-related errors."""
    pass


class ConfigurationError(YQiException):
    """Configuration-related errors."""
    pass


class ValidationError(YQiException):
    """Data validation errors."""
    pass
