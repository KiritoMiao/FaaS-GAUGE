"""
Exception classes for the faas_gauge AI client module
"""


class AIClientError(Exception):
    """Base exception for all Unified AI Client errors"""

    pass


class ConfigurationError(AIClientError):
    """Raised when there's an error in configuration"""

    pass


class ProviderError(AIClientError):
    """Raised when there's an error with a specific provider"""

    def __init__(self, message, provider=None, original_error=None):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


class ModelError(AIClientError):
    """Raised when there's an error with a specific model"""

    def __init__(self, message, provider=None, model=None, original_error=None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.original_error = original_error


class AuthenticationError(ProviderError):
    """Raised when authentication fails"""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded"""

    pass


class QuotaExceededError(ProviderError):
    """Raised when quota is exceeded"""

    pass
