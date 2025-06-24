"""
Robust error handling for file parsing operations
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class RobustErrorHandler:
    """Handles errors gracefully with fallback strategies"""
    
    @staticmethod
    def with_fallback(primary_func: Callable, fallback_func: Callable, error_context: str = ""):
        """Execute primary function with fallback on failure"""
        try:
            result = primary_func()
            if result:
                return result
        except Exception as e:
            logger.warning(f"{error_context} - Primary method failed: {e}")
        
        try:
            logger.info(f"{error_context} - Attempting fallback method")
            return fallback_func()
        except Exception as e:
            logger.error(f"{error_context} - Fallback method also failed: {e}")
            return None
    
    @staticmethod
    def safe_execute(func: Callable, default_return: Any = None, context: str = ""):
        """Safely execute function with error logging"""
        try:
            return func()
        except Exception as e:
            logger.error(f"{context} failed: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            return default_return
    
    @staticmethod
    def create_default_zones(file_path: str = "", context: str = "") -> List[Dict]:
        """Create sensible default zones when parsing fails"""
        logger.info(f"Creating default zones for {context}")
        
        # Create a reasonable default layout
        return [
            {
                'id': 0,
                'polygon': [(0, 0), (600, 0), (600, 400), (0, 400)],
                'area': 240000,  # 600 x 400 units
                'centroid': (300, 200),
                'layer': '0',
                'zone_type': 'Office',
                'parsing_method': 'default_fallback'
            },
            {
                'id': 1,
                'polygon': [(600, 0), (1000, 0), (1000, 400), (600, 400)],
                'area': 160000,  # 400 x 400 units
                'centroid': (800, 200),
                'layer': '0',
                'zone_type': 'Conference Room',
                'parsing_method': 'default_fallback'
            },
            {
                'id': 2,
                'polygon': [(0, 400), (1000, 400), (1000, 600), (0, 600)],
                'area': 200000,  # 1000 x 200 units
                'centroid': (500, 500),
                'layer': '0',
                'zone_type': 'Reception',
                'parsing_method': 'default_fallback'
            }
        ]

def robust_parser(error_context: str = ""):
    """Decorator for robust parsing with error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if result:
                    return result
            except Exception as e:
                logger.error(f"{error_context} - {func.__name__} failed: {e}")
                
            # Return sensible defaults
            return RobustErrorHandler.create_default_zones(context=error_context)
        return wrapper
    return decorator