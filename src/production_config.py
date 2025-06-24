"""
Production configuration for AI Architectural Space Analyzer PRO
Handles environment setup, security, and performance optimizations
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ProductionConfig:
    """Production configuration settings"""
    
    # Server Configuration
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 5000
    DEBUG: bool = False
    
    # Security Configuration
    ENABLE_CORS: bool = False
    ENABLE_XSRF: bool = False
    MAX_UPLOAD_SIZE: int = 200  # MB
    MAX_MESSAGE_SIZE: int = 200  # MB
    
    # Database Configuration
    DATABASE_URL: Optional[str] = None
    DATABASE_POOL_SIZE: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Performance Configuration
    CACHE_TTL: int = 300  # seconds
    ENABLE_COMPRESSION: bool = True
    WORKER_THREADS: int = 4
    
    # AI Configuration
    AI_BATCH_SIZE: int = 10
    AI_TIMEOUT: int = 30
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # File Processing Configuration
    TEMP_DIR: str = "/tmp/dwg_analyzer"
    MAX_FILE_SIZE: int = 200 * 1024 * 1024  # 200MB
    SUPPORTED_FORMATS: list = None
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    def __post_init__(self):
        """Initialize default values and environment variables"""
        if self.SUPPORTED_FORMATS is None:
            self.SUPPORTED_FORMATS = ['.dwg', '.dxf', '.pdf']
        
        # Load from environment variables
        self.DATABASE_URL = os.getenv('DATABASE_URL')
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        # Ensure temp directory exists
        os.makedirs(self.TEMP_DIR, exist_ok=True)

class SecurityManager:
    """Handles security validations and sanitization"""
    
    @staticmethod
    def validate_file_upload(filename: str, file_size: int, config: ProductionConfig) -> bool:
        """Validate uploaded file for security and size constraints"""
        
        # Check file size
        if file_size > config.MAX_FILE_SIZE:
            return False
        
        # Check file extension
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in config.SUPPORTED_FORMATS:
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', '<', '>', '|', ':', '*', '?', '"']
        if any(char in filename for char in dangerous_chars):
            return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        import re
        # Remove dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        return sanitized

class PerformanceOptimizer:
    """Handles performance optimizations and monitoring"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.metrics = {}
    
    def optimize_memory(self):
        """Apply memory optimizations"""
        import gc
        gc.collect()
    
    def track_performance(self, operation: str, duration: float):
        """Track operation performance"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for operation, times in self.metrics.items():
            if times:
                stats[operation] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_calls': len(times)
                }
        return stats

class DatabaseManager:
    """Production database manager with connection pooling"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.pool = None
        self._setup_database()
    
    def _setup_database(self):
        """Setup database connection with production settings"""
        try:
            if self.config.DATABASE_URL:
                from sqlalchemy import create_engine
                from sqlalchemy.pool import QueuePool
                
                self.engine = create_engine(
                    self.config.DATABASE_URL,
                    poolclass=QueuePool,
                    pool_size=self.config.DATABASE_POOL_SIZE,
                    pool_timeout=self.config.DATABASE_POOL_TIMEOUT,
                    pool_recycle=3600,  # 1 hour
                    echo=False
                )
                logging.info("PostgreSQL database connected")
            else:
                # Fallback to SQLite
                from sqlalchemy import create_engine
                self.engine = create_engine('sqlite:///dwg_analyzer.db', echo=False)
                logging.info("SQLite database connected (fallback)")
                
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            raise
    
    def get_connection(self):
        """Get database connection from pool"""
        return self.engine.connect()
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            with self.get_connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

# Global configuration instance
config = ProductionConfig()
security = SecurityManager()
performance = PerformanceOptimizer(config)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Production configuration initialized")