# S3PRL JIT-compatible modules
from .s3prl_frontend_jit_standalone import JITCompatibleS3prlFrontendStandalone
from .jit_compatible_attention import JITCompatibleMultiheadAttention

__all__ = [
    'JITCompatibleS3prlFrontendStandalone',
    'JITCompatibleMultiheadAttention',
]
