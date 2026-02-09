# S3PRL JIT-compatible modules
from .s3prl_frontend_jit import JITCompatibleS3prlFrontend
from .jit_compatible_attention import JITCompatibleMultiheadAttention

__all__ = [
    'JITCompatibleS3prlFrontend',
    'JITCompatibleMultiheadAttention',
]
