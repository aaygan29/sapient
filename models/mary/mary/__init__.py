"""mary — faithful 6-stream ORCLE-Nano brain encoder package."""
from .model import MaryConfig, MaryModel
from .losses import composite_loss, masked_mse_loss, negcorr_loss, infonce_loss
from .metrics import vertex_pearson, parcel_pearson, pearson_r
from .adapter import STREAM_DIMS, STREAM_ORDER
from .head import VertexHead

__version__ = "0.1.0"
__all__ = [
    "MaryConfig",
    "MaryModel",
    "composite_loss",
    "masked_mse_loss",
    "negcorr_loss",
    "infonce_loss",
    "vertex_pearson",
    "parcel_pearson",
    "pearson_r",
    "STREAM_DIMS",
    "STREAM_ORDER",
    "VertexHead",
    "__version__",
]
