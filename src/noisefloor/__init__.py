"""noisefloor: random-matrix diagnostics for neural-network weights."""

from .denoise import denoise, low_rank_params, shrink_singular_values, truncate
from .mp import mp_cdf, mp_edges, mp_median, mp_pdf, mp_quantile
from .spectrum import SpectrumReport, analyze, analyze_model, as_matrix, format_reports
from .tw import edge_threshold, johnstone_centre_scale, tw1_cdf, tw1_pdf, tw1_ppf

__version__ = "0.1.0"

__all__ = [
    "SpectrumReport",
    "analyze",
    "analyze_model",
    "as_matrix",
    "denoise",
    "edge_threshold",
    "format_reports",
    "johnstone_centre_scale",
    "low_rank_params",
    "mp_cdf",
    "mp_edges",
    "mp_median",
    "mp_pdf",
    "mp_quantile",
    "shrink_singular_values",
    "truncate",
    "tw1_cdf",
    "tw1_pdf",
    "tw1_ppf",
]
