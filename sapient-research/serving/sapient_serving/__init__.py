"""Sapient-1 serving layer.

A thin, security-first hosting wrapper around the Sapient-1 brain encoder.

Design rule: the model weights live ONLY on the server. Clients send a stimulus
over an authenticated API and receive coarse outputs (ROI + parcel summary +
brain image). The model package is never shipped to clients.

Provenance honesty: this serves *Sapient-1* (a clean-room, CC0-trained trimodal
encoder), which is NOT "Mary", the 6-stream research model in the papers. The
/v1/info endpoint states this explicitly. See sapient_serving/engine/provenance.py.
"""

__version__ = "0.1.0"
