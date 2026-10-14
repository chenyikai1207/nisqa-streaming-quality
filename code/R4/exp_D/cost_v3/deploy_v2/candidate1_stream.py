"""Candidate1 minimalstream factory: same original NumPyFIR/stream checked against candidate1 whole.
No Torch import/wholeGRU container, no repeatedfullwave history. WholeTorch RSS is measured separately.
"""
from .numpy_backend import Backend as OriginalBackend
class Backend(OriginalBackend):
    def __init__(self,candidate='v2_pure',precision='float32'):
        if precision!='float32':raise ValueError('Candidate1 qualification is float32 only')
        super().__init__(candidate,precision)
        self.implementation_version='candidate1_stream_originalFIR_numpy_fp32'
