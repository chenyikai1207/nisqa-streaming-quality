"""Qualified B2 uses its registered TF32-off setting without altering student training flags."""
from contextlib import contextmanager
import torch
@contextmanager
def inference_precision():
    matmul=torch.backends.cuda.matmul.allow_tf32;cudnn=torch.backends.cudnn.allow_tf32;dtype=torch.get_default_dtype()
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul;torch.backends.cudnn.allow_tf32=cudnn;torch.set_default_dtype(dtype)
b2_precision=inference_precision
