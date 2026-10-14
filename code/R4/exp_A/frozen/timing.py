"""Stationary model, independently callable raw-audio cost stages."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
from runtime import Predictor
_current=None
def make_predictor(model,device='cpu'):
    return Predictor(ROOT/model,device)
def configure(model,device='cpu'):
    global _current
    _current=make_predictor(model,device);return _current
def load_audio(path):return _current.load_audio(path)
def extract_features(wav,sr):return _current.extract_features(wav,sr)
def infer(features):return _current.infer(features)
