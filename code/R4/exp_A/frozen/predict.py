from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code'))
from runtime import Predictor
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--csv',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    Predictor(ROOT/a.model,a.device).predict_csv(a.csv,a.out)
if __name__=='__main__':main()
