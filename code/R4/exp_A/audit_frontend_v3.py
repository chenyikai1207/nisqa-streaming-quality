from pathlib import Path
ROOT=Path(__file__).resolve().parent
STAGE=ROOT.parent
DIMS=['mos','noi','col','dis','loud']
def note(message):
    print(message)
