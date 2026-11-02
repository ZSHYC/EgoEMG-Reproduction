"""Run the complete frozen HOT3D intake with explicit read-only cache paths."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from egoemg_repro.hot3d_cache_intake import main

if __name__=='__main__':main(ROOT)
