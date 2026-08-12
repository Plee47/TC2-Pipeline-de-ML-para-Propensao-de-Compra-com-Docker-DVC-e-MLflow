#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ecommerce_buy_predictor.pipeline.train_stage import train_stage

if __name__ == "__main__":
    train_stage()
