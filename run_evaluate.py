#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ecommerce_buy_predictor.pipeline.evaluate_stage import evaluate_stage

if __name__ == "__main__":
    evaluate_stage()
