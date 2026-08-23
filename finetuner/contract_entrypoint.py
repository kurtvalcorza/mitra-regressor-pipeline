#!/usr/bin/env python3
from __future__ import annotations
import sys,contract_hardening,train
contract_hardening.install_finetuner(train,"tabular_regression")
if __name__=="__main__":sys.exit(train.main())
