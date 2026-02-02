#!/bin/bash
for r in $(seq 0.5 0.35 2.6)
do
    python run_LiH.py $r
done