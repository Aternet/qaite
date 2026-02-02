#!/bin/bash
for r in $(seq 0.5 0.15 2.6)
do
    python run_H4.py $r
done