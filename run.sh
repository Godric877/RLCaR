#!/bin/bash

python3 main.py -policy LRU -cs 20 -rl random_eviction &
python3 main.py -policy LRU -cs 20 -rl random_eviction