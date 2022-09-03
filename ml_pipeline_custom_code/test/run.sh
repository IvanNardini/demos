#!/bin/bash

cd ./object-oriented-pipe

for script in $*; do
    if [ $script == 'test' ]; then
        python3 test.py
    fi
done