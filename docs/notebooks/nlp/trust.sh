#!/usr/bin/env bash

# List all the jupyter notebooks, and
# mark each as trusted.

for notebook in $(ls *.ipynb); do echo Marking as trusted: $notebook; jupyter trust $notebook; done
