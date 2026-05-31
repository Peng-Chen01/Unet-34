#!/bin/bash

echo "=============================="
echo "Installing dependencies..."
echo "=============================="

pip install --upgrade pip

pip install -r requirements.txt

echo "=============================="
echo "Checking GPU..."
echo "=============================="

nvidia-smi

echo "=============================="
echo "Starting training..."
echo "=============================="

python main.py

