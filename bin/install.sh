#!/bin/bash
echo 'Installing/Updating ollama...' && \
curl -fsSL https://ollama.ai/install.sh | sh && \
echo 'Creating virtual environment...' && \
python3 -m venv .myvenv && \
echo 'Activating virtual environment...' && \
source .myvenv/bin/activate && \
echo 'Installing requirements...' && \
pip3 install -r requirements.txt
echo 'Pulling ollama model...' && \
ollama pull llama3.1 || echo 'choose and download a model with $ ollama pull <your_model_of_choice>'
