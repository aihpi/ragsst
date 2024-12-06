#!/bin/bash
echo 'Installing/Updating ollama...' && \
curl -fsSL https://ollama.ai/install.sh | sh && \
read -p "Install the Local RAG tool in a virtual environment? (Type 'y' to agree, or leave blank to skip. Press Enter):" VENV_CHOICE
if [[ $VENV_CHOICE == [yY] ]]; then
	echo 'Creating virtual environment...' && \
	python3 -m venv .myvenv && \
	echo 'Activating virtual environment...' && \
	source .myvenv/bin/activate
fi
echo 'Installing requirements...' && \
pip3 install -r requirements.txt
echo 'Pulling ollama model...' && \
ollama pull llama3.2 || echo 'choose and download a model with $ ollama pull <your_model_of_choice>'
