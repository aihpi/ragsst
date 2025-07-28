#!/bin/bash
tmux new -s ollama -d
tmux splitw -v -t ollama
type nvtop > /dev/null 2>&1 && tmux send-keys -t ollama:0.0 'nvtop' Enter || tmux send-keys -t ollama:0.0 'watch -n 1 nvidia-smi' Enter
tmux send-keys -t ollama:0.1 'journalctl -xefu ollama.service' Enter
tmux attach -t ollama
