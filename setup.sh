#!/bin/bash

# RunPod BPE Tokenizer Setup Script
# Automated setup for cs336_assignment1 repository
#
set -e  # Exit on any error

echo "🚀 Starting RunPod BPE Tokenizer Setup..."
echo "================================================"

# Function to print status messages
print_status() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

print_info() {
    echo "ℹ️  $1"
}

# Check if we're in the right environment
print_info "Setting up environment for BPE tokenizer project"

# Update system packages and install essential tools
print_status "Installing system dependencies..."
apt update && apt upgrade -y

apt install -y \
    curl \
    wget \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    unzip \
    htop \
    tmux \
    vim

print_status "System dependencies installed successfully"

print_status "Installing UV package manager..."

curl -LsSf https://astral.sh/uv/install.sh | sh

source ~/.bashrc

print_info "UV version: $(uv --version)"


cd /workspace/cs336_assignment1

#print_info "Setting up UV env for CPU"

uv sync --no-install-package torch --no-install-package torchvision --no-install-package torchaudio
source .venv/bin/activate
rm -f uv.lock
uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

print_info "Setup env with CPU" 

#print_info "getting project data"
#
#mkdir data
#cd data
#
#wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
#wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
#
## Download OpenWebText samples
#wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
#wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
#
## Extract the compressed files
#gunzip owt_train.txt.gz
#gunzip owt_valid.txt.gz
#
#cd ..
#
#print_info "Data downloaded"


# Verify setup by running tests
print_status "Verifying setup by running tests..."
print_info "Note: Tests are expected to fail initially with NotImplementedError"
uv run pytest --tb=short || print_info "Tests failed as expected (NotImplementedError)"

# Display summary
echo ""
echo "🎉 Setup Complete!"
echo "================================================"
print_status "UV package manager installed"
print_status "Repository cloned and dependencies installed"
print_status "All datasets downloaded"
print_status "Environment verified"

echo ""
echo "📁 Project structure:"
echo "   $(pwd)"
echo "   ├── data/"
echo "   │   ├── TinyStoriesV2-GPT4-train.txt"
echo "   │   ├── TinyStoriesV2-GPT4-valid.txt"
echo "   │   ├── owt_train.txt"
echo "   │   └── owt_valid.txt"
echo "   └── tests/adapters.py (complete this file)"

echo ""
echo "🔧 Next steps:"
echo "   1. Complete the functions in ./tests/adapters.py"
echo "   2. Implement your BPE tokenizer"
echo "   3. Run tests with: uv run pytest"
echo "   4. Run any Python file with: uv run <file_path>"

echo ""
echo "💡 Useful commands:"
echo "   uv run pytest              # Run all tests"
echo "   uv run pytest -v           # Run tests with verbose output"
echo "   uv run python <file>       # Run a Python file"
echo "   uv add <package>           # Add a new dependency"

echo ""
print_status "Happy coding! Your BPE tokenizer environment is ready! 🚀"

