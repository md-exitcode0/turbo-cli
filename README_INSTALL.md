# Turbo CLI

Fast LLM server launcher with TurboQuant engine.

## Installation

### Option 1: Quick Install (Windows)

```batch
install.bat
```

### Option 2: Manual Install

```bash
# Ensure you have Python 3.8+
python -m venv venv
venv\Scripts\activate
pip install -e .
```

## Usage

### Start Server

```bash
# Interactive mode
turbo launch

# Load preset
turbo launch mymodel
```

### Chat

```bash
turbo chat
```

### Presets

```bash
# List presets
turbo presets

# Create preset
turbo preset create
```

## Features

- **One-click install**: Bundled engine, no compilation needed
- **VRAM estimation**: Auto-calculate memory requirements
- **Preset management**: Save and load configurations
- **TurboQuant support**: Native turbo2/turbo3/turbo4 quantization

## Requirements

- Python 3.8+
- Windows 10/11
- NVIDIA GPU (optional, for CUDA acceleration)

## Troubleshooting

If the server won't start:
1. Check that `llama-server.exe` exists in `C:\Users\<user>\.turbo\`
2. Verify your model file is in GGUF format
3. Ensure enough VRAM/RAM is available
