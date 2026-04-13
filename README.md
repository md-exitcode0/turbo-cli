# Turbo CLI

**One-click LLM Server with TurboQuant Engine**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](https://windows.microsoft.com/)

## Setup (30 seconds)

```batch
# Download and extract source code
setup.bat

# Start server
turbo launch
```

**That's it!** No CMake, no Visual Studio, no compilation needed.

## Usage

```bash
# Launch server interactively (auto-calculates VRAM)
turbo launch

# Load saved preset
turbo launch mystar

# Chat with model
turbo chat

# List saved presets
turbo presets
```

## Features

- ⚡ **One-click install** - Pre-built engine bundled, no compilation
- 🚀 **TurboQuant** - Native turbo2/turbo3/turbo4 quantization support
- 📊 **VRAM estimation** - Shows memory requirements before loading model
- 💾 **Presets** - Save configurations for quick launch next time

## How It Works

1. `setup.bat` bundles the engine and installs the CLI
2. First use unpacks `llama-server.exe` to `C:\Users\<you>\.turbo\`
3. `turbo launch` starts the server with your GGUF model

## Requirements

- Python 3.8+ (Windows)
- GGUF model files
- NVIDIA GPU (optional, CPU fallback available)

## License

MIT
