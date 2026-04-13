# Turbo CLI

**Fast LLM Server Launcher with TurboQuant Engine**

[![Build Status](https://github.com/yourusername/turbo-cli/actions/workflows/build.yml/badge.svg)](https://github.com/yourusername/turbo-cli/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](https://windows.microsoft.com/)

## Features

- ⚡ **One-Click Install** - No compilation, no CMake, no Visual Studio needed
- 🚀 **TurboQuant Engine** - Native support for turbo2/turbo3/turbo4 quantization
- 📊 **VRAM Estimation** - Auto-calculate memory requirements before launch
- 💾 **Preset Management** - Save and load server configurations
- 🎯 **Interactive CLI** - Rich terminal interface with real-time status

## Quick Start

### Windows (One-Click)

```batch
# Download and extract turbo-cli-1.0.0-windows.zip
install.bat

# Start server
turbo launch
```

### From Source

```bash
pip install -e .
turbo launch
```

## Usage

### Launch Server

```bash
# Interactive mode with VRAM estimation
turbo launch

# Load saved preset
turbo launch mystar
```

### Chat Interface

```bash
turbo chat
```

### Presets

```bash
# List all presets
turbo presets

# Create new preset
turbo preset create mymodel

# Remove preset
turbo preset remove
```

## Architecture

```
turbo-cli/
├── src/turbo/
│   ├── __init__.py
│   ├── cli.py          # Main CLI interface
│   ├── engine.py       # Engine unpacker and builder
│   └── data/
│       └── engine.zip  # Bundled llama-server.exe (3.36 MB)
├── install.bat         # Windows installer
└── README.md
```

The engine is bundled as a pre-compiled binary, no compilation required.

## Configuration

Presets are stored in `~/.turbo/presets.json` with the following structure:

```json
{
  "mymodel": {
    "model": "C:/models/mistral-7b-turbo4.gguf",
    "k": "q8_0",
    "v": "turbo3",
    "ctx": 262144,
    "ngl": 99,
    "port": 8080,
    "host": "127.0.0.1"
  }
}
```

## Requirements

- **Python**: 3.8 or higher
- **OS**: Windows 10/11
- **GPU**: NVIDIA (optional, CPU fallback available)

## License

MIT License - See LICENSE file for details.

## Credits

Built with TurboQuant engine for ultra-fast LLM inference.
