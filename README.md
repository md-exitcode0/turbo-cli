# Turbo CLI

I built this because I was tired of dealing with llama.cpp builds every time I wanted to test TurboQuant KV cache compression. It's not fancy but it works.

## Quick Start

```batch
setup.bat
turbo launch
```

That's it. No CMake, no Visual Studio, no compiling.

## What It Does

- Bundles a pre-compiled llama-server.exe with TurboQuant support
- `turbo launch` starts an interactive setup wizard
- Shows VRAM estimates before loading so you don't OOM on your 8GB card
- Save presets for models you use often

## Commands

```batch
turbo launch              # Interactive server setup
turbo launch mypreset    # Load saved preset
turbo presets            # List all presets
turbo vram               # Show current VRAM limit
turbo vram set 12        # Set VRAM limit to 12GB
turbo update             # Pull latest release from GitHub
turbo -v                 # Show version
```

## Example: Running a 9B Model on 8GB VRAM

```
$ turbo launch
Model path: C:\models\Qwen2.5-9B-Q4_K_M.gguf
Context: 8192
KV Cache K: q8_0
KV Cache V: turbo3
GPU Layers: 99
Port: 8080

VRAM Estimate:
  Model on GPU:  5.4 GB
  KV Cache:      0.8 GB
  Overhead:      ~0.5 GB
  Total:         6.7 GB

Fits in 24GB VRAM ✓
```

## Requirements

- Windows 10/11
- Python 3.8+
- NVIDIA GPU with CUDA (CPU fallback works but slow)
- GGUF model files

## How It Works

1. `setup.bat` bundles the engine and installs the pip package
2. First run unpacks `llama-server.exe` to `%USERPROFILE%\.turbo\`
3. `turbo launch` starts the server with your config

## Technical Details

- Engine: TheTom's llama.cpp fork with TurboQuant KV cache
- Build: CMake + CUDA, Release mode
- Binaries: llama-server.exe + 6 DLLs (ggml-cuda.dll, etc.)

## My Setup

- GPU: RTX 3060 Ti 8GB
- CPU: AMD Ryzen 5 5600X
- RAM: 32GB
- OS: Windows 11

## Source

https://github.com/md-exitcode0/turbo-cli

License: Apache 2.0
