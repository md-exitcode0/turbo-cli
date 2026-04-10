# Turbo CLI

Self-contained LLM server launcher. Ships with full TurboQuant engine source.

## Prerequisites
- **Python 3.10+**
- **Git**
- **CMake**
- **NVIDIA GPU + CUDA**

## Install
```cmd
git clone https://github.com/md-exitcode0/turbo-cli.git
cd turbo-cli
pip install -e .
```

## First Run
```cmd
turbo launch
```
*The engine compiles automatically from bundled source. Takes 2-5 minutes. Happens only once.*

## Usage
```cmd
turbo launch          # Interactive setup
turbo launch <name>   # Launch preset
turbo chat            # Chat with server
turbo presets         # Manage presets
```

## Troubleshooting
- **Build fails?** Ensure Git, CMake, and CUDA Toolkit are installed and in PATH.
- **CUDA error?** Update GPU drivers and verify CUDA installation.

---
*Created by **md-exitcode0***
