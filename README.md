# Turbo CLI

**One-click LLM Server with TurboQuant Engine**

## Setup

```batch
git clone https://github.com/md-exitcode0/turbo-cli.git
cd turbo-cli
setup.bat
```

## Commands

### Launch Server

```bash
# Interactive mode (auto-calculates VRAM)
turbo launch

# Load saved preset
turbo launch mystar

# Help
turbo --help
turbo launch --help
```

**Options:**
- **Model path** - Path to `.gguf` file
- **K Cache** - `q8_0`, `q4_0`, `turbo2`, `turbo3`, `turbo4`
- **V Cache** - `turbo3` recommended
- **Context** - 8192 to 262144
- **GPU Layers** - 99 (all GPU), or lower to save VRAM

---

### Presets

```bash
# List all presets
turbo presets

# Create preset
turbo preset create mymodel

# Remove preset
turbo preset remove
```

### VRAM Settings

```bash
# Show current VRAM setting
turbo vram

# Set VRAM (e.g., if you have 12GB instead of 24GB)
turbo vram set 12
```

---

## Features

- Pre-built engine bundled (140MB)
- TurboQuant support (turbo2/3/4)
- VRAM estimation before launch
- Presets for quick launch

---

## Requirements

- **Python**: 3.8+
- **OS**: Windows 10/11
- **GPU**: NVIDIA (optional, CPU fallback available)
- **Disk**: ~145MB for engine + models
- **Internet**: Only for first run (downloads engine)

---

## Architecture

```
turbo-cli/
├── setup.bat           # One-click installer
├── src/turbo/
│   ├── cli.py          # Main CLI
│   ├── engine.py       # Engine unpacker
│   ├── bin/            # Pre-built llama-server.exe + DLLs
│   └── data/
│       └── engine.zip  # Bundled engine (140MB)
└── turboquant_plus/    # Engine source (optional)
```

---

## License

Apache 2.0

---

[md-exitcode0](https://github.com/md-exitcode0)
