# Turbo CLI Setup

## One-Click Install

```batch
setup.bat
```

That's it. This bundles the engine and installs everything.

## Development

### Bundle Engine
```bash
python package_engine.py
```

### Install
```bash
pip install -e .
```

### Test
```bash
turbo --help
turbo launch
```

## Files

- `setup.bat` - One-click installer (bundles engine + installs)
- `package_engine.py` - Bundles llama-server.exe into engine.zip
- `turboquant_plus/` - Engine source code
- `src/turbo/data/engine.zip` - Bundled engine (3.36 MB)

## End User Flow

1. Download source from GitHub
2. Run `setup.bat`
3. Run `turbo launch`
