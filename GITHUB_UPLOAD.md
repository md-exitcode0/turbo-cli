# GitHub Upload Guide

## Files to Upload

Create a new repository on GitHub, then upload these files:

### Root Files
- `pyproject.toml` - Package configuration
- `setup.py` - Installation script  
- `install.bat` - One-click Windows installer
- `package_engine.py` - Engine packaging script
- `setup_dist.py` - Distribution builder
- `README.md` - Copy `README_GITHUB.md` to this name
- `AGENTS.md` - Development documentation
- `.github/workflows/build.yml` - CI/CD pipeline

### Source Files  
- `src/turbo/__init__.py`
- `src/turbo/cli.py`
- `src/turbo/engine.py`
- `src/turbo/data/engine.zip` (**Important: This is 3.36 MB - the pre-built engine**)

## Distribution

### Build Release Package

```bash
python setup_dist.py
```

This creates `dist/turbo-cli-1.0.0-windows.zip` (3.35 MB)

### Upload to GitHub Releases

1. Go to your repo → Releases → "Draft new release"
2. Tag: `v1.0.0`
3. Title: "Turbo CLI v1.0.0"
4. Attach `dist/turbo-cli-1.0.0-windows.zip`
5. Publish

## Installation (End User)

1. Download `turbo-cli-1.0.0-windows.zip` from Releases
2. Extract to `C:\turbo_test\` or any folder
3. Double-click `install.bat`
4. Run `turbo launch`

## Key Features

- **No compilation needed** - Pre-built llama-server.exe bundled in engine.zip
- **No CMake/VS required** - Just Python 3.8+
- **First run unpacks engine** to `C:\Users\<you>\.turbo\llama-server.exe`
- **Auto VRAM estimation** before loading models

## Testing Your Upload

After pushing to GitHub:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/turbo-cli.git
cd turbo-cli

# Install
pip install -e .

# Test
turbo --help
```

## Repository Structure

```
turbo-cli/
├── .github/
│   └── workflows/
│       └── build.yml
├── src/
│   └── turbo/
│       ├── __init__.py
│       ├── cli.py
│       ├── engine.py
│       └── data/
│           └── engine.zip (3.36 MB)
├── install.bat
├── package_engine.py
├── pyproject.toml
├── README.md
├── setup.py
└── setup_dist.py
```

**Total size: ~4 MB** (includes pre-built engine)

## Notes

- The `engine.zip` contains the pre-compiled `llama-server.exe` - users don't need to compile from source
- If you update the engine, run `python package_engine.py` to recreate the zip
- The `.github/workflows/` folder enables automatic builds on tag pushes
