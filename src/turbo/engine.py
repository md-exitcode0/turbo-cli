import os, sys, subprocess, shutil
from pathlib import Path

CLI_ROOT = Path(__file__).parent.parent.parent
ENGINE_SRC = CLI_ROOT / "turboquant_plus"
BUILD_DIR = CLI_ROOT / "build"
EXE_PATH = BUILD_DIR / "bin" / "llama-server.exe"

def get_engine():
    if EXE_PATH.exists(): return str(EXE_PATH)
    if not ENGINE_SRC.exists():
        print("Error: Engine source not bundled. Reinstall package."); sys.exit(1)
    
    print("Building bundled engine (first run, please wait)...")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(["cmake", str(ENGINE_SRC), "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"], 
                      cwd=str(BUILD_DIR), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j"], 
                      cwd=str(BUILD_DIR), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Engine built successfully.")
        return str(EXE_PATH)
    except Exception as e:
        print(f"Build failed: {e}\nEnsure Git, CMake, and CUDA are installed."); sys.exit(1)
