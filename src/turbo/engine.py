import os, sys, subprocess, shutil, zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
USER_DATA = Path.home() / ".turbo"
EXE_PATH = USER_DATA / "llama-server.exe"
LOG_FILE = USER_DATA / "server.log"


def unpack_engine():
    if EXE_PATH.exists():
        return str(EXE_PATH)

    engine_zip = DATA_DIR / "engine.zip"
    if not engine_zip.exists():
        return None

    print("Unpacking engine...")
    USER_DATA.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(engine_zip, "r") as zip_ref:
            zip_ref.extractall(USER_DATA)
            print("Engine unpacked.")
            return str(EXE_PATH)
    except Exception as e:
        print(f"Error unpacking engine: {e}")
        return None


def add_dll_directory():
    exe_dir = str(Path(EXE_PATH).parent)
    if sys.platform == "win32" and os.path.exists(exe_dir):
        os.add_dll_directory(exe_dir)


def get_engine():
    exe = unpack_engine()
    if exe and os.path.exists(exe):
        return exe

    print("Error: Could not find or extract engine binary.")
    print("Please ensure engine.zip contains llama-server.exe or has been pre-built.")
    sys.exit(1)

    print("Building bundled engine (first run, please wait)...")
    print("Source:", str(engine_src))
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("Running CMake configure...")
        cmake_config = subprocess.run(
            ["cmake", str(engine_src), "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=str(BUILD_DIR),
            capture_output=True,
            text=True,
        )
        if cmake_config.returncode != 0:
            print("\nCMAKE CONFIGURE FAILED:")
            print(cmake_config.stderr[-500:])
            print("\nEnsure CMake and Visual Studio Build Tools are installed.")
            sys.exit(1)

        print("Running CMake build...")
        cmake_build = subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=str(BUILD_DIR),
            capture_output=True,
            text=True,
        )
        if cmake_build.returncode != 0:
            print("\nBUILD FAILED:")
            print(cmake_build.stderr[-500:])
            sys.exit(1)

        if not EXE_PATH.exists():
            print("\nBinary not found after build.")
            print("Check CMake logs for details.")
            sys.exit(1)

        print("Engine built successfully.")
        return str(EXE_PATH)
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)
