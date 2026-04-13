#!/usr/bin/env python3
import os
import sys
import shutil
import zipfile
from pathlib import Path


def main():
    script_dir = Path(__file__).parent.resolve()
    turboquant_plus = script_dir / "turboquant_plus"
    data_dir = script_dir / "src" / "turbo" / "data"
    engine_zip = data_dir / "engine.zip"

    # Check for pre-built exe in parent bin or build
    prebuilt_exe = None
    parent_bin = script_dir / "src" / "turbo" / "bin" / "llama-server.exe"
    build_bin = script_dir / "build" / "bin" / "llama-server.exe"

    if parent_bin.exists():
        prebuilt_exe = parent_bin
        print(f"Using pre-built: {parent_bin}")
    elif build_bin.exists():
        prebuilt_exe = build_bin
        print(f"Using pre-built: {build_bin}")

    data_dir.mkdir(parents=True, exist_ok=True)

    # If engine.zip already exists, just copy it
    if engine_zip.exists():
        print(f"Engine already bundled: {engine_zip.name}")
        return

    if not prebuilt_exe:
        print("Error: No pre-built exe found")
        sys.exit(1)

    print(f"Packaging engine from: {prebuilt_exe.parent}")
    print(f"Creating: {engine_zip}")

    with zipfile.ZipFile(engine_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add pre-built exe first
        zipf.write(prebuilt_exe, "llama-server.exe")
        print("  Added: llama-server.exe")

        # Add DLLs from same directory
        bin_dir = prebuilt_exe.parent
        for dll in bin_dir.glob("*.dll"):
            zipf.write(dll, dll.name)
            print(f"  Added: {dll.name}")

    size_mb = engine_zip.stat().st_size / (1024 * 1024)
    print(f"Done! Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
