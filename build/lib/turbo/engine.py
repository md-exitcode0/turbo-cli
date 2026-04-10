import os, sys, json, urllib.request, zipfile
from pathlib import Path
import time

BIN = Path.home() / ".turbo" / "bin"
EXE = BIN / "llama-server.exe"
API = "https://api.github.com/repos/TheTom/turboquant_plus/releases/latest"

def get_engine():
    if EXE.exists(): return str(EXE)
    BIN.mkdir(parents=True, exist_ok=True)
    print("Downloading engine... (this may take a minute)")
    print(f"Fetching from: {API}")
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(API, headers={"User-Agent": "turbo"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
                assets = data.get("assets", [])
            
            # Try to find an executable or zip
            dl = None
            for a in assets:
                name = a["name"].lower()
                if name.endswith(".exe"):
                    dl = a
                    break
            if not dl:
                for a in assets:
                    if a["name"].lower().endswith(".zip"):
                        dl = a
                        break
            
            if not dl:
                print("Available assets:", [a["name"] for a in assets])
                raise SystemExit("No suitable engine binary found in release.")
            
            tmp = BIN / "tmp.bin"
            print(f"Downloading {dl['name']}...")
            urllib.request.urlretrieve(dl["browser_download_url"], tmp)
            
            if dl["name"].lower().endswith(".zip"):
                with zipfile.ZipFile(tmp) as z:
                    for n in z.namelist():
                        if "llama-server" in n.lower() and (n.endswith(".exe") or n.endswith(".dll") or n.endswith(".bin")):
                            EXE.write_bytes(z.read(n))
                            break
                tmp.unlink()
            else:
                tmp.rename(EXE)
            
            if not EXE.exists():
                raise SystemExit("Extraction failed. Please download manually.")
                
            print("Engine installed successfully.")
            return str(EXE)
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
            
    print("\nManual installation required:")
    print(f"1. Go to: https://github.com/TheTom/turboquant_plus/releases")
    print("2. Download the latest 'llama-server.exe' (or zip containing it)")
    print(f"3. Place it in: {BIN}")
    sys.exit(1)
