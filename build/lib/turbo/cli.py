import os, sys, json, struct, subprocess, time, urllib.request, signal, atexit, argparse
from pathlib import Path
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich import box

console = Console()
CFG = Path.home() / ".turbo"
PRESETS = CFG / "presets.json"
LOG = CFG / "server.log"
TYPES = ["f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4"]
CTX = [8192, 16384, 32768, 65536, 131072, 200000, 262144]
proc = None

def cleanup():
    global proc
    if proc and proc.poll() is None:
        try: proc.terminate(); proc.wait(timeout=3)
        except: proc.kill()
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))

def run(cfg):
    global proc
    from .engine import get_engine
    exe = get_engine()
    cmd = [exe, "-m", cfg["model"], "--ctx-size", str(cfg["ctx"]),
           "--port", str(cfg["port"]), "--host", cfg["host"],
           "-ctk", cfg["k"], "-ctv", cfg["v"], "-ngl", str(cfg["ngl"]), "--flash-attn", "on"]
    if cfg.get("ncmoe"): cmd += ["-ncmoe", str(cfg["ncmoe"])]
    console.print(Panel("[bold]Starting Server...[/]", border_style="green", box=box.ROUNDED))
    t = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    t.add_row("Model", os.path.basename(cfg["model"]), style="cyan")
    t.add_row("Context", f"{cfg['ctx']:,}")
    t.add_row("KV Cache", f"K={cfg['k']}  V={cfg['v']}")
    t.add_row("GPU Layers", str(cfg["ngl"]) + (f"  MoE->CPU: {cfg['ncmoe']}" if cfg.get("ncmoe") else ""))
    t.add_row("Address", f"http://{cfg['host']}:{cfg['port']}", style="yellow")
    console.print(t)
    proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=str(Path(exe).parent))
    console.print(f"[dim]PID: {proc.pid}  Log: {LOG}[/]")
    try:
        with Live(Spinner("dots", text="Loading model...", style="cyan"), console=console, transient=False) as live:
            for i in range(300):
                time.sleep(1)
                if proc.poll() is not None:
                    console.print("\n[bold red]Server exited unexpectedly.[/]"); return
                try:
                    with urllib.request.urlopen(f"http://{cfg['host']}:{cfg['port']}/health", timeout=2) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            live.stop()
                            console.print("\n[bold green]Model Loaded -- Server Ready![/]")
                            console.print(f"http://{cfg['host']}:{cfg['port']}/v1")
                            console.print("[dim]Press Ctrl+C to stop[/]")
                            break
                except: pass
                live.update(Spinner("dots", text=f"Loading... ({i}s)", style="cyan"))
        while proc.poll() is None: time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/]")
        proc.terminate()
        try: proc.wait(timeout=3)
        except: proc.kill()
        console.print("[green]Done.[/]")

def launch(name=None):
    p = json.loads(PRESETS.read_text()) if PRESETS.exists() else {}
    if name:
        if name not in p: console.print(f"[red]Preset '{name}' not found.[/]"); return
        console.print(Panel(f"[bold]Loading Preset: {name}[/]", border_style="cyan", box=box.ROUNDED))
        cfg = p[name]
    else:
        console.print(Panel("[bold]Turbo CLI -- LLM Server Launcher[/]", border_style="blue", box=box.ROUNDED))
        m = questionary.path("Model path (.gguf):", only_directories=False, qmark="").ask()
        if not m or not os.path.exists(m): console.print("[red]Cancelled or file not found.[/]"); return
        console.print(f"[dim]-> {os.path.basename(m)} ({os.path.getsize(m)/1024**3:.1f} GB)[/]")
        console.print()
        k = questionary.select("K Cache:", choices=TYPES, default="q8_0", qmark="").ask()
        v = questionary.select("V Cache:", choices=TYPES, default="turbo3", qmark="").ask()
        console.print()
        c = questionary.select("Context Length:", choices=[str(x) for x in CTX]+["custom"], default="262144", qmark="").ask()
        cs = int(c) if c != "custom" else int(questionary.text("Custom context:", default="262144", qmark="").ask() or "262144")
        n = int(questionary.text("GPU Layers (-ngl):", default="99", qmark="").ask() or "99")
        port = int(questionary.text("Server Port:", default="8080", qmark="").ask() or "8080")
        console.print()
        if questionary.confirm("Save as preset?", default=False, qmark="").ask():
            nm = questionary.text("Preset name:", default=os.path.basename(m).replace(".gguf",""), qmark="").ask()
            if nm:
                p[nm] = {"model":m, "k":k, "v":v, "ctx":cs, "ngl":n, "port":port, "host":"127.0.0.1"}
                CFG.mkdir(parents=True, exist_ok=True)
                PRESETS.write_text(json.dumps(p, indent=2))
                console.print(f"[green]Saved preset '{nm}'[/]")
        cfg = {"model":m, "k":k, "v":v, "ctx":cs, "ngl":n, "port":port, "host":"127.0.0.1"}
    run(cfg)

def chat(port=None):
    if port is None: port = int(questionary.text("Server Port (default 8080):", default="8080", qmark="").ask())
    console.print(Panel("[bold]Turbo Chat[/]", border_style="cyan", box=box.ROUNDED))
    console.print(f"[dim]Connected to http://127.0.0.1:{port}  (type /exit to quit)[/]\n")
    while True:
        msg = questionary.text("", qmark="> ").ask()
        if not msg or msg.strip() == "/exit": break
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                data=json.dumps({"prompt":msg, "max_tokens":512}).encode(), headers={"Content-Type":"application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.loads(r.read())["choices"][0]["text"].strip()
                console.print(f"[bold][AI][/]{txt}\n")
        except Exception as e: console.print(f"[red]Error: {e}[/]")

def main():
    p = argparse.ArgumentParser(prog="turbo", description="Turbo CLI -- LLM Server Launcher")
    s = p.add_subparsers(dest="cmd")
    s.add_parser("launch", help="Start server interactively or with preset").add_argument("preset", nargs="?", help="Preset name")
    s.add_parser("chat", help="Chat with running server")
    a = p.parse_args()
    if a.cmd == "chat": chat()
    else: launch(a.preset)

if __name__ == "__main__": main()
