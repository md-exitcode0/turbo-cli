import os, sys, json, subprocess, time, urllib.request, signal, atexit, argparse
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

def load_presets():
    if PRESETS.exists():
        try: return json.loads(PRESETS.read_text())
        except: pass
    return {}

def save_presets(p):
    CFG.mkdir(parents=True, exist_ok=True)
    PRESETS.write_text(json.dumps(p, indent=2))

def cmd_launch(name=None):
    p = load_presets()
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
        
        model_gb = os.path.getsize(m) / 1024**3
        kv_gb = (model_gb / 7) * (cs / 32000) * 0.4
        est = model_gb * min(1, n/100) + kv_gb + 0.5
        
        console.print()
        t = Table(title="VRAM Estimate", show_header=False, box=box.SIMPLE, padding=(0, 2))
        t.add_row("Model on GPU", f"{model_gb * min(1, n/100):.1f} GB", style="cyan")
        t.add_row("KV Cache", f"{kv_gb:.1f} GB", style="yellow")
        t.add_row("Overhead", "~0.5 GB", style="dim")
        t.add_row("---", "---")
        t.add_row("[bold]Total[/]", f"[bold]{est:.1f} GB[/]")
        console.print(t)
        if est <= 24: console.print("  [green]Fits in 24GB VRAM[/]")
        else: console.print(f"  [yellow]Exceeds 24GB -- spills to RAM[/]")
        
        console.print()
        if questionary.confirm("Save as preset?", default=False, qmark="").ask():
            nm = questionary.text("Preset name:", default=os.path.basename(m).replace(".gguf",""), qmark="").ask()
            if nm:
                p[nm] = {"model":m, "k":k, "v":v, "ctx":cs, "ngl":n, "port":port, "host":"127.0.0.1"}
                save_presets(p)
                console.print(f"[green]Saved preset '{nm}'[/]")
        cfg = {"model":m, "k":k, "v":v, "ctx":cs, "ngl":n, "port":port, "host":"127.0.0.1"}
    run(cfg)

def cmd_chat(port=None):
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

def cmd_presets():
    p = load_presets()
    if not p: console.print("[dim]No presets saved.[/]"); return
    console.print(Panel("[bold]Saved Presets[/]", border_style="cyan", box=box.ROUNDED))
    t = Table("Name", "Model", "Context", "Created", box=box.SIMPLE, padding=(0, 2))
    for nm, c in sorted(p.items()):
        t.add_row(f"[cyan]{nm}[/]", os.path.basename(c.get("model","?"))[:20], f"{c.get('ctx','?')}", c.get("created","?"))
    console.print(t)
    if questionary.confirm("Launch one?", default=False, qmark="").ask():
        ch = questionary.select("Preset:", choices=sorted(p.keys()), qmark="").ask()
        if ch: cmd_launch(ch)

def cmd_preset_list():
    p = load_presets()
    if not p: console.print("[dim]No presets.[/]"); return
    console.print(Panel("[bold]Presets[/]", border_style="cyan", box=box.ROUNDED))
    t = Table("Name", "Model", "Context", box=box.SIMPLE, padding=(0, 2))
    for nm, c in sorted(p.items()):
        t.add_row(f"[cyan]{nm}[/]", os.path.basename(c.get("model","?"))[:20], f"{c.get('ctx','?')}")
    console.print(t)

def cmd_preset_create(name=None):
    console.print(Panel("[bold]Create Preset[/]", border_style="green", box=box.ROUNDED))
    m = questionary.path("Model path:", only_directories=False, qmark="").ask()
    if not m: return
    k = questionary.select("K Cache:", choices=TYPES, default="q8_0", qmark="").ask()
    v = questionary.select("V Cache:", choices=TYPES, default="turbo3", qmark="").ask()
    c = int(questionary.select("Context:", choices=[str(x) for x in CTX], default="262144", qmark="").ask())
    n = int(questionary.text("GPU Layers:", default="99", qmark="").ask())
    port = int(questionary.text("Port:", default="8080", qmark="").ask())
    nm = name or questionary.text("Name:", default=os.path.basename(m).replace(".gguf",""), qmark="").ask()
    if not nm: return
    p = load_presets(); p[nm] = {"model":m, "k":k, "v":v, "ctx":c, "ngl":n, "port":port, "host":"127.0.0.1", "created":time.strftime("%H:%M")}
    save_presets(p)
    console.print(f"[green]Saved '{nm}'[/]")

def cmd_preset_remove():
    p = load_presets()
    if not p: return
    ch = questionary.checkbox("Remove:", choices=sorted(p.keys()), qmark="").ask()
    if not ch: return
    for nm in ch: del p[nm]
    save_presets(p)
    console.print(f"[green]Removed {len(ch)}[/]")

def cmd_preset_export(file=None):
    p = load_presets()
    out = file or questionary.text("Export to:", default="presets.json", qmark="").ask()
    if out: Path(out).write_text(json.dumps(p, indent=2)); console.print(f"[green]Exported to {out}[/]")

def cmd_preset_import(file=None):
    src = file or questionary.path("Import from:", only_directories=False, qmark="").ask()
    if not src or not os.path.exists(src): return
    try:
        imp = json.loads(Path(src).read_text()); p = load_presets()
        new = sum(1 for n in imp if p.setdefault(n, imp[n]) == imp[n])
        save_presets(p)
        console.print(f"[green]Imported {new} new[/]")
    except Exception as e: console.print(f"[red]Error: {e}[/]")

def main():
    parser = argparse.ArgumentParser(
        prog="turbo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Turbo CLI -- LLM Server Launcher",
        epilog="""
Commands:
  turbo launch              Start server interactively or with preset
  turbo launch <name>       Launch saved preset
  turbo chat                Chat with running server
  turbo presets             List presets interactively
  turbo preset create       Create new preset
  turbo preset list         List all presets
  turbo preset remove       Remove presets
  turbo preset export       Export to JSON
  turbo preset import       Import from JSON

Run 'turbo <command> --help' for more information on a command.
""")
    sub = parser.add_subparsers(dest="cmd", help="Command to run")
    sub.add_parser("launch", help="Start server interactively or with preset").add_argument("preset", nargs="?", help="Preset name")
    sub.add_parser("chat", help="Chat with running server")
    sub.add_parser("presets", help="List presets interactively")
    pp = sub.add_parser("preset", help="Manage presets")
    pps = pp.add_subparsers(dest="preset_cmd")
    pps.add_parser("list", help="List presets")
    pps.add_parser("remove", help="Remove presets")
    pps.add_parser("create", help="Create preset").add_argument("name", nargs="?")
    pps.add_parser("export", help="Export to JSON").add_argument("file", nargs="?")
    pps.add_parser("import", help="Import from JSON").add_argument("file", nargs="?")
    
    a = parser.parse_args()
    if a.cmd == "chat": cmd_chat()
    elif a.cmd == "launch": cmd_launch(a.preset)
    elif a.cmd == "presets": cmd_presets()
    elif a.cmd == "preset":
        cmd = a.preset_cmd
        if cmd == "list": cmd_preset_list()
        elif cmd == "create": cmd_preset_create(a.name)
        elif cmd == "remove": cmd_preset_remove()
        elif cmd == "export": cmd_preset_export(a.file)
        elif cmd == "import": cmd_preset_import(a.file)
        else: pp.print_help()
    else: parser.print_help()

if __name__ == "__main__": main()
