import os, sys, json, subprocess, time, urllib.request, signal, atexit, argparse, zipfile, tempfile, shutil
from pathlib import Path
import questionary

__version__ = "1.1.5"
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich import box

console = Console()
CFG = Path.home() / ".turbo"
PRESETS = CFG / "presets.json"
CONFIG = CFG / "config.json"
TYPES = ["f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4"]
CTX = [8192, 16384, 32768, 65536, 131072, 200000, 262144]
proc = None


def load_config():
    if CONFIG.exists():
        try:
            return json.loads(CONFIG.read_text())
        except:
            pass
    return {"vram": 8}


def save_config(c):
    CFG.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(c, indent=2))


def get_vram():
    cfg = load_config()
    return cfg.get("vram", 8)


def set_vram(gb):
    cfg = load_config()
    cfg["vram"] = gb
    save_config(cfg)


def cleanup():
    global proc
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except:
            proc.kill()


atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))


def run(cfg):
    global proc
    from .engine import get_engine, add_dll_directory, LOG_FILE

    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
            else:
                proc.wait()
        except:
            pass
        proc = None
        time.sleep(0.5)

    exe = get_engine()
    add_dll_directory()
    cmd = [
        exe,
        "-m",
        cfg["model"],
        "--ctx-size",
        str(cfg["ctx"]),
        "--port",
        str(cfg["port"]),
        "--host",
        cfg["host"],
        "-ctk",
        cfg["k"],
        "-ctv",
        cfg["v"],
        "-ngl",
        str(cfg["ngl"]),
        "--flash-attn",
        "on",
        "--log-file",
        str(LOG_FILE),
    ]
    # Add optional flags
    if cfg.get("ts"):
        cmd += ["-ts", str(cfg["ts"])]
    if cfg.get("parallel") and int(cfg["parallel"]) > 1:
        cmd += ["--parallel", str(cfg["parallel"])]
    if cfg.get("no_mmap"):
        cmd += ["--no-mmap"]
    if cfg.get("reasoning"):
        cmd += ["--reasoning", "on"]
    else:
        cmd += ["--reasoning", "off"]
    if cfg.get("mmproj"):
        cmd += ["--mmproj", str(cfg["mmproj"])]
    if cfg.get("ncmoe"):
        cmd += ["-ncmoe", str(cfg["ncmoe"])]
    console.print(
        Panel("[bold]Starting Server...[/]", border_style="green", box=box.ROUNDED)
    )
    t = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    t.add_row("Model", os.path.basename(cfg["model"]), style="cyan")
    t.add_row("Context", f"{cfg['ctx']:,}")
    t.add_row("KV Cache", f"K={cfg['k']}  V={cfg['v']}")
    t.add_row(
        "GPU Layers",
        str(cfg["ngl"]) + (f"  MoE->CPU: {cfg['ncmoe']}" if cfg.get("ncmoe") else ""),
    )
    if cfg.get("ts"):
        t.add_row("Tensor Split", str(cfg["ts"]))
    if cfg.get("parallel") and int(cfg["parallel"]) > 1:
        t.add_row("Parallel Slots", str(cfg["parallel"]))
    if cfg.get("mmproj"):
        t.add_row("MMProj", os.path.basename(cfg["mmproj"]))
    t.add_row("Address", f"http://{cfg['host']}:{cfg['port']}", style="yellow")
    console.print(t)

    CFG.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        cwd=str(Path(exe).parent),
    )
    console.print(f"[dim]PID: {proc.pid}  Log: {LOG_FILE}[/]")
    try:
        with Live(
            Spinner("dots", text="Loading model...", style="cyan"),
            console=console,
            transient=False,
        ) as live:
            for i in range(300):
                time.sleep(1)
                if proc.poll() is not None:
                    if LOG_FILE.exists():
                        try:
                            lines = LOG_FILE.read_text().split("\n")[-10:]
                            console.print("\n[red]Server log:[/]")
                            for line in lines:
                                if line.strip():
                                    console.print(f"  {line}")
                        except:
                            pass
                    console.print("\n[bold red]Server exited unexpectedly.[/]")
                    return
                try:
                    with urllib.request.urlopen(
                        f"http://{cfg['host']}:{cfg['port']}/health", timeout=2
                    ) as r:
                        if json.loads(r.read()).get("status") == "ok":
                            live.stop()
                            console.print(
                                "\n[bold green]Model Loaded -- Server Ready![/]"
                            )
                            console.print(f"http://{cfg['host']}:{cfg['port']}/v1")
                            console.print("[dim]Press Ctrl+C to stop[/]")
                            break
                except:
                    pass
                live.update(Spinner("dots", text=f"Loading... ({i}s)", style="cyan"))
        while proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/]")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except:
            proc.kill()
        console.print("[green]Done.[/]")


def load_presets():
    if PRESETS.exists():
        try:
            return json.loads(PRESETS.read_text())
        except:
            pass
    return {}


def save_presets(p):
    CFG.mkdir(parents=True, exist_ok=True)
    PRESETS.write_text(json.dumps(p, indent=2))


def cmd_launch(name=None):
    p = load_presets()
    if name:
        if name not in p:
            console.print(f"[red]Preset '{name}' not found.[/]")
            return
        console.print(
            Panel(
                f"[bold]Loading Preset: {name}[/]", border_style="cyan", box=box.ROUNDED
            )
        )
        cfg = p[name]
    else:
        console.print(
            Panel(
                "[bold]Turbo CLI -- LLM Server Launcher[/]",
                border_style="blue",
                box=box.ROUNDED,
            )
        )
        m = questionary.path(
            "Model path (.gguf):", only_directories=False, qmark=""
        ).ask()
        if not m or not os.path.exists(m):
            console.print("[red]Cancelled or file not found.[/]")
            return
        console.print(
            f"[dim]-> {os.path.basename(m)} ({os.path.getsize(m) / 1024**3:.1f} GB)[/]"
        )
        console.print()
        k = questionary.select(
            "K Cache:", choices=TYPES, default="q8_0", qmark=""
        ).ask()
        v = questionary.select(
            "V Cache:", choices=TYPES, default="turbo3", qmark=""
        ).ask()
        console.print()
        c = questionary.select(
            "Context Length:",
            choices=[str(x) for x in CTX] + ["custom"],
            default="262144",
            qmark="",
        ).ask()
        cs = (
            int(c)
            if c != "custom"
            else int(
                questionary.text("Custom context:", default="262144", qmark="").ask()
                or "262144"
            )
        )
        n = int(
            questionary.text("GPU Layers (-ngl):", default="99", qmark="").ask() or "99"
        )
        port = int(
            questionary.text("Server Port:", default="8080", qmark="").ask() or "8080"
        )

        # Optional flags
        ts = questionary.text(
            "Tensor Split (-ts, e.g. 13,14 for dual GPU, leave empty for single):",
            qmark="",
        ).ask()
        parallel = int(
            questionary.text(
                "Parallel slots (--parallel):", default="1", qmark=""
            ).ask()
            or "1"
        )
        no_mmap = questionary.confirm(
            "Disable memory mapping (--no-mmap)?", default=False, qmark=""
        ).ask()
        reasoning = questionary.confirm(
            "Enable reasoning mode (--reasoning)?", default=False, qmark=""
        ).ask()
        mmproj = questionary.path(
            "Multimodal projector (--mmproj, leave empty if none):",
            only_directories=False,
            qmark="",
        ).ask()

        model_gb = os.path.getsize(m) / 1024**3
        # Assume hidden dimension of 4096 (typical for 7B-9B models like Qwen 2.5)
        HIDDEN_DIM = 4096

        # Calculate KV cache: K + V per layer × number of layers
        # Bytes per element: f16=2, q8_0=1, turbo3/4=~0.4 (compressed)
        def cache_bytes(cache_type):
            if cache_type == "f16":
                return 2.0
            elif cache_type in ["turbo3", "turbo4"]:
                return 0.4
            else:  # q8_0, q4_0
                return 1.0

        bytes_per_token = cache_bytes(k) + cache_bytes(v)
        # KV cache formula: ctx * n_kv_heads * head_dim * 2 * bytes_per_token / 1GB
        # n_kv_heads=8 (GQA), head_dim=128 standard for most models
        # factor of 2 accounts for both K and V tensors
        # Using n_kv_heads * head_dim * 2 = 2048 (typical for GQA models)
        KV_DIM = 2048  # n_kv_heads * head_dim * 2
        kv_gb = cs * KV_DIM * bytes_per_token / (1024**3)

        est = model_gb * min(1, n / 100) + kv_gb + 0.5
        vram = get_vram()

        console.print()
        t = Table(
            title="VRAM Estimate", show_header=False, box=box.SIMPLE, padding=(0, 2)
        )
        t.add_row("Model on GPU", f"{model_gb * min(1, n / 100):.1f} GB", style="cyan")
        t.add_row("KV Cache", f"{kv_gb:.1f} GB", style="yellow")
        t.add_row("Overhead", "~0.5 GB", style="dim")
        t.add_row("---", "---")
        t.add_row("[bold]Total[/]", f"[bold]{est:.1f} GB[/]")
        console.print(t)
        if est <= vram:
            console.print(f"  [green]Fits in {vram}GB VRAM[/]")
        else:
            console.print(f"  [yellow]Exceeds {vram}GB -- spills to RAM[/]")
            console.print(f"  [dim]Consider using -ngl 60-70 or turbo3 cache[/]")

        console.print()
        if questionary.confirm("Save as preset?", default=False, qmark="").ask():
            nm = questionary.text(
                "Preset name:",
                default=os.path.basename(m).replace(".gguf", ""),
                qmark="",
            ).ask()
            if nm:
                p[nm] = {
                    "model": m,
                    "k": k,
                    "v": v,
                    "ctx": cs,
                    "ngl": n,
                    "port": port,
                    "host": "127.0.0.1",
                    "ts": ts,
                    "parallel": parallel,
                    "no_mmap": no_mmap,
                    "reasoning": reasoning,
                    "mmproj": mmproj,
                }
                save_presets(p)
                console.print(f"[green]Saved preset '{nm}'[/]")
        cfg = {
            "model": m,
            "k": k,
            "v": v,
            "ctx": cs,
            "ngl": n,
            "port": port,
            "host": "127.0.0.1",
            "ts": ts,
            "parallel": parallel,
            "no_mmap": no_mmap,
            "reasoning": reasoning,
            "mmproj": mmproj,
        }
    run(cfg)


def cmd_presets():
    p = load_presets()
    if not p:
        console.print("[dim]No presets saved.[/]")
        return
    console.print(Panel("[bold]Saved Presets[/]", border_style="cyan", box=box.ROUNDED))
    t = Table("Name", "Model", "Context", "Created", box=box.SIMPLE, padding=(0, 2))
    for nm, c in sorted(p.items()):
        t.add_row(
            f"[cyan]{nm}[/]",
            os.path.basename(c.get("model", "?"))[:20],
            f"{c.get('ctx', '?')}",
            c.get("created", "?"),
        )
    console.print(t)
    if questionary.confirm("Launch one?", default=False, qmark="").ask():
        ch = questionary.select("Preset:", choices=sorted(p.keys()), qmark="").ask()
        if ch:
            cmd_launch(ch)


def cmd_preset_list():
    p = load_presets()
    if not p:
        console.print("[dim]No presets.[/]")
        return
    console.print(Panel("[bold]Presets[/]", border_style="cyan", box=box.ROUNDED))
    t = Table("Name", "Model", "Context", box=box.SIMPLE, padding=(0, 2))
    for nm, c in sorted(p.items()):
        t.add_row(
            f"[cyan]{nm}[/]",
            os.path.basename(c.get("model", "?"))[:20],
            f"{c.get('ctx', '?')}",
        )
    console.print(t)


def cmd_preset_create(name=None):
    console.print(
        Panel("[bold]Create Preset[/]", border_style="green", box=box.ROUNDED)
    )
    m = questionary.path("Model path:", only_directories=False, qmark="").ask()
    if not m:
        return
    k = questionary.select("K Cache:", choices=TYPES, default="q8_0", qmark="").ask()
    v = questionary.select("V Cache:", choices=TYPES, default="turbo3", qmark="").ask()
    c = int(
        questionary.select(
            "Context:", choices=[str(x) for x in CTX], default="262144", qmark=""
        ).ask()
    )
    n = int(questionary.text("GPU Layers:", default="99", qmark="").ask())
    port = int(questionary.text("Port:", default="8080", qmark="").ask())
    ts = questionary.text("Tensor Split (-ts, e.g. 13,14):", qmark="").ask()
    parallel = int(
        questionary.text("Parallel slots:", default="1", qmark="").ask() or "1"
    )
    no_mmap = questionary.confirm("No mmap?", default=False, qmark="").ask()
    reasoning = questionary.confirm("Reasoning mode?", default=False, qmark="").ask()
    mmproj = questionary.path(
        "MMProj path (optional):", only_directories=False, qmark=""
    ).ask()
    nm = (
        name
        or questionary.text(
            "Name:", default=os.path.basename(m).replace(".gguf", ""), qmark=""
        ).ask()
    )
    if not nm:
        return
    p = load_presets()
    p[nm] = {
        "model": m,
        "k": k,
        "v": v,
        "ctx": c,
        "ngl": n,
        "port": port,
        "host": "127.0.0.1",
        "ts": ts,
        "parallel": parallel,
        "no_mmap": no_mmap,
        "reasoning": reasoning,
        "mmproj": mmproj,
        "created": time.strftime("%H:%M"),
    }
    save_presets(p)
    console.print(f"[green]Saved '{nm}'[/]")


def cmd_preset_remove():
    p = load_presets()
    if not p:
        return
    ch = questionary.checkbox("Remove:", choices=sorted(p.keys()), qmark="").ask()
    if not ch:
        return
    for nm in ch:
        del p[nm]
    save_presets(p)
    console.print(f"[green]Removed {len(ch)}[/]")


def cmd_preset_export(file=None):
    p = load_presets()
    out = file or questionary.text("Export to:", default="presets.json", qmark="").ask()
    if out:
        Path(out).write_text(json.dumps(p, indent=2))
        console.print(f"[green]Exported to {out}[/]")


def cmd_preset_import(file=None):
    src = (
        file or questionary.path("Import from:", only_directories=False, qmark="").ask()
    )
    if not src or not os.path.exists(src):
        return
    try:
        imp = json.loads(Path(src).read_text())
        p = load_presets()
        new = sum(1 for n in imp if p.setdefault(n, imp[n]) == imp[n])
        save_presets(p)
        console.print(f"[green]Imported {new} new[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


def cmd_vram(subcmd=None, value=None):
    if subcmd is None:
        vram = get_vram()
        console.print(f"[cyan]Current VRAM setting:[/] {vram} GB")
        console.print("[dim]Use 'turbo vram set <gb>' to change[/]")
    elif subcmd == "set":
        if value is None:
            val = int(
                questionary.text("VRAM (GB):", default=str(get_vram()), qmark="").ask()
                or str(get_vram())
            )
        else:
            val = int(value)
        set_vram(val)
        console.print(f"[green]VRAM set to {val} GB[/]")
    else:
        console.print("[red]Unknown subcommand. Use: turbo vram [set <gb>][/]")
        sys.exit(1)


def cmd_update():
    """Download latest version from GitHub and update installation"""
    console.print(
        Panel("[bold]Updating Turbo CLI...[/]", border_style="cyan", box=box.ROUNDED)
    )

    # Get current installation path
    install_path = Path(__file__).parent.parent.parent
    console.print(f"[dim]Current install: {install_path}[/]")

    try:
        # Get latest release version
        api_url = "https://api.github.com/repos/md-exitcode0/turbo-cli/releases/latest"
        console.print("[dim]Checking for latest release...[/]")

        with urllib.request.urlopen(api_url) as resp:
            release_data = json.loads(resp.read())
            tag_name = release_data["tag_name"]
            zipball_url = release_data["zipball_url"]

        console.print(f"[dim]Latest release: {tag_name}[/]")

        # Download release zip
        console.print("[dim]Downloading...[/]")

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            urllib.request.urlretrieve(zipball_url, tmp.name)
            zip_path = tmp.name

        # Extract to temp
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)

            # Find extracted folder (turbo-cli-main/)
            extracted = next(Path(tmpdir).iterdir())

            # Copy updated files, preserving user config
            console.print("[dim]Updating files...[/]")

            # Files to update
            files_to_update = [
                ("src/turbo/cli.py", "src/turbo/cli.py"),
                ("src/turbo/engine.py", "src/turbo/engine.py"),
                ("setup.py", "setup.py"),
                ("pyproject.toml", "pyproject.toml"),
                ("setup.bat", "setup.bat"),
            ]

            for src_rel, dst_rel in files_to_update:
                src = extracted / src_rel
                if src.exists():
                    dst = install_path / dst_rel
                    shutil.copy2(src, dst)
                    console.print(f"  Updated: {dst_rel}", style="green")

            # Update engine.zip if exists
            engine_zip_src = extracted / "src/turbo/data/engine.zip"
            engine_zip_dst = install_path / "src/turbo/data/engine.zip"
            if engine_zip_src.exists():
                shutil.copy2(engine_zip_src, engine_zip_dst)
                console.print("  Updated: engine.zip", style="green")

            # Update bin/ if exists
            bin_src = extracted / "src/turbo/bin"
            bin_dst = install_path / "src/turbo/bin"
            if bin_src.exists():
                if bin_dst.exists():
                    shutil.rmtree(bin_dst)
                shutil.copytree(bin_src, bin_dst)
                console.print("  Updated: binaries", style="green")

            os.unlink(zip_path)

        console.print()
        console.print("[bold green]Update complete![/] Run 'turbo --help' to verify.")
        console.print("[dim]Your presets in ~/.turbo/ were preserved.[/]")

    except Exception as e:
        console.print(f"[red]Update failed: {e}[/]")
        console.print(
            "[dim]Manual update: git clone https://github.com/md-exitcode0/turbo-cli.git[/]"
        )


def main():
    parser = argparse.ArgumentParser(
        prog="turbo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Turbo CLI -- LLM Server Launcher",
        add_help=False,
        epilog="""
Commands:
  turbo launch              Start server interactively or with preset
  turbo launch <name>       Launch saved preset
  turbo presets             List presets interactively
  turbo preset create       Create new preset
  turbo preset list         List all presets
  turbo preset remove       Remove presets
  turbo preset export       Export to JSON
  turbo preset import       Import from JSON
  turbo vram                Show current VRAM setting
  turbo vram set <gb>       Set VRAM limit (e.g., 'turbo vram set 12')
  turbo update              Download latest version from GitHub

Options:
  -v, --version             Show version number

Run 'turbo <command> --help' for more information on a command.
""",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"turbo-cli {__version__}"
    )
    parser.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    sub = parser.add_subparsers(dest="cmd", help="Command to run")
    sub.add_parser(
        "launch", help="Start server interactively or with preset"
    ).add_argument("preset", nargs="?", help="Preset name")
    sub.add_parser("presets", help="List presets interactively")
    pp = sub.add_parser("preset", help="Manage presets")
    pps = pp.add_subparsers(dest="preset_cmd")
    pps.add_parser("list", help="List presets")
    pps.add_parser("remove", help="Remove presets")
    pps.add_parser("create", help="Create preset").add_argument("name", nargs="?")
    pps.add_parser("export", help="Export to JSON").add_argument("file", nargs="?")
    pps.add_parser("import", help="Import from JSON").add_argument("file", nargs="?")
    vram_parser = sub.add_parser("vram", help="Manage VRAM settings")
    vram_subs = vram_parser.add_subparsers(dest="vram_cmd", help="VRAM subcommand")
    vram_set = vram_subs.add_parser("set", help="Set VRAM limit")
    vram_set.add_argument("gb", type=int, help="VRAM limit in GB")
    sub.add_parser("update", help="Download latest version from GitHub")

    a = parser.parse_args()
    if a.cmd == "launch":
        cmd_launch(a.preset)
    elif a.cmd == "presets":
        cmd_presets()
    elif a.cmd == "preset":
        cmd = a.preset_cmd
        if cmd == "list":
            cmd_preset_list()
        elif cmd == "create":
            cmd_preset_create(a.name)
        elif cmd == "remove":
            cmd_preset_remove()
        elif cmd == "export":
            cmd_preset_export(a.file)
        elif cmd == "import":
            cmd_preset_import(a.file)
        else:
            pp.print_help()
    elif a.cmd == "vram":
        if a.vram_cmd == "set":
            cmd_vram(subcmd="set", value=a.gb)
        else:
            cmd_vram(subcmd=None)
    elif a.cmd == "update":
        cmd_update()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
