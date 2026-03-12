#!/usr/bin/env python3
"""
Bodega Inference Engine - Interactive Shell
===========================================

This interactive client showcases the power of the Bodega Inference Engine,
including:
  1) Multi-model Registry Dynamic Loading & Unloading
  2) Server-Sent Events (SSE) Streaming Downloads
  3) Continuous batching configuration
  4) Engine Health & RAM Usage Monitoring
  5) Live Continuous Batching Visualization
  6) Interactive Model Chat

Ensure the engine is running on port 44468 (or configure BASE_URL below).
"""

import sys
import json
import time
import asyncio
import httpx
import logging
import subprocess
import os
import random
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint
from rich.prompt import Prompt, IntPrompt, Confirm

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

BASE_URL = "http://localhost:44468"
console = Console()

MACTOP_SPECS_CACHE = None

def get_silicon_specs():
    global MACTOP_SPECS_CACHE
    if MACTOP_SPECS_CACHE is not None:
         return MACTOP_SPECS_CACHE
         
    try:
        if os.system("command -v mactop >/dev/null 2>&1") == 0:
            res = subprocess.run(["mactop", "--headless", "--count", "1"], capture_output=True, text=True, timeout=2.0)
            if res.returncode == 0:
                data = json.loads(res.stdout)
                if isinstance(data, list) and len(data) > 0:
                     info = data[0].get("system_info", {})
                     name = info.get("name", "Apple Silicon")
                     cores = info.get("core_count", "?")
                     pcores = info.get("p_core_count", "?")
                     ecores = info.get("e_core_count", "?")
                     gpu = info.get("gpu_core_count", "?")
                     
                     mem = data[0].get("memory", {})
                     total_mem = mem.get("total", 0) / (1024**3)
                     
                     MACTOP_SPECS_CACHE = f"[bold yellow]System:[/bold yellow] {name} ({cores} CPU Cores: {pcores}P/{ecores}E | {gpu} GPU Cores) - [bold yellow]RAM:[/bold yellow] {total_mem:.0f} GB"
                     return MACTOP_SPECS_CACHE
    except Exception:
         pass
         
    MACTOP_SPECS_CACHE = ""
    return MACTOP_SPECS_CACHE

def print_header():
    console.print("\n" + "=" * 70, style="bold magenta")
    console.print(" 🏪 BODEGA INFERENCE ENGINE — INTERACTIVE SHELL", style="bold cyan", justify="center")
    specs = get_silicon_specs()
    if specs:
        console.print(specs, justify="center")
    console.print("=" * 70, style="bold magenta")

def print_menu():
    table = Table(show_header=False, expand=True, box=None)
    table.add_column("Option", style="bold green", width=5)
    table.add_column("Description", style="white")
    
    table.add_row("1", "View Engine Health & Loaded Models")
    table.add_row("2", "Download a Model (Streaming SSE)")
    table.add_row("3", "Load a Model into Registry")
    table.add_row("4", "Unload a Model from Registry")
    table.add_row("5", "[bold yellow]Test Live Continuous Batching (Parallel Requests)[/bold yellow]")
    table.add_row("6", "[bold magenta]Interactive Chat Mode[/bold magenta]")
    table.add_row("7", "Read about config.yaml (Static Registry Setup)")
    table.add_row("8", "[bold bright_cyan]Launch Real-Time Apple Silicon Telemetry (mactop)[/bold bright_cyan]")
    table.add_row("9", "Exit")
    
    console.print("\n[bold cyan]--- Main Menu ---[/bold cyan]")
    console.print(table)

def check_health():
    console.print("\n[bold cyan][+][/bold cyan] Checking Engine Health...")
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            console.print(f"    [bold]Status:[/bold] {data.get('status')}")
            console.print(f"    [bold]Loaded Models:[/bold] {data.get('model_status')}")
        else:
            console.print(f"    [red]HTTP Error {r.status_code}[/red]")
        
        # Also check /v1/admin/loaded-models
        r_loaded = httpx.get(f"{BASE_URL}/v1/admin/loaded-models", timeout=5.0)
        if r_loaded.status_code == 200:
            models = r_loaded.json().get("data", [])
            console.print("\n    [bold cyan]Detailed Process Status[/bold cyan]")
            if not models:
                console.print("    [yellow]No models currently running.[/yellow]")
            else:
                table = Table(show_header=True, header_style="bold magenta", expand=True)
                table.add_column("Model ID")
                table.add_column("Status")
                table.add_column("PID")
                table.add_column("GPU Metal Active", justify="right")
                table.add_column("CPU Python RSS", justify="right")
                
                for m in models:
                    mem = m.get("memory", {})
                    gpu = f"{mem.get('metal_active_mb', 0):.1f} MB" if mem else "N/A"
                    cpu = f"{mem.get('rss_mb', 0):.1f} MB" if mem else "N/A"
                    status_c = "[green]running[/green]" if m.get('status') == "running" else f"[red]{m.get('status')}[/red]"
                    table.add_row(m.get('id', 'N/A'), status_c, str(m.get('pid', 'N/A')), gpu, cpu)
                
                console.print(table)
        else:
            console.print("    [yellow](Engine might not be running in multi-handler dynamic mode)[/yellow]")
            
    except Exception as e:
        console.print(f"    [red]Error connecting to server: {e}[/red]")

def get_first_loaded_model() -> str:
    """Helper to get the first currently loaded model_id from the engine."""
    try:
        r = httpx.get(f"{BASE_URL}/v1/admin/loaded-models", timeout=2.0)
        if r.status_code == 200:
            models = r.json().get("data", [])
            for m in models:
                if m.get('status') == "running":
                    return m.get('id')
    except Exception:
        pass
    return "srswti/bodega-raptor-90m"

def stream_download():
    console.print("\n[bold cyan][+][/bold cyan] Stream Model Download")
    console.print("    This invokes the /v1/admin/download-model-stream endpoint.")
    console.print("    1) srswti/bodega-raptor-90m (Small, quick test)")
    console.print("    2) srswti/bodega-raptor-8b-mxfp4 (Full 8B model)")
    console.print("    3) Custom HuggingFace repo ID")
    
    choice = input("Select an option: ")
    
    if choice == '1':
        model_path = "srswti/bodega-raptor-90m"
    elif choice == '2':
        model_path = "srswti/bodega-raptor-8b-mxfp4"
    elif choice == '3':
        model_path = input("Enter repo ID (e.g. mlx-community/Qwen2.5-1.5B-Instruct-4bit): ")
    else:
        console.print("[red]Invalid choice.[/red]")
        return

    url = f"{BASE_URL}/v1/admin/download-model-stream"
    console.print(f"\n[bold yellow]Downloading {model_path} from HuggingFace Hub...[/bold yellow]")
    try:
        with httpx.stream("POST", url, json={"model_path": model_path}, timeout=None) as r:
            if r.status_code != 200:
                console.print(f"[red]Failed to initiate download. Status: {r.status_code}[/red]")
                try: console.print(r.read().decode())
                except: pass
                return
            for line in r.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        console.print("\n[bold green][✔] Download sequence [DONE].[/bold green]")
                        break
                    try:
                        payload = json.loads(data_str)
                        msg = payload.get("message", "")
                        prog = payload.get("progress", 0)
                        sys.stdout.write(f"\r\033[K[Progress: {prog:>3}%] {msg:<60}")
                        sys.stdout.flush()
                        
                        if payload.get("status") == "error":
                            console.print("\n[red][!] Error during download.[/red]")
                            break
                    except json.JSONDecodeError:
                        pass
        console.print(f"\n[bold green][✔] {model_path} is cached on disk![/bold green]")
    except Exception as e:
        console.print(f"\n[red]Error mapping download stream: {e}[/red]")

def get_model_type(model_path: str) -> str:
    """Check HuggingFace API to see if the model needs multimodal or lm."""
    if not "/" in model_path:
         return "lm"
    try:
         resp = httpx.get(f"https://huggingface.co/api/models/{model_path}", timeout=5.0)
         if resp.status_code == 200:
              data = resp.json()
              pipeline_tag = data.get("pipeline_tag", "")
              if pipeline_tag == "image-text-to-text":
                   return "multimodal"
    except Exception:
         pass
    return "lm"

def load_model():
    console.print("\n[bold cyan][+][/bold cyan] Dynamically Load a Model")
    path = Prompt.ask("Enter model_path", default="srswti/bodega-raptor-90m")
    if not path: return
    mid = Prompt.ask("Enter model_id (alias)", default=path)
    
    mtype = get_model_type(path)
    if mtype == "multimodal":
        console.print("  [cyan]-> Detected vision capabilities. Loading as multimodal.[/cyan]")
        
    cb_choice = Confirm.ask("Enable Continuous Batching (High throughput)?")
    payload = {
        "model_path": path,
        "model_id": mid,
        "model_type": mtype,
        "max_concurrency": 1
    }
    
    if cb_choice:
        payload["continuous_batching"] = True
        payload["cb_max_num_seqs"] = 256
        payload["cb_prefill_batch_size"] = 16
        payload["cb_completion_batch_size"] = 32
        payload["max_concurrency"] = 64
        console.print(f"  [green]-> Configured CB with {payload['cb_max_num_seqs']} max seqs.[/green]")
        
    console.print(f"\n[bold yellow]Spawning isolated handler process for {mid}...[/bold yellow]")
    try:
        r = httpx.post(f"{BASE_URL}/v1/admin/load-model", json=payload, timeout=120)
        if r.status_code in [200, 201]:
            console.print(f"[bold green][✔] Model '{mid}' successfully loaded to memory and is ready for inference![/bold green]")
        else:
            console.print(f"[red][!] Failed. Code: {r.status_code}[/red]")
            console.print(r.text)
    except Exception as e:
        console.print(f"[red][!] Request error: {e}[/red]")

def unload_model():
    console.print("\n[bold cyan][+][/bold cyan] Unload a Model")
    mid = Prompt.ask("Enter the model_id to gracefully terminate")
    if not mid: return
    
    console.print(f"[yellow]Unloading handler for {mid}...[/yellow]")
    try:
        r = httpx.delete(f"{BASE_URL}/v1/admin/unload-model/{mid}", timeout=30)
        if r.status_code in [200, 204]:
            console.print(f"[bold green][✔] Unified Memory freed. Process gracefully killed.[/bold green]")
        else:
            console.print(f"[red][!] Failed. Code: {r.status_code}[/red]")
            console.print(r.text)
    except Exception as e:
        console.print(f"[red][!] Request error: {e}[/red]")

# --- Real-Time Continuous Batching Visualizer ---

async def run_one_stream(client, url, payload, index, state):
    """Stream one request, tracking think vs visible tokens separately."""
    t0 = time.perf_counter()
    ttft = None
    in_think = False
    try:
        async with client.stream("POST", url, json=payload, timeout=120) as r:
            if r.status_code != 200:
                state[index]["think_text"] = ""
                state[index]["visible_text"] = f"Error {r.status_code}"
                state[index]["status"] = "[red]Error[/red]"
                state[index]["done"] = True
                return
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    dstr = line[6:]
                    if dstr == "[DONE]":
                        tt = time.perf_counter() - t0
                        state[index]["status"] = f"[green]Done in {tt:.1f}s[/green]"
                        state[index]["done"] = True
                        break
                    try:
                        data = json.loads(dstr)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                 token = delta["content"]
                                 # Set TTFT on the very first token regardless
                                 if ttft is None:
                                     ttft = time.perf_counter() - t0
                                     state[index]["ttft"] = f"{ttft*1000:.0f}ms"
                                 # Correctly split on think boundaries within a single token
                                 if "<think>" in token:
                                     in_think = True
                                     parts = token.split("<think>", 1)
                                     if parts[0]: state[index]["visible_text"] += parts[0]
                                     state[index]["think_text"] += "<think>" + parts[1]
                                 elif "</think>" in token:
                                     in_think = False
                                     parts = token.split("</think>", 1)
                                     state[index]["think_text"] += parts[0] + "</think>"
                                     if parts[1]: state[index]["visible_text"] += parts[1]
                                 elif in_think:
                                     state[index]["think_text"] += token
                                 else:
                                     state[index]["visible_text"] += token
                    except: pass
    except Exception as e:
        state[index]["visible_text"] += f" Error: {e}"
        state[index]["status"] = "[red]Error[/red]"
        state[index]["done"] = True

def live_continuous_batching():
    console.print("\n[bold cyan][+] Real-Time Continuous Batching Visualizer[/bold cyan]")
    default_model = get_first_loaded_model()
    mid = Prompt.ask("Enter target model_id", default=default_model)
    if not mid: return
    
    n_req = IntPrompt.ask("How many parallel requests?", default=2)
    
    sample_prompts = [
        # Simple / Fun
        "Name a primary color.",
        "Translate 'apple' to Spanish.",
        "Say 'Good morning' in French.",
        "Write a haiku about a robot.",
        "What noise does a cow make?",
        
        # Coding (Short)
        "Write a short Python function to reverse a string.",
        "What is a 'pointer' in C? Explain in one sentence.",
        "Write a SQL query to find the maximum salary in an Employee table.",
        "Explain the difference between POST and GET in HTTP briefly.",
        
        # Physics / Science (Short)
        "What is Einstein's equation for mass-energy equivalence?",
        "Briefly explain the Heisenberg Uncertainty Principle.",
        "Why does a helium balloon float? Give a one-sentence answer.",
        "If you drop a feather and a bowling ball on the moon, which hits the ground first?",
        
        # Reasoning / Life / Riddles
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "What is the meaning of life? Keep it under 10 words.",
        "I speak without a mouth and hear without ears. What am I?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        
        # Math
        "What is 25 * 4? Say just the number.",
        "What is the square root of 144?",
        "Calculate the derivative of x^2."
    ]
    random.shuffle(sample_prompts)
    
    prompts = []
    for i in range(n_req):
        default_prompt = sample_prompts[i % len(sample_prompts)]
        p = Prompt.ask(f"Enter prompt #{i+1}", default=default_prompt)
        prompts.append(p)
        
    state = [
        {"think_text": "", "visible_text": "", "status": "[yellow]Waiting...[/yellow]",
         "ttft": "...", "done": False, "prompt": p}
        for p in prompts
    ]

    def generate_table():
        table = Table(show_header=True, header_style="bold magenta", expand=True, show_lines=True)
        table.add_column("Req", justify="center", width=5)
        table.add_column("Prompt", width=22)
        table.add_column("Status / TTFT", width=16)
        table.add_column("Live Output  [dim](grey = thinking)[/dim]", ratio=1)

        for i, s in enumerate(state):
            trims_prompt = s["prompt"][:35] + "..." if len(s["prompt"]) > 35 else s["prompt"]

            think_clean = s["think_text"].replace("<think>", "").replace("</think>", "").strip()
            vis_clean = s["visible_text"].strip()
            
            lines = []
            for line in think_clean.split("\n"):
                if line.strip(): lines.append(f"[dim white]{line.strip()}[/dim white]")
            for line in vis_clean.split("\n"):
                if line.strip(): lines.append(f"[bold green]{line.strip()}[/bold green]")
            
            # Keep only the last 5 lines to stop infinite vertical scrolling!
            display = "\n".join(lines[-5:]) if lines else "[dim]streaming...[/dim]"

            table.add_row(
                f"#{i+1}",
                f"[white]{trims_prompt}[/white]",
                f"{s['status']}\nTTFT: {s['ttft']}",
                display
            )
        return table

    url = f"{BASE_URL}/v1/chat/completions"

    async def run_all(live_display):
        async with httpx.AsyncClient() as client:
            tasks = []
            for i, p in enumerate(prompts):
                payload = {
                    "model": mid,
                    "messages": [{"role": "user", "content": p}],
                    "stream": True,
                    "max_tokens": 2000
                }
                tasks.append(run_one_stream(client, url, payload, i, state))
            
            # Update live display concurrently while generating
            async def refresh():
                while not all(s["done"] for s in state):
                    live_display.update(generate_table())
                    await asyncio.sleep(0.08)
                live_display.update(generate_table())
                
            await asyncio.gather(refresh(), *tasks)

    console.print("\n[bold yellow]Firing continuous batching cluster...[/bold yellow]\n")
    
    # Open mactop side-by-side in a new Terminal window
    console.print("  [dim]Opening mactop telemetry window...[/dim]")
    os.system("osascript -e 'tell application \"Terminal\" to do script \"mactop\"' >/dev/null 2>&1")
    
    # Auto-load the model with lm→multimodal fallback
    console.print(f"  [cyan]Loading model {mid}...[/cyan]")
    load_ok = False
    mtype_detected = get_model_type(mid)
    types_to_try = [mtype_detected, "multimodal" if mtype_detected == "lm" else "lm"]
    for attempt_type in types_to_try:
        console.print(f"  [dim]-> Applying Continuous Batching configs for model_type '{attempt_type}'...[/dim]")
        try:
            r = httpx.post(f"{BASE_URL}/v1/admin/load-model", json={
                "model_path": mid, "model_id": mid,
                "model_type": attempt_type,
                "continuous_batching": True,
                "cb_max_num_seqs": 128,
                "context_length": 8192
            }, timeout=120)
            if r.status_code == 409:
                console.print(f"  [green]✓ Already loaded (as {attempt_type})[/green]")
                if attempt_type == "multimodal":
                    console.print("  [yellow]⚠ Note: Continuous batching for 'multimodal' models is coming soon to Bodega.\n"
                                  "    The engine currently falls back to sequential execution for vision models.[/yellow]")
                    if not Confirm.ask("  Continue anyway?", default=True):
                        return
                load_ok = True
                break
            elif r.status_code in [200, 201]:
                console.print(f"  [green]✓ Loaded as {attempt_type}[/green]")
                if attempt_type == "multimodal":
                    console.print("  [yellow]⚠ Note: Continuous batching for 'multimodal' models is coming soon to Bodega.\n"
                                  "    The engine currently falls back to sequential execution for vision models.[/yellow]")
                    if not Confirm.ask("  Continue anyway?", default=True):
                        return
                load_ok = True
                break
            elif r.status_code == 500:
                console.print(f"  [yellow]Load as '{attempt_type}' failed (500), trying next...[/yellow]")
                continue
            else:
                try:
                    err = r.json()
                    msg = err.get("error", {}).get("message", r.text[:120])
                except Exception:
                    msg = r.text[:120]
                console.print(f"  [red]failed ({r.status_code}): {msg}[/red]")
                break
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            break
    
    if not load_ok:
        console.print("  [red]✗ This model could not be loaded as 'lm' or 'multimodal'. "
                      "Ensure it has an MLX tag on HuggingFace to be compatible with the Bodega Inference Engine.[/red]")
        return
    
    with Live(generate_table(), console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        asyncio.run(run_all(live))
    
    # Unload after done
    try:
        httpx.delete(f"{BASE_URL}/v1/admin/unload-model/{mid}", timeout=30)
        console.print(f"  [dim]Model {mid} unloaded.[/dim]")
    except Exception:
        pass

# --- Interactive Chat Mode ---

def interactive_chat():
    console.print("\n[bold magenta][+] Interactive Chat Mode[/bold magenta]")
    default_model = get_first_loaded_model()
    mid = Prompt.ask("Enter target model_id", default=default_model)
    if not mid: return
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    
    url = f"{BASE_URL}/v1/chat/completions"
    
    console.print(f"\n[green]Connected to {mid}. Type 'exit' to quit.[/green]\n")
    
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})
            
            payload = {
                "model": mid,
                "messages": messages,
                "stream": True,
                "max_tokens": 8192
            }
            
            console.print("[bold magenta]Assistant:[/bold magenta] ", end="")
            
            raw_reply = ""
            in_think = False
            with httpx.stream("POST", url, json=payload, timeout=120) as r:
                if r.status_code != 200:
                    console.print(f"[red]Error {r.status_code}[/red]")
                    messages.pop()
                    continue
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        dstr = line[6:]
                        if dstr == "[DONE]": break
                        try:
                            data = json.loads(dstr)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "content" in delta:
                                    token = delta["content"]
                                    raw_reply += token
                                    if "<think>" in token:
                                        in_think = True
                                        sys.stdout.write("\033[2m")  # dim ANSI
                                        sys.stdout.flush()
                                    if "</think>" in token:
                                        in_think = False
                                        sys.stdout.write(token + "\033[0m")  # reset ANSI
                                        sys.stdout.flush()
                                        continue
                                    sys.stdout.write(token)
                                    sys.stdout.flush()
                        except: pass
            if in_think:
                sys.stdout.write("\033[0m")
            console.print("\n")
            messages.append({"role": "assistant", "content": raw_reply})
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting chat.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            break

def print_config_explanation():
    console.print("\n[bold cyan][+] About config.yaml (Static Setup)[/bold cyan]")
    text = """
The engine natively supports a dynamic multi-model registry where models
are isolated in their own processes. While this interactive shell uses the
API endpoints (/v1/admin/load-model), you can also set up a static registry
using a `config.yaml` file on engine startup.

Example config.yaml:
----------------------------------------
server:
  host: "0.0.0.0"
  port: 44468

models:
  - model_id: "bodega-solomon-9b"
    model_type: "multimodal"
    model_path: "srswti/bodega-solomon-9b"
    max_concurrency: 1

  - model_id: "bodega-raptor-90m"
    model_type: "lm"
    model_path: "srswti/bodega-raptor-90m"
    continuous_batching: true
    cb_max_num_seqs: 128
----------------------------------------
To launch the engine using this: The config path is supplied via environment 
variables or args depending on the entry point wrapper.
"""
    console.print(Panel(text, title="config.yaml Explained", border_style="blue"))

def launch_mactop():
    console.print("\n[bold cyan][+][/bold cyan] Launching Real-Time Apple Silicon Telemetry (mactop)...")
    if os.system("command -v mactop >/dev/null 2>&1") != 0:
        console.print("[red]mactop is not installed. Please run: brew install mactop[/red]")
        return
    
    # Open mactop in a NEW Terminal window side-by-side via osascript
    script = '''
    tell application "Terminal"
        do script "mactop"
        activate
    end tell
    '''
    ret = os.system(f"osascript -e '{script}'")
    if ret == 0:
        console.print("  [green]✓ mactop launched in a new Terminal window.[/green]")
        console.print("  [dim]Close that window or press q inside mactop to stop it.[/dim]")
    else:
        console.print("  [yellow]osascript failed — falling back to running mactop here (press q to exit).[/yellow]")
        time.sleep(1)
        os.system("mactop")

def main():
    while True:
        try:
            print_header()
            print_menu()
            choice = Prompt.ask("\nSelect an option", choices=[str(i) for i in range(1, 10)])
            
            if choice == '1': check_health()
            elif choice == '2': stream_download()
            elif choice == '3': load_model()
            elif choice == '4': unload_model()
            elif choice == '5': live_continuous_batching()
            elif choice == '6': interactive_chat()
            elif choice == '7': print_config_explanation()
            elif choice == '8': launch_mactop()
            elif choice == '9': 
                console.print("\n[bold green]Exiting. Thank you for using the Bodega Inference Engine.[/bold green]")
                break
                
            input("\nPress Enter to return to menu...")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
