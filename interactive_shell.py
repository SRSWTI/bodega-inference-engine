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

def print_header():
    console.print("\n" + "=" * 70, style="bold magenta")
    console.print(" 🏪 BODEGA INFERENCE ENGINE — INTERACTIVE SHELL", style="bold cyan", justify="center")
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
    table.add_row("8", "Exit")
    
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

def load_model():
    console.print("\n[bold cyan][+][/bold cyan] Dynamically Load a Model")
    path = Prompt.ask("Enter model_path", default="srswti/bodega-raptor-90m")
    if not path: return
    mid = Prompt.ask("Enter model_id (alias)", default=path)
    
    cb_choice = Confirm.ask("Enable Continuous Batching (High throughput)?")
    payload = {
        "model_path": path,
        "model_id": mid,
        "model_type": "lm",
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
    mid = Prompt.ask("Enter target model_id", default="srswti/bodega-raptor-90m")
    if not mid: return
    
    n_req = IntPrompt.ask("How many parallel requests?", default=2)
    
    import random
    sample_prompts = [
        "Write a Python function to calculate fibonacci numbers with memoization.",
        "Explain the difference between a list and a tuple in Python.",
        "Explain quantum computing in detail, covering: qubits, superposition, and entanglement.",
        "Write a comprehensive guide to building a production REST API in Python.",
        "What is 234 * 455? Explain your reasoning step-by-step.",
        "Solve for x: 2x + 5 = 15.",
        "Write a short, suspenseful story about a detective finding a strange artifact.",
        "Translate 'Hello, how are you today? I am learning artificial intelligence.' into French.",
        "If Mary is twice as old as her brother, and her brother is 5, how old will Mary be in 10 years? Let's think step by step.",
        "What are the benefits of continuous batching in LLM inference?",
        "Explain the plot of the movie Dune.",
        "Discuss the philosophy of existentialism and its main proponents.",
        "Write a bash script to bulk rename all .txt files to .md in a directory.",
        "Can you explain the Theory of Relativity to a 10-year-old?",
        "Create a React component that displays a counter with increment and decrement buttons.",
        "Write an SQL query to find the 3nd highest salary in an Employee table.",
        "Explain how a Transformer architecture works in neural networks.",
        "Provide a recipe for authentic Italian carbonara.",
        "Write a haiku about a rainy day in the city.",
        "What are the key differences between Postgres and MongoDB?",
        "Why is the sky blue? Explain the physics behind it.",
        "Write an email to a manager asking for a week of vacation next month.",
        "List 5 strategies to improve the performance of a website.",
        "Write a C++ program that implements binary search.",
        "Explain the difference between concurrent and parallel programming.",
        "Write a poem about the inevitability of time.",
        "How do you implement a load balancer? Describe the algorithms.",
        "Who was Alan Turing? Summarize his contributions to computer science.",
        "Explain Rust's ownership and borrowing system.",
        "Calculate the derivative of f(x) = x^3 + 2x^2 - x + 5."
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

            # Build combined display: think in dim, visible in bold green — full content, no truncation
            display = ""
            if s["think_text"]:
                display += f"[dim white]{s['think_text']}[/dim white]"
            if s["visible_text"]:
                display += f"[bold green]{s['visible_text']}[/bold green]"

            table.add_row(
                f"#{i+1}",
                f"[white]{trims_prompt}[/white]",
                f"{s['status']}\nTTFT: {s['ttft']}",
                display if display else "[dim]streaming...[/dim]"
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
    with Live(generate_table(), console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        asyncio.run(run_all(live))

# --- Interactive Chat Mode ---

def interactive_chat():
    console.print("\n[bold magenta][+] Interactive Chat Mode[/bold magenta]")
    mid = Prompt.ask("Enter target model_id", default="srswti/bodega-raptor-90m")
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

def main():
    while True:
        try:
            print_header()
            print_menu()
            choice = Prompt.ask("\nSelect an option", choices=[str(i) for i in range(1, 9)])
            
            if choice == '1': check_health()
            elif choice == '2': stream_download()
            elif choice == '3': load_model()
            elif choice == '4': unload_model()
            elif choice == '5': live_continuous_batching()
            elif choice == '6': interactive_chat()
            elif choice == '7': print_config_explanation()
            elif choice == '8': 
                console.print("\n[bold green]Exiting. Thank you for using the Bodega Inference Engine.[/bold green]")
                break
                
            input("\nPress Enter to return to menu...")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
