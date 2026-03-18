#!/bin/bash
# Bodega Inference Engine Setup Script

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Bodega Inference Engine Setup ===${NC}"
echo ""
echo -e "${YELLOW}Step 1: Bodega Sensors & Inference Engine${NC}"
echo ""
echo -e "  The ${GREEN}Bodega Inference Engine${NC} runs inside the ${GREEN}Bodega Sensors${NC} app."
echo ""
read -p "  Do you already have Bodega Sensors installed? [y/N]: " has_sensors
echo ""

if [[ "$has_sensors" == "y" || "$has_sensors" == "Y" ]]; then
    echo -e "${GREEN}✓ Skipping download — you already have Bodega Sensors.${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}ACTION REQUIRED:${NC}"
    echo -e "  1. Open ${GREEN}Bodega Sensors${NC} from your Applications folder."
    echo -e "  2. Find the ${YELLOW}Bodega Inference Engine${NC} toggle and turn it ON."
    echo -e "  3. Wait for the toggle to turn ${GREEN}GREEN${NC}."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "  Have you turned the toggle GREEN? [y/N]: " toggle_done
else
    echo -e "  Downloading Bodega Sensors (contains the Inference Engine)..."
    echo ""

    if [ -f "./install_sensors.sh" ]; then
        bash ./install_sensors.sh
    else
        echo -e "${RED}Error: install_sensors.sh not found in the current directory.${NC}"
        exit 1
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}Step 1:${NC} Open the DMG from ${BLUE}~/Downloads${NC} and drag ${GREEN}Bodega Sensors${NC} into Applications."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "  Have you copied Bodega Sensors to Applications? [y/N]: " copy_done

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}Step 2:${NC} Open ${GREEN}Bodega Sensors${NC} from your Applications folder."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "  Is the Bodega Sensors app open now? [y/N]: " app_open

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  ${GREEN}Step 3:${NC} Find the ${YELLOW}Bodega Inference Engine${NC} toggle and turn it ON. Wait for ${GREEN}GREEN${NC}."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "  Have you turned the toggle GREEN? [y/N]: " toggle_done
fi

echo ""
echo -e "${YELLOW}Step 1.b: Apple Silicon Telemetry (mactop)${NC}"
echo -e "  ${BLUE}mactop${NC} is a real-time monitor for Apple Silicon — it shows GPU/CPU/RAM usage"
echo -e "  and power draw while benchmarks run, so you can see your chip working live."

SKIP_TELEMETRY=0

if command -v mactop &> /dev/null; then
    echo -e "  ${GREEN}✓ mactop is already installed. Telemetry is ready.${NC}"
else
    echo ""
    read -p "  Would you like to install mactop for live telemetry? [Y/n]: " install_mactop
    if [[ "$install_mactop" == "n" || "$install_mactop" == "N" ]]; then
        echo -e "  ${YELLOW}Skipping telemetry. Benchmarks will still run, just without live chip stats.${NC}"
        SKIP_TELEMETRY=1
    else
        if command -v brew &> /dev/null; then
            # Homebrew already there — just install mactop silently
            echo -e "  ${BLUE}Installing mactop via Homebrew...${NC}"
            brew install mactop
            echo -e "  ${GREEN}✓ mactop installed.${NC}"
        else
            echo ""
            echo -e "  ${YELLOW}Homebrew is a package manager for macOS — needed to install mactop.${NC}"
            echo -e "  It's safe, widely used, and takes ~1-2 minutes to install."
            read -p "  Install Homebrew now? [Y/n]: " install_brew
            if [[ "$install_brew" == "n" || "$install_brew" == "N" ]]; then
                echo -e "  ${YELLOW}Skipping Homebrew & mactop. Telemetry will be unavailable.${NC}"
                SKIP_TELEMETRY=1
            else
                echo -e "  ${BLUE}Installing Homebrew...${NC}"
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || eval "$(/usr/local/bin/brew shellenv 2>/dev/null)"
                if command -v brew &> /dev/null; then
                    echo -e "  ${BLUE}Installing mactop...${NC}"
                    brew install mactop
                    echo -e "  ${GREEN}✓ mactop installed.${NC}"
                else
                    echo -e "  ${RED}Homebrew installation failed. Skipping mactop.${NC}"
                    SKIP_TELEMETRY=1
                fi
            fi
        fi
    fi
fi

echo ""
echo -e "${YELLOW}Step 1.c: Python Dependencies${NC}"
echo -e "  ${BLUE}Ensuring Python packages (httpx, huggingface_hub, rich, etc.) are installed...${NC}"
BODEGA_TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BODEGA_TESTS_DIR"
if [ -f "requirements.txt" ]; then
    if [ -d ".venv" ]; then
        echo -e "  ${GREEN}✓ Using existing .venv${NC}"
        source .venv/bin/activate
    else
        echo -e "  ${BLUE}Creating .venv and installing packages...${NC}"
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -q --upgrade pip
        pip install -q -r requirements.txt
        echo -e "  ${GREEN}✓ Packages installed${NC}"
    fi
    # Ensure we use this venv for subsequent python calls in this script
    export VIRTUAL_ENV="$BODEGA_TESTS_DIR/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
else
    echo -e "  ${YELLOW}requirements.txt not found — skipping venv setup.${NC}"
    echo -e "  ${YELLOW}Install manually: pip install httpx huggingface_hub rich tabulate loguru${NC}"
fi

echo ""
echo -e "${YELLOW}Step 1.d: Hardware Detection${NC}"
if HW_STR=$(python3 hardware_info.py 2>/dev/null); then
    echo -e "  ${GREEN}✓ Detected: ${HW_STR}${NC}"
else
    echo -e "  ${YELLOW}(Could not detect hardware — ensure psutil is installed)${NC}"
fi

echo ""
echo -e "${YELLOW}Step 1.e: Connecting to Bodega Inference Engine${NC}"
echo -e "  Verifying engine is ready on localhost:44468..."
POLL_INTERVAL=12
while ! curl -s http://localhost:44468/health >/dev/null; do
    echo -e "${RED}  Waiting for engine. Please ensure the toggle is GREEN in Bodega Sensors!${NC}"
    echo -e "  (Checking again in ${POLL_INTERVAL} seconds...)"
    sleep "$POLL_INTERVAL"
done
echo -e "  ${GREEN}✓ Engine is ready.${NC}"

echo ""
echo -e "${YELLOW}Step 2: Model Selection${NC}"
echo "Which model would you like to download?"
echo "1) Bodega ORION 0.6B (srswti/bodega-orion-0.6b) - Ultra-fast, great for continuous batching tests"
echo "2) Custom Model Repository from HuggingFace"
read -p "Select an option [1-2]: " model_choice

MODELS=()
if [[ "$model_choice" == "1" ]]; then
    MODELS+=("srswti/bodega-orion-0.6b")
fi
if [[ "$model_choice" == "2" ]]; then
    read -p "Enter HuggingFace model path (e.g. mlx-community/JOSIE-IT1-Qwen3-0.6B-4bit): " custom_model
    if [ -n "$custom_model" ]; then
        MODELS+=("$custom_model")
    fi
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    TARGET_MODEL="srswti/bodega-orion-0.6b"
else
    TARGET_MODEL=${MODELS[0]}
fi

echo -e "\n${GREEN}Starting downloads...${NC}"

for model in "${MODELS[@]}"; do
    echo -e "\n${BLUE}Downloading $model...${NC}"
    if ! python3 -c "
import sys, json, httpx

url = 'http://localhost:44468/v1/admin/download-model-stream'
model_path = '''$model'''
try:
    with httpx.stream('POST', url, json={'model_path': model_path}, timeout=None) as r:
        if r.status_code != 200:
            print(f'\033[0;31mError {r.status_code} - Is the engine running?\033[0m')
            sys.exit(1)
            
        got_done = False
        for line in r.iter_lines():
            if line.startswith('data: '):
                dstr = line[6:]
                if dstr == '[DONE]':
                    print('\n\033[0;32m✓ Download Complete!\033[0m')
                    got_done = True
                    break
                try:
                    data = json.loads(dstr)
                    if 'message' in data:
                        prog = data.get('progress', 0)
                        sys.stdout.write(f'\r\033[K[Progress: {prog:>3}%] ' + data['message'][:60])
                        sys.stdout.flush()
                except Exception:
                    pass
        if not got_done:
            print('\n\033[0;31mDownload stream did not complete [DONE] - model may not be fully downloaded.\033[0m')
            sys.exit(1)
except Exception as e:
    print(f'\n\033[0;31mError downloading: {e}\033[0m')
    sys.exit(1)
"; then
        DOWNLOAD_FAILED=1
        echo -e "${RED}✗ Download failed for $model. Cannot proceed to load.${NC}"
        echo -e "${YELLOW}  Please check: Bodega Sensors is running, toggle is GREEN, and you have network/HuggingFace access.${NC}"
        exit 1
    fi
done

echo -e "\n${GREEN}=== Downloads Complete ===${NC}"
echo ""

if [ ${#MODELS[@]} -gt 0 ]; then
    TARGET_MODEL=${MODELS[0]}
fi

# ─── Load model and inspect what adapter type the engine assigns ───────────────
echo -e "${YELLOW}Loading and inspecting model adapter type...${NC}"

# Write the inspector to a temp file. Uses detect_model_type.py (config.json) to detect
# model_type before load — no lm→multimodal retry.
BODEGA_TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"
cat > /tmp/bodega_model_inspect.py << 'PYEOF'
import sys, json, httpx, time

# Add bodega_engine_tests to path for detect_model_type (passed as argv[2])
if len(sys.argv) >= 3:
    sys.path.insert(0, sys.argv[2])
from detect_model_type import detect_model_type

BASE = "http://localhost:44468"
model_id = sys.argv[1]

# Detect model_type from config.json — no retry
mtype = detect_model_type(model_id)
print(f"  [->] Detected model_type from config.json: {mtype}", flush=True)

def load_model(mtype):
    try:
        r = httpx.post(f"{BASE}/v1/admin/load-model", json={
            "model_path": model_id,
            "model_id": model_id,
            "model_type": mtype,
            "context_length": 8192,
            "max_concurrency": 8,
        }, timeout=120)
        return r.status_code
    except Exception:
        return 0

code = load_model(mtype)
if code in [200, 201, 409]:
    print(f"  [ok] Loaded as {mtype} (status={code})", flush=True)
else:
    print(f"  [!] Load failed (status={code}). Proceeding with inspection.", flush=True)

# Wait for engine to settle, then read actual adapter type
time.sleep(1)
try:
    r = httpx.get(f"{BASE}/v1/admin/loaded-models", timeout=5)
    models = r.json().get("data", [])
    m = next((x for x in models if x.get("id") == model_id), None)
    if m:
        mem   = m.get("memory", {})
        mtype = m.get("type", m.get("model_type", "lm"))
        rss   = mem.get("rss_mb", 0)
        metal = mem.get("metal_peak_mb", mem.get("metal_active_mb", 0))
        total = mem.get("total_mb", 0)
        pid   = m.get("pid", "N/A")
        print(f"MODEL_TYPE={mtype}")
        print(f"RSS_MB={rss:.0f}")
        print(f"METAL_MB={metal:.0f}")
        print(f"TOTAL_MB={total:.0f}")
        print(f"PID_VAL={pid}")
    else:
        print("MODEL_TYPE=lm"); print("RSS_MB=0"); print("METAL_MB=0"); print("TOTAL_MB=0"); print("PID_VAL=N/A")
except Exception:
    print("MODEL_TYPE=lm"); print("RSS_MB=0"); print("METAL_MB=0"); print("TOTAL_MB=0"); print("PID_VAL=N/A")
PYEOF

# Run it and capture only the KEY=VALUE lines into shell vars (with defaults)
MODEL_TYPE=lm
RSS_MB=0
METAL_MB=0
TOTAL_MB=0
PID_VAL=N/A
eval "$(python3 /tmp/bodega_model_inspect.py "$TARGET_MODEL" "$BODEGA_TESTS_DIR" 2>&1 | grep -E '^(MODEL_TYPE|RSS_MB|METAL_MB|TOTAL_MB|PID_VAL)=')" 2>/dev/null || true

# Validate: did the model actually load? (RSS_MB > 0 means real load)
MODEL_LOADED=0
if [[ "${RSS_MB:-0}" -gt 0 ]] || [[ "${TOTAL_MB:-0}" -gt 0 ]]; then
    MODEL_LOADED=1
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [[ "$MODEL_LOADED" == "1" ]]; then
    echo -e "  ${GREEN}Model Loaded:${NC}   $TARGET_MODEL"
    echo -e "  ${GREEN}Adapter Type:${NC}   ${MODEL_TYPE:-lm}"
    echo -e "  ${GREEN}RAM (RSS):${NC}      ${RSS_MB:-0} MB"
    echo -e "  ${GREEN}Metal Peak:${NC}     ${METAL_MB:-0} MB  (Total: ${TOTAL_MB:-0} MB)"
    echo -e "  ${GREEN}PID:${NC}            ${PID_VAL:-N/A}"
else
    echo -e "  ${RED}Model FAILED to load:${NC} $TARGET_MODEL"
    echo -e "  ${YELLOW}RAM/Metal: 0 MB — no model process is running.${NC}"
    echo ""
    echo -e "  ${YELLOW}Possible causes:${NC}"
    echo -e "    • Download may have failed (check messages above)"
    echo -e "    • Bodega Inference Engine toggle may not be fully GREEN"
    echo -e "    • Model path may be incorrect or not MLX-compatible"
    echo ""
    echo -e "  ${BLUE}Try:${NC} Re-run setup, ensure toggle is GREEN before continuing,"
    echo -e "  and that the model downloaded successfully (look for '✓ Download Complete!')."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"




IS_MULTIMODAL=0
if [[ "$MODEL_TYPE" == "multimodal" ]]; then
    IS_MULTIMODAL=1
    echo ""
    echo -e "${YELLOW}⚠  WARNING: This model loaded as a MULTIMODAL adapter.${NC}"
    echo -e "${YELLOW}   Continuous batching for multimodal models is coming soon to${NC}"
    echo -e "${YELLOW}   Bodega Inference Engine — it is NOT yet enabled for vision models.${NC}"
    echo ""
    echo -e "${GREEN}   What you CAN do right now:${NC}"
    echo -e "     ${GREEN}✓${NC} Use the Interactive Chat Shell to have a full conversation with it"
    echo -e "     ${YELLOW}⚠${NC} Benchmarks will run in SEQUENTIAL mode (max 3 concurrent requests)"
    echo ""
fi

echo "Would you like to run a benchmark now to test performance?"
if [[ "$IS_MULTIMODAL" == "1" ]]; then
    echo "1) Advanced Benchmark (Throughput Sweep, Sequential mode)"
    echo "2) Compare Engines (LM Studio vs Bodega CB)"
else
    echo "1) Advanced Benchmark (Continuous Batching Config Sweep)"
    echo "2) Compare Engines (LM Studio vs Bodega CB)"
fi
echo "3) No, just let me use the Interactive Chat Shell!"
echo "4) Skip"
read -p "Select an option [1-4]: " run_bench

# Export telemetry preference so benchmark scripts can respect it
export BODEGA_SKIP_TELEMETRY=$SKIP_TELEMETRY

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results

LAST_JSON=""
if [[ "$run_bench" == "1" ]]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ⚠  BEFORE BENCHMARK:${NC}"
    echo -e "  Please close other apps and IDEs so the benchmark can have"
    echo -e "  true headroom and full SoC cores utilized for accurate results."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "  Press Enter when ready to start..."
    echo -e "\n${BLUE}Running CB Sweep — results will open in browser when done...${NC}"
    if [[ "$IS_MULTIMODAL" == "1" ]]; then
        python sweep_cb_configs.py --model "$TARGET_MODEL" --multimodal-sequential \
            --output "results/sweep_${TIMESTAMP}.json"
    else
        python sweep_cb_configs.py --model "$TARGET_MODEL" \
            --output "results/sweep_${TIMESTAMP}.json"
    fi
    LAST_JSON="results/sweep_${TIMESTAMP}.json"
elif [[ "$run_bench" == "2" ]]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ⚠  BEFORE BENCHMARK:${NC}"
    echo -e "  Please close other apps and IDEs so the benchmark can have"
    echo -e "  true headroom and full SoC cores utilized for accurate results."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "  Press Enter when ready to start..."
    echo -e "\n${BLUE}Running compare_engines.py — results will open in browser when done...${NC}"
    LMSTUDIO_ID="${TARGET_MODEL##*/}"
    python compare_engines.py --model "$TARGET_MODEL" \
        --lmstudio-model-id "$LMSTUDIO_ID" \
        --output "results/compare_${TIMESTAMP}.json" \
        --leaderboard-url "https://leaderboard.srswti.com"
    LAST_JSON="results/compare_${TIMESTAMP}.json"
elif [[ "$run_bench" == "3" ]]; then
    echo -e "\n${BLUE}Launching Interactive Shell...${NC}"
    python interactive_shell.py
else
    echo -e "\nYou can run benchmarks and interact with models anytime:"
    echo -e "  ${YELLOW}python sweep_cb_configs.py --model $TARGET_MODEL${NC}  (CB config sweep)"
    echo -e "  ${YELLOW}python compare_engines.py --model $TARGET_MODEL${NC}  (LM Studio vs Bodega)"
    echo -e "  ${YELLOW}python interactive_shell.py${NC}  (live chat and visuals)"
fi

# ─── Leaderboard prompt (after CB sweep only; compare auto-uploads) ───────────
if [[ -n "$LAST_JSON" && -f "$LAST_JSON" && "$run_bench" == "1" ]]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  🏆 Share your results with the community!${NC}"
    echo ""
    echo -e "  Would you like to post your results to the global leaderboard?"
    echo -e "  ${YELLOW}Only your chip, RAM, and best system TPS are shared${NC} — nothing"
    echo -e "  personal. It helps everyone see what's possible on different hardware."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "  Post to leaderboard? [Y/n]: " post_leaderboard
    if [[ "$post_leaderboard" != "n" && "$post_leaderboard" != "N" ]]; then
        echo -e "\n${BLUE}Uploading to leaderboard.srswti.com...${NC}"
        python show_results.py "$LAST_JSON" --upload "https://leaderboard.srswti.com"
    fi
fi

