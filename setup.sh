#!/bin/bash
# Bodega Inference Engine Setup Script

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Bodega Inference Engine Setup ===${NC}"
echo ""
echo -e "${YELLOW}Step 1: Installing Bodega Sensors & Inference Engine${NC}"
read -p "Do you already have Bodega Sensors installed? [y/N]: " has_sensors
echo ""

if [[ "$has_sensors" == "y" || "$has_sensors" == "Y" ]]; then
    echo -e "${GREEN}Skipping application download/install...${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}ACTION REQUIRED TO PROCEED:${NC}"
    echo -e "1. Open ${GREEN}Bodega Sensors${NC} from your Applications folder."
    echo -e "2. Find the ${YELLOW}Bodega Inference Engine${NC} toggle and turn it ON."
    echo -e "3. Wait for the toggle to turn ${GREEN}GREEN${NC}."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once you have turned the toggle GREEN..."
else
    echo -e "This will download and install the Bodega Sensors app, which contains the Inference Engine."
    echo ""

    # Run the local installation script
    if [ -f "./install_sensors.sh" ]; then
        bash ./install_sensors.sh
    else
        echo -e "${RED}Error: install_sensors.sh not found in the current directory.${NC}"
        echo -e "Please ensure you are running this from the correct folder."
        exit 1
    fi

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}ACTION REQUIRED TO PROCEED:${NC}"
    echo -e "1. Open the downloaded Bodega Sensors.dmg file from your current folder."
    echo -e "2. Drag and drop ${GREEN}Bodega Sensors${NC} into your Applications folder."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once you have copied it to Applications..."

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "3. Double-click ${GREEN}Bodega Sensors${NC} in your Applications folder to open it."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once the app is open..."

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "4. Find the ${YELLOW}Bodega Inference Engine${NC} toggle and click to turn it ON."
    echo -e "5. Wait for the toggle to turn ${GREEN}GREEN${NC}."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once you have turned the toggle GREEN..."
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
echo -e "${YELLOW}Step 2: Model Selection${NC}"
echo "Which model(s) would you like to download?"
echo "1) Bodega ORION 0.6B (srswti/bodega-orion-0.6b) - Ultra-fast, great for continuous batching tests"
echo "2) Bodega Raptor 8B (srswti/bodega-raptor-8b-mxfp4) - Powerful and small parameter model"
echo "3) Both Models"
echo "4) Custom Model Repository from HuggingFace"
echo "5) Skip Model Download"
read -p "Select an option [1-5]: " model_choice

MODELS=()
if [[ "$model_choice" == "1" || "$model_choice" == "3" ]]; then
    MODELS+=("srswti/bodega-orion-0.6b")
fi
if [[ "$model_choice" == "2" || "$model_choice" == "3" ]]; then
    MODELS+=("srswti/bodega-raptor-8b-mxfp4")
fi
if [[ "$model_choice" == "4" ]]; then
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

if [[ "$model_choice" == "5" ]]; then
    echo -e "\n${GREEN}Setup complete!${NC}"
    echo ""
    echo "Would you like to run a benchmark now?"
    echo "1) Advanced Benchmark (Continuous Batching Config Sweep)"
    echo "2) Compare Engines (LM Studio vs Bodega CB)"
    echo "3) No, exit setup"
    read -p "Select an option [1-3]: " run_bench

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p results
    SKIP_LAST_JSON=""

    if [[ "$run_bench" == "1" ]]; then
        echo -e "\n${BLUE}Running CB Configuration Sweep — results will open in browser when done...${NC}"
        python sweep_cb_configs.py --model "$TARGET_MODEL" --output "results/sweep_${TIMESTAMP}.json"
        SKIP_LAST_JSON="results/sweep_${TIMESTAMP}.json"
    elif [[ "$run_bench" == "2" ]]; then
        echo -e "\n${BLUE}Running compare_engines.py — results will open in browser when done...${NC}"
        LMSTUDIO_ID="${TARGET_MODEL##*/}"
        python compare_engines.py --model "$TARGET_MODEL" \
            --lmstudio-model-id "$LMSTUDIO_ID" \
            --output "results/compare_${TIMESTAMP}.json" \
            --leaderboard-url "https://leaderboard.srswti.com"
        SKIP_LAST_JSON="results/compare_${TIMESTAMP}.json"
    else
        echo -e "\nYou can run benchmarks anytime:"
        echo -e "  ${YELLOW}python sweep_cb_configs.py --model $TARGET_MODEL${NC}  (CB config sweep)"
        echo -e "  ${YELLOW}python compare_engines.py --model $TARGET_MODEL${NC}  (LM Studio vs Bodega)"
    fi

    if [[ -n "$SKIP_LAST_JSON" && -f "$SKIP_LAST_JSON" && "$run_bench" == "1" ]]; then
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
            python show_results.py "$SKIP_LAST_JSON" --upload "https://leaderboard.srswti.com"
        fi
    fi
    exit 0
fi

echo -e "\n${YELLOW}Connecting to Bodega Inference Engine on localhost:44468...${NC}"

# Wait until health check passes
while ! curl -s http://localhost:44468/health >/dev/null; do
    echo -e "${RED}Waiting for localhost:44468. Please ensure the toggle is GREEN in Bodega Sensors!${NC}"
    sleep 3
done

echo -e "${GREEN}✓ Connected to Engine! Starting downloads...${NC}"

for model in "${MODELS[@]}"; do
    echo -e "\n${BLUE}Downloading $model...${NC}"
    python3 -c "
import sys, json, httpx

url = 'http://localhost:44468/v1/admin/download-model-stream'
try:
    with httpx.stream('POST', url, json={'model_path': '$model'}, timeout=None) as r:
        if r.status_code != 200:
            print(f'\033[0;31mError {r.status_code} - Is the engine running?\033[0m')
            sys.exit(1)
            
        for line in r.iter_lines():
            if line.startswith('data: '):
                dstr = line[6:]
                if dstr == '[DONE]':
                    print('\n\033[0;32m✓ Download Complete!\033[0m')
                    break
                try:
                    data = json.loads(dstr)
                    if 'message' in data:
                        prog = data.get('progress', 0)
                        sys.stdout.write(f'\r\033[K[Progress: {prog:>3}%] ' + data['message'][:60])
                        sys.stdout.flush()
                except Exception:
                    pass
except Exception as e:
    print(f'\n\033[0;31mError downloading: {e}\033[0m')
"
done

echo -e "\n${GREEN}=== Setup Complete! ===${NC}"
echo ""

TARGET_MODEL=${MODELS[0]}

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

# Run it and capture only the KEY=VALUE lines into shell vars
eval "$(python3 /tmp/bodega_model_inspect.py "$TARGET_MODEL" "$BODEGA_TESTS_DIR" 2>&1 | grep -E '^(MODEL_TYPE|RSS_MB|METAL_MB|TOTAL_MB|PID_VAL)=')"


echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}Model Loaded:${NC}   $TARGET_MODEL"
echo -e "  ${GREEN}Adapter Type:${NC}   $MODEL_TYPE"
echo -e "  ${GREEN}RAM (RSS):${NC}      ${RSS_MB} MB"
echo -e "  ${GREEN}Metal Peak:${NC}     ${METAL_MB} MB  (Total: ${TOTAL_MB} MB)"
echo -e "  ${GREEN}PID:${NC}            ${PID_VAL}"
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

