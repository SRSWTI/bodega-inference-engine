#!/bin/bash
# Bodega Inference Engine Setup Script

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Bodega Inference Engine Setup ===${NC}"
echo ""
echo -e "${YELLOW}Step 1: Installing BodegaOS Sensors & Inference Engine${NC}"
read -p "Do you already have BodegaOS Sensors installed? [y/N]: " has_sensors
echo ""

if [[ "$has_sensors" == "y" || "$has_sensors" == "Y" ]]; then
    echo -e "${GREEN}Skipping application download/install...${NC}"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}ACTION REQUIRED TO PROCEED:${NC}"
    echo -e "1. Open ${GREEN}BodegaOS Sensors${NC} from your Applications folder."
    echo -e "2. Find the ${YELLOW}Bodega Inference Engine${NC} toggle and turn it ON."
    echo -e "3. Click ${GREEN}Yes${NC} to proceed."
    echo -e "4. Wait for the toggle to turn ${GREEN}GREEN${NC}."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once you have turned the toggle GREEN..."
else
    echo -e "This will download and install the BodegaOS Sensors app, which contains the Inference Engine."
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
    echo -e "1. Open the downloaded .dmg file from your current folder."
    echo -e "2. Drag and drop ${GREEN}BodegaOS Sensors${NC} into your Applications folder."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once you have copied it to Applications..."

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "3. Double-click ${GREEN}BodegaOS Sensors${NC} in your Applications folder to open it."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once the app is open..."

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "4. Find the ${YELLOW}Bodega Inference Engine${NC} toggle and click to turn it ON."
    echo -e "5. A prompt will appear — click ${GREEN}Yes${NC} to proceed."
    echo -e "6. Wait for the toggle to turn ${GREEN}GREEN${NC}."
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    read -p "Press Enter once you have turned the toggle GREEN..."
fi

echo ""
echo -e "${YELLOW}Step 1.b: Installing Apple Silicon Telemetry Tools${NC}"
if ! command -v mactop &> /dev/null; then
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}Homebrew not found. Installing Homebrew automatically...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to path for the rest of the script (common locations)
        eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || eval "$(/usr/local/bin/brew shellenv 2>/dev/null)"
    fi
    
    if command -v brew &> /dev/null; then
        echo -e "${BLUE}Installing mactop (Real-time Apple Silicon monitor)...${NC}"
        brew install mactop
    else
        echo -e "${RED}Homebrew could not be installed. Skipping mactop installation.${NC}"
    fi
else
    echo -e "${GREEN}mactop is already installed.${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Model Selection${NC}"
echo "Which model(s) would you like to download?"
echo "1) Bodega Raptor 90M (srswti/bodega-raptor-90m) - Ultra-fast, great for continuous batching tests"
echo "2) Bodega Raptor 8B (srswti/bodega-raptor-8b-mxfp4) - Powerful and small parameter model"
echo "3) Both Models"
echo "4) Custom Model Repository from HuggingFace"
echo "5) Skip Model Download"
read -p "Select an option [1-5]: " model_choice

MODELS=()
if [[ "$model_choice" == "1" || "$model_choice" == "3" ]]; then
    MODELS+=("srswti/bodega-raptor-90m")
fi
if [[ "$model_choice" == "2" || "$model_choice" == "3" ]]; then
    MODELS+=("srswti/bodega-raptor-8b-mxfp4")
fi
if [[ "$model_choice" == "4" ]]; then
    read -p "Enter HuggingFace model path (e.g. mlx-community/Qwen3.5-0.5B-Instruct-4bit): " custom_model
    if [ -n "$custom_model" ]; then
        MODELS+=("$custom_model")
    fi
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    TARGET_MODEL="srswti/bodega-raptor-90m"
else
    TARGET_MODEL=${MODELS[0]}
fi

if [[ "$model_choice" == "5" ]]; then
    echo -e "\n${GREEN}Setup complete!${NC}"
    echo ""
    echo "Would you like to run a benchmark now?"
    echo "1) Basic Benchmark (HTTP Concurrency Load Test)"
    echo "2) Advanced Benchmark (Continuous Batching Config Sweep)"
    echo "3) No, exit setup"
    read -p "Select an option [1-3]: " run_bench
    
    if [[ "$run_bench" == "1" ]]; then
        echo -e "\n${BLUE}Running benchmark_http_concurrency.py...${NC}"
        python benchmark_http_concurrency.py --model "$TARGET_MODEL"
    elif [[ "$run_bench" == "2" ]]; then
        echo -e "\n${BLUE}Executing sweep_cb_configs.py...${NC}"
        python sweep_cb_configs.py --model "$TARGET_MODEL"
    else
        echo -e "\nYou can run the tests anytime with: ${YELLOW}python benchmark_http_concurrency.py --model $TARGET_MODEL${NC} or ${YELLOW}python sweep_cb_configs.py --model $TARGET_MODEL${NC}"
    fi
    exit 0
fi

echo -e "\n${YELLOW}Connecting to Bodega Inference Engine on localhost:44468...${NC}"

# Wait until health check passes
while ! curl -s http://localhost:44468/health >/dev/null; do
    echo -e "${RED}Waiting for localhost:44468. Please ensure the toggle is GREEN in BodegaOS Sensors!${NC}"
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

echo "Would you like to run a benchmark now to test performance?"
echo "1) Basic Benchmark (HTTP Concurrency Load Test)"
echo "2) Advanced Benchmark (Continuous Batching Config Sweep)"
echo "3) No, just let me use the Interactive Chat Shell!"
echo "4) Skip"
read -p "Select an option [1-4]: " run_bench

if [[ "$run_bench" == "1" ]]; then
    echo -e "\n${BLUE}Running benchmark_http_concurrency.py...${NC}"
    python benchmark_http_concurrency.py --model "$TARGET_MODEL"
elif [[ "$run_bench" == "2" ]]; then
    echo -e "\n${BLUE}Executing sweep_cb_configs.py...${NC}"
    python sweep_cb_configs.py --model "$TARGET_MODEL"
elif [[ "$run_bench" == "3" ]]; then
    echo -e "\n${BLUE}Launching Interactive Shell...${NC}"
    python interactive_shell.py
else
    echo -e "\nYou can test continuous batching and interact with models anytime by running:"
    echo -e "  ${YELLOW}python benchmark_http_concurrency.py --model $TARGET_MODEL${NC}  (for basic load testing)"
    echo -e "  ${YELLOW}python sweep_cb_configs.py --model $TARGET_MODEL${NC}  (for config benchmarking)"
    echo -e "  ${YELLOW}python interactive_shell.py${NC}  (for live chat and visuals)"
fi
