# Windows shortcuts for this machine (Linux uses the Makefile).  `just --list` shows the recipes.
# The paths below are this PC's; override one on the command line, e.g. `just data='E:\models' serve`.

set windows-shell := ["pwsh", "-NoLogo", "-NoProfile", "-Command"]

# where the Coder's files are (see README.md, Windows)
data := 'D:\Strata-data'
model := data + '\models\coder-IQ1_M\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf'
pack := data + '\packs\coder-iq1_m'
mtp := data + '\mtp\rt'
profile := 'D:\Strata\data\expert-profile-coder.bin'
ctx := '65536'

# build the server for the RTX 50 series only (sm_120): a third of the CUDA work of the default three architectures
build *args:
    .\build.ps1 -CudaArchs 120 {{args}}

# run the server for Qwen3.8 Flash-Next Coder IQ1_M on 127.0.0.1:8080 (restarted if the engine fails; Ctrl+C stops it)
serve *args:
    .\serve.ps1 --model '{{model}}' --pack '{{pack}}' --mtp '{{mtp}}' --expert-profile '{{profile}}' --ctx {{ctx}} {{args}}
