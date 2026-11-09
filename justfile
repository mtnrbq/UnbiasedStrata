# Windows shortcuts for this machine (Linux uses the Makefile).  `just --list` shows the recipes.
# The paths below are this PC's; override one on the command line, e.g. `just data='E:\models' serve`.

set windows-shell := ["pwsh", "-NoLogo", "-NoProfile", "-Command"]
set default-list := true

# where the Coder's files are (see README.md, Windows)
data := 'D:\Strata-data'
model := data + '\models\coder-IQ1_M\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf'
pack := data + '\packs\coder-iq1_m'
mtp := data + '\mtp\rt'
profile := 'D:\Strata\data\expert-profile-coder.bin'
ctx := '65536'

# where the full model's files are (C: has the room; it is the same SSD as D:).  The pack holds its own expert profile
iq3_data := 'C:\Strata-data'
iq3_model := iq3_data + '\models\IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'
iq3_pack := iq3_data + '\packs\iq3-xxs'

# Qwen3.6-35B-A3B UD-Q6_K_XL (Unsloth, with the MTP block): one GGUF, no pack or draft head; the picture reader beside it
q36_model := iq3_data + '\models\qwen36\Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf'
q36_mmproj := iq3_data + '\models\qwen36\mmproj-F16.gguf'

# build the server for the RTX 50 series only (sm_120): a third of the CUDA work of the default three architectures
build *args:
    .\build.ps1 -CudaArchs 120 {{args}}

# run the server for Qwen3.8 Flash-Next Coder IQ1_M on 0.0.0.0:8080 (reachable from the network) (restarted if the engine fails; Ctrl+C stops it)
serve *args:
    .\serve.ps1 --host 0.0.0.0 --model '{{model}}' --pack '{{pack}}' --mtp '{{mtp}}' --expert-profile '{{profile}}' --ctx {{ctx}} {{args}}

# run the server for Qwen3.8 Flash-Next IQ3_XXS (the full model) the same way; it shares the Coder's draft head
serve-iq3 *args:
    .\serve.ps1 --host 0.0.0.0 --model '{{iq3_model}}' --pack '{{iq3_pack}}' --mtp '{{mtp}}' --ctx {{ctx}} {{args}}

# run the server for Qwen3.6-35B-A3B UD-Q6_K_XL the same way (its own engine path; images: add --mmproj '{{q36_mmproj}}')
serve-qwen36 *args:
    .\serve.ps1 --host 0.0.0.0 --model '{{q36_model}}' --ctx {{ctx}} {{args}}
