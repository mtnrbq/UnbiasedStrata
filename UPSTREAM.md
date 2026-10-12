# Relationship to upstream

## Sources

| What | Where | Version |
| --- | --- | --- |
| Engine | https://github.com/Niko1221/Strata | tag `v0.1.42`, commit `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`, engine 0.1.42 |
| ggml (CPU backend, CUDA matrix kernels) | https://github.com/ggml-org/llama.cpp, as the submodule `third_party/llama.cpp` | commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the one upstream Strata pins |

The first commit of this repository is the upstream snapshot as published, engine 0.1.39 (commit `6f32ec0`; upstream
has since rewritten its history, and its tag `v0.1.39`, commit `a1641e9`, has the same tree). One file was not
imported: `bench/results/2026-09-30-community-rtx-5090/engine.log`, which upstream's own ignore rules exclude.

The engine was then moved to 0.1.42 by applying upstream's `git diff v0.1.39 v0.1.42 -- src include cmake
CMakeLists.txt` (942 upstream commits; 220 files) and merging this fork's changes back in. Nothing else from that
range was taken: not the Python server, setup or tools, the benchmarks, the tests or the Intel port, and none of
the experimental ports beyond the tag (robertkarl/UnbiasedStrata#2 was used as a reference only).
To see only this fork's engine changes, compare `src/`, `include/` and `cmake/` with upstream's tag `v0.1.42`.

## What this fork changed

- **Build:** ggml is taken from the submodule. Upstream cloned llama.cpp during the build. `make build` on Linux,
  `build.ps1` on Windows. In `CMakeLists.txt`: the tests that live in upstream's `tests/` (not imported) are built
  only when it exists; llama.cpp is built as C++17 under MSVC; a `chat-parser-test` target; a manifest that makes
  the Windows program's code page UTF-8.
- **Removed:** the Python installer and launchers, the Python server and web app, the Python tools, the
  Intel SYCL port, Docker files, benchmark results, the test suite, translations, and all docs except
  `docs/upstream/DETAILS.md` and the paper.
- **Added:** `frontend/`, the C++ front end (tokenizer, chat template, output parser and HTTP API, using
  llama.cpp's code from the submodule), and a build target that links it with the engine into one binary.
- **ggml / llama.cpp common (`third_party/llama.cpp`):** one change, kept as
  `patches/llama.cpp-qwen3-coder-any-argument-order.patch` and applied in the submodule's working tree:
  the Qwen3-Coder tool-call grammar accepts a call's arguments in any order. Upstream requires the
  required arguments first; Qwen3.8 Flash-Next often writes the optional ones first (`limit`, `offset`,
  `path`), and because the grammar's tool-call group is optional and has no end anchor, such a call was
  dropped silently: the client got `finish_reason=stop` and an empty answer (2026-10-08).
- **Engine source (`src/`, `include/`):** four changes, three for loading from a slow or remote store. The
  engine's own program is compiled into the binary with its entry point renamed; the front end starts it
  as a child process.
  - `include/strata/platform/robust_read.hpp` (new): positional reads that retry an error a network store
    gives for a moment, and lower the number of parallel readers while it does.
  - `src/core/expert_source.cpp`: the expert loader reads through it, reports progress, and names the file,
    offset and error when it fails. On Windows, where `robust_read.hpp` is not built, it keeps upstream's
    `fread` (#230), with the progress and the named errors.
  - `src/kernels/ngram.cpp`: on Linux, `--ple-io ram` reads the n-gram table into memory of its own with 16
    readers, where upstream locks the file mapping (which fails under a container's `ulimit -l`). On Windows
    upstream's own path is kept: it locks the mapping with `VirtualLock` (new in 0.1.42).
  - `src/program/generate.cpp`: the stall report names the stage the decode loop was in, and lists the
    threads' wait channels on Linux.

Code paths for other models and for AMD cards are still present in the engine source. They are not built by
`make build` or `build.ps1`, not tested and not supported here. The engine's Windows paths are upstream's and are
used; the retrying reads above are Linux only. In `CMakeLists.txt`, the tests and the ISA checker that live in
upstream's `tests/` and `tools/` are built only when those are present.

## Prepared model files

The pack and the draft head are produced once with upstream Strata's tools at the commit above. These are
the commands that were used, with `U` an upstream checkout, `G` this repository's
`third_party/llama.cpp/gguf-py`, and Python 3 with `numpy` and `regex`:

```
export STRATA_GGUF_PY=$G
# the pack, from the model's first shard (the second must be beside it)
python $U/tools/iq_pack.py --gguf Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out pack-iq3-xxs
cp $U/data/expert-profile.bin pack-iq3-xxs/

# the draft head, from the original checkpoint's 31 draft-head tensors (about 5 GB, fetched over HTTP)
python $U/tools/mtp_fetch.py fetch  --out mtp-bf16
python $U/tools/mtp_fetch.py verify --out mtp-bf16
python $U/tools/mtp_pack.py --src mtp-bf16 --experts q2_0 --out mtp-bf16/mtp-q2_0.gguf
python $U/tools/mtp_rt.py --gguf mtp-bf16/mtp-q2_0.gguf --out mtp
cp $U/data/draft_vocab.bin mtp/
```

The model shards used were revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` of
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, verified against their published SHA-256 checksums. The
prepared files have not been published yet. Building them twice gave the same bytes; their SHA-256:

```
68beec784ea64aeecace591b33debb8ba331a9ad257a3e1b68c5c701dd22b267  pack-iq3-xxs/conversions.json
d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13  pack-iq3-xxs/dense.bin
8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf  pack-iq3-xxs/expert-profile.bin
1afe12b36cd2d13bf122b741fa0e81e1f751319cc3c451e54ba75524d825f148  pack-iq3-xxs/index.txt
ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee  pack-iq3-xxs/native_experts.txt
c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a  mtp/dense.bin
991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd  mtp/dense.txt
b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91  mtp/draft_vocab.bin
09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f  mtp/experts.bin
```
