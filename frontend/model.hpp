// frontend/model.hpp - the models this front end serves, and which one a GGUF is.
//
// Two releases of Qwen3.8-Flash-Next, told apart by the GGUF itself: the Coder keeps 256 of the 512 experts and
// comes in one size (IQ1_M); the full model is served at IQ3_XXS only, which only its file name says.  Anything
// else is refused, as before: the engine runs other packs, but nothing here was checked against them.
#pragma once
#include <cstdint>
#include <string>

namespace strata_frontend {

struct ModelInfo {
    const char* id;            // the name the API reports
    const char* about;
    double experts_gb;         // upstream's setup.py MODELS arena_gb: the experts the engine holds in RAM
    const char* spec_min_p;    // the draft probability a speculative window needs
    const char* const* engine_args;   // more engine arguments, nullptr-terminated
    const char* const* engine_env;    // NAME=value for the engine, unless already set; nullptr-terminated
};

// IQ3_XXS: the values this fork was measured with.  Coder IQ1_M: upstream's tested config on an RTX 5070 Ti, where
// the expert cache that adapts every round, slowly forgetting, gave 84.0 tok/s against the engine defaults' 72.6
// (8 prompts of 400 tokens, 32K context); --spec-min-p 0.5 measured the same as 0.70 (83.3).
// STRATA_ADAPT_OVERLAP=1: the adaptive tier's copies land while the next decode window runs (same tokens).  On since
// 0.1.43 on Linux CUDA, opt-in on Windows, where an RTX 5070 measured -2.6% on Q2_0; for the Coder on the 5070 Ti,
// engine 0.1.43 at 64K, it gave 113.3 tok/s against 110.9 (+2.2%, 5 alternating pairs of 8 prompts x 400 tokens).
inline constexpr const char* kNone[] = {nullptr};
inline constexpr const char* kCoderArgs[] = {"--adapt-every", "1", "--adapt-swaps", "80", "--adapt-decay", "0.97", nullptr};
inline constexpr const char* kCoderEnv[] = {"STRATA_ADAPT_OVERLAP=1", nullptr};
inline constexpr ModelInfo kFlashNextIq3Xxs{"qwen3.8-flash-next-iq3_xxs", "Qwen3.8-Flash-Next IQ3_XXS", 42.9, "0.5",
                                            kNone, kNone};
inline constexpr ModelInfo kCoderIq1M{"qwen3.8-flash-next-coder-iq1_m", "Qwen3.8-Flash-Next Coder IQ1_M", 23.4, "0.70",
                                      kCoderArgs, kCoderEnv};

// `name` is general.name, `experts` qwen4exp.expert_count, `file` shard 1's path.  nullptr: not a served model.
inline const ModelInfo* identify_model(const std::string& name, int64_t experts, const std::string& file) {
    if (experts == 256 && name.find("Coder") != std::string::npos) return &kCoderIq1M;
    if (experts == 512 && name.find("Coder") == std::string::npos && file.find("-IQ3_XXS-") != std::string::npos)
        return &kFlashNextIq3Xxs;
    return nullptr;
}

}  // namespace strata_frontend
