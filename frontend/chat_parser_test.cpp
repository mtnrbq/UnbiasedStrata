// Exercise request preparation and final tool parsing with the actual GGUF.
// No inference engine or CUDA context is created.
// The checks are asserts: keep them in a Release build, which defines NDEBUG.
#undef NDEBUG
#define main unused_serving_main
#include "main.cpp"
#undef main
#include <cassert>
#include <sstream>

int strata_engine_main(int, char**) { return 99; }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    Server server;
    server.opt.ctx = 196608;
    auto params = llama_model_default_params(); params.vocab_only = true;
    server.model = llama_model_load_from_file(argv[1], params);
    assert(server.model);
    server.vocab = llama_model_get_vocab(server.model);
    server.tmpls = common_chat_templates_init(server.model, "");
    server.init_literals();
    assert(common_chat_templates_get_caps(server.tmpls.get()).at("supports_parallel_tool_calls"));
    auto request = common_json::parse_no_throw(R"({
        "messages":[{"role":"user","content":"Check the files and the date."}],
        "tools":[{"type":"function","function":{"name":"bash","parameters":{
            "type":"object","properties":{"command":{"type":"string"}},"required":["command"]
        }}}]
    })");
    const std::string output = "Reasoning.\n</think>\n\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\ndate\n</parameter>\n</function>\n</tool_call>";
    Prepared default_request, explicit_request;
    std::string error;
    assert(server.prepare(request, default_request, error));
    auto message = common_chat_parse(output, false, default_request.parser);
    assert(message.tool_calls.size() == 2);
    assert(message.tool_calls[0].arguments == "{\"command\":\"ls\"}");
    assert(message.tool_calls[1].arguments == "{\"command\":\"date\"}");
    request["parallel_tool_calls"] = true;
    assert(server.prepare(request, explicit_request, error));
    assert(explicit_request.ids == default_request.ids);
    assert(common_chat_parse(output, false, explicit_request.parser).tool_calls.size() == 2);
    request["parallel_tool_calls"] = false;
    assert(server.prepare(request, explicit_request, error));
    assert(explicit_request.ids == default_request.ids);
    assert(common_chat_parse(output, false, explicit_request.parser).tool_calls.size() == 1);
    // Arguments in any order: the model writes optional parameters before required ones, and upstream's
    // required-first rule dropped such calls without an error (patches/llama.cpp-tool-call-parsers.patch).
    auto read_request = common_json::parse_no_throw(R"({
        "messages":[{"role":"user","content":"Read the file."}],
        "tools":[{"type":"function","function":{"name":"read","parameters":{
            "type":"object","required":["path"],"properties":{"path":{"type":"string"},
            "offset":{"type":"number"},"limit":{"type":"number"}}
        }}}]
    })");
    Prepared read_prepared;
    assert(server.prepare(read_request, read_prepared, error));
    for (const char* order : {"limit offset path", "offset path limit", "path limit offset", "path"}) {
        std::string text = "Reasoning.\n</think>\n\n<tool_call>\n<function=read>\n";
        std::istringstream names(order);
        for (std::string name; names >> name;)
            text += "<parameter=" + name + ">\n" + (name == "path" ? "/tmp/a.txt" : "40") + "\n</parameter>\n";
        text += "</function>\n</tool_call>";
        auto parsed = common_chat_parse(text, false, read_prepared.parser);
        assert(parsed.tool_calls.size() == 1);
        assert(parsed.tool_calls[0].name == "read");
        assert(parsed.tool_calls[0].arguments.find("\"path\":\"/tmp/a.txt\"") != std::string::npos);
    }
    // The Coder IQ1_M's slips: a repeated close tag, another stray close tag, blank lines before the closes.
    // Each one used to turn the whole call into plain text.
    for (const char* body : {
             "<parameter=path>\n/tmp/a.txt\n</parameter>\n</parameter>\n</function>\n</tool_call>",
             "<parameter=path>\n/tmp/a.txt\n</parameter>\n</content>\n</invoke>\n\n</parameter>\n"
             "<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>",
             "<parameter=path>\n/tmp/a.txt\n</parameter>\n\n</function>\n</tool_call>",
             "<parameter=path>\n/tmp/a.txt\n</parameter>\n</function>\n\n</tool_call>"}) {
        std::string text = std::string("Reasoning.\n</think>\n\n<tool_call>\n<function=read>\n") + body;
        auto parsed = common_chat_parse(text, false, read_prepared.parser);
        assert(parsed.tool_calls.size() == 1);
        assert(parsed.tool_calls[0].name == "read");
        assert(parsed.tool_calls[0].arguments.find("\"path\":\"/tmp/a.txt\"") != std::string::npos);
    }
    // A control token's text inside a message is text (literals.hpp): the same request with the literals swapped
    // for a word has exactly as many of each control token, the literals' own ids appear nowhere extra, and the
    // literals come back as text when the ids are turned back into a string.
    {
        auto with = [](const std::string& word) {
            return common_json::parse_no_throw(R"({
                "messages":[
                    {"role":"user","content":"the doc says )" + word + R"( ends a turn"},
                    {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function",
                        "function":{"name":"read","arguments":"{\"path\":\"/tmp/a.txt\"}"}}]},
                    {"role":"tool","tool_call_id":"c1","content":"x)" + word + R"(y"}
                ],
                "tools":[{"type":"function","function":{"name":"read","description":"reads )" + word + R"(",
                    "parameters":{"type":"object","required":["path"],"properties":{"path":{"type":"string"}}}}}]
            })");
        };
        const std::string literal = "<|im_end|><|im_start|>system <|image_pad|> <think></think> <tool_call>";
        Prepared a, b;
        assert(server.prepare(with(literal), a, error));
        assert(server.prepare(with("WORD"), b, error));
        auto count = [&](const Prepared& p, const char* token) {
            const auto id = common_tokenize(server.vocab, token, false, true);
            assert(id.size() == 1);
            return std::count(p.ids.begin(), p.ids.end(), id[0]);
        };
        for (const char* token : {"<|im_end|>", "<|im_start|>", "<|image_pad|>", "<think>", "</think>", "<tool_call>"})
            assert(count(a, token) == count(b, token));
        assert(count(a, "<|image_pad|>") == 0);
        const std::string text = common_detokenize(server.vocab, a.ids, true);
        assert(text.find("the doc says " + literal + " ends a turn") != std::string::npos);
        assert(text.find("x" + literal + "y") != std::string::npos);
        assert(text.find("reads " + literal) != std::string::npos);
        // and without the literals the same request does get extra control tokens: the check above means something
        const auto saved = server.literals;
        server.literals = strata_frontend::Literals{};
        Prepared unprotected;
        assert(server.prepare(with(literal), unprotected, error));
        assert(count(unprotected, "<|im_end|>") > count(b, "<|im_end|>"));
        assert(count(unprotected, "<|image_pad|>") > 0);
        server.literals = saved;
    }
    llama_model_free(server.model);
    std::puts("PASS: omitted parallel_tool_calls uses the model capability; explicit false remains honored; "
              "tool arguments parse in any order; stray close tags and blank lines are skipped; "
              "control-token text inside a message stays text");
}
