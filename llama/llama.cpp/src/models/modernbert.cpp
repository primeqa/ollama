#include "models.h"
#include "../llama-impl.h"
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstring>

// DEBUG: Global storage for intermediate tensors
static std::vector<debug_tensor_info> g_debug_tensors;
static bool g_debug_layers_enabled = false;

// Add CLS token tracking
static bool g_track_cls_token = false;

// Token 20778 debugging
static bool g_debug_token_20778 = false;

// RoPE bypass for debugging
static bool g_disable_rope = false;

// Dump actual tensor values for detailed comparison
static bool g_dump_tensor_values = false;
static FILE* g_tensor_dump_file = nullptr;

// Helper to dump tensor values to file
static void dump_tensor_values(const char* name, ggml_tensor* tensor, int layer) {
    if (!g_dump_tensor_values || !g_tensor_dump_file) return;

    fprintf(g_tensor_dump_file, "--- TENSOR: %s (layer %d) ---\n", name, layer);
    fprintf(g_tensor_dump_file, "Shape: [%lld, %lld, %lld, %lld]\n",
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fprintf(g_tensor_dump_file, "Type: %d\n", tensor->type);
    fflush(g_tensor_dump_file);
}

// Helper to mark tensor for CLS tracking
static void track_cls_if_enabled(ggml_tensor* tensor, const char* label, int layer) {
    if (g_track_cls_token && tensor) {
        ggml_set_output(tensor);
        LLAMA_LOG_INFO("[CLS_TRACK] Layer %d: %s marked for inspection\n", layer, label);
    }
}

llm_build_modernbert::llm_build_modernbert(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    // Check if layer debugging is enabled
    const char* debug_env = std::getenv("OLLAMA_DEBUG_LAYERS");
    g_debug_layers_enabled = (debug_env != nullptr && std::strcmp(debug_env, "1") == 0);

    // Check if CLS token tracking is enabled
    const char* track_cls_env = std::getenv("OLLAMA_TRACK_CLS");
    g_track_cls_token = (track_cls_env != nullptr && std::strcmp(track_cls_env, "1") == 0);

    // Check if token 20778 debugging is enabled
    const char* debug_20778_env = std::getenv("OLLAMA_DEBUG_TOKEN_20778");
    g_debug_token_20778 = (debug_20778_env != nullptr && std::strcmp(debug_20778_env, "1") == 0);

    // Check if RoPE should be disabled for debugging
    const char* disable_rope_env = std::getenv("OLLAMA_DEBUG_DISABLE_ROPE");
    g_disable_rope = (disable_rope_env != nullptr && std::strcmp(disable_rope_env, "1") == 0);

    // Check if tensor value dumping is enabled
    const char* dump_values_env = std::getenv("OLLAMA_DUMP_TENSOR_VALUES");
    g_dump_tensor_values = (dump_values_env != nullptr && std::strcmp(dump_values_env, "1") == 0);
    if (g_dump_tensor_values && !g_tensor_dump_file) {
        const char* dump_file = std::getenv("OLLAMA_TENSOR_DUMP_FILE");
        if (!dump_file) dump_file = "/tmp/ollama_tensor_dump.txt";
        g_tensor_dump_file = fopen(dump_file, "w");
        if (g_tensor_dump_file) {
            LLAMA_LOG_INFO("[DEBUG] Tensor value dumping enabled to: %s\n", dump_file);
        } else {
            LLAMA_LOG_ERROR("[DEBUG] Failed to open tensor dump file: %s\n", dump_file);
            g_dump_tensor_values = false;
        }
    }

    if (g_debug_layers_enabled) {
        g_debug_tensors.clear();
        LLAMA_LOG_INFO("[DEBUG] Layer debugging enabled for ModernBERT\n");
        LLAMA_LOG_INFO("[DEBUG] model.arch = %d, LLM_ARCH_MODERNBERT = %d\n", (int)model.arch, (int)LLM_ARCH_MODERNBERT);
    }

    if (g_track_cls_token) {
        LLAMA_LOG_INFO("[CLS_TRACK] CLS token tracking enabled\n");
    }

    if (g_debug_token_20778) {
        LLAMA_LOG_INFO("[TOKEN_20778] Token 20778 debugging enabled - will log Q/K/V values\n");
    }

    if (g_disable_rope) {
        LLAMA_LOG_WARN("⚠️  [ROPE_BYPASS] RoPE is DISABLED for debugging - embeddings will be incorrect!\n");
    }

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * inp_pos = nullptr;

    if (model.arch != LLM_ARCH_JINA_BERT_V2) {
        inp_pos = build_inp_pos();
    }

    // construct input embeddings (token, type, position)
    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "tok_embd_lookup", -1);

    // CRITICAL FIX: For ModernBERT, mark embedding result as output to prevent buffer reuse
    // The allocator aggressively reuses buffers, causing embedding values to be overwritten
    // before they're used by downstream operations. Marking as OUTPUT extends the liveness window.
    if (model.arch == LLM_ARCH_MODERNBERT) {
        ggml_set_output(inpL);
    }

    // DEBUG: Track token embeddings immediately after lookup
    if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
        g_debug_tensors.push_back({inpL, "tok_embd_only", -1});
        LLAMA_LOG_INFO("[DEBUG] Input tokens: n_tokens=%ld\n", n_tokens);
        LLAMA_LOG_INFO("[DEBUG] tok_embd tensor: %p, shape=[%ld, %ld], type=%d\n",
                       (void*)model.tok_embd, (long)model.tok_embd->ne[0], (long)model.tok_embd->ne[1], (int)model.tok_embd->type);
        LLAMA_LOG_INFO("[DEBUG] inpL (after get_rows): %p, shape=[%ld, %ld], type=%d, op=%d\n",
                       (void*)inpL, (long)inpL->ne[0], (long)inpL->ne[1], (int)inpL->type, (int)inpL->op);
        LLAMA_LOG_INFO("[DEBUG] Marked inpL as output to preserve buffer\n");
    }

    // token types are hardcoded to zero ("Sentence A")
    if (model.type_embd) {
        ggml_tensor * type_row0 = ggml_view_1d(ctx0, model.type_embd, n_embd, 0);
        // WORKAROUND: For ModernBERT, force explicit copy of BOTH operands to avoid memory aliasing
        LLAMA_LOG_INFO("[DEBUG RESIDUAL] Type embedding add: model.arch=%d, LLM_ARCH_MODERNBERT=%d\n", (int)model.arch, (int)LLM_ARCH_MODERNBERT);
        if (model.arch == LLM_ARCH_MODERNBERT) {
            LLAMA_LOG_INFO("[DEBUG RESIDUAL] Using ModernBERT workaround for type embedding add\n");
            ggml_tensor* inpL_copy = ggml_dup_tensor(ctx0, inpL);
            inpL_copy = ggml_cpy(ctx0, inpL, inpL_copy);
            inpL_copy->flags |= GGML_TENSOR_FLAG_OUTPUT;

            ggml_tensor* type_row0_copy = ggml_dup_tensor(ctx0, type_row0);
            type_row0_copy = ggml_cpy(ctx0, type_row0, type_row0_copy);
            type_row0_copy->flags |= GGML_TENSOR_FLAG_OUTPUT;

            inpL = ggml_add(ctx0, inpL_copy, type_row0_copy);
        } else {
            LLAMA_LOG_INFO("[DEBUG RESIDUAL] Using standard add for type embedding\n");
            inpL = ggml_add(ctx0, inpL, type_row0);
        }
    }
    if (model.arch == LLM_ARCH_BERT) {
        inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
    }
    cb(inpL, "inp_embd", -1);

    // CRITICAL FIX: Protect embeddings before norm for ModernBERT
    if (model.arch == LLM_ARCH_MODERNBERT) {
        ggml_set_output(inpL);
    }

    // DEBUG: Track embeddings BEFORE norm
    if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
        g_debug_tensors.push_back({inpL, "embeddings_pre_norm", -1});
    }

    // embed layer norm
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, -1);
    cb(inpL, "inp_norm", -1);

    // CRITICAL FIX: Protect normalized embeddings for ModernBERT
    if (model.arch == LLM_ARCH_MODERNBERT) {
        ggml_set_output(inpL);
    }

    // DEBUG: Track embeddings AFTER norm
    if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
        g_debug_tensors.push_back({inpL, "embeddings_post_norm", -1});
        LLAMA_LOG_INFO("[DEBUG] Embeddings tensors added to debug list\n");
    }

    auto * inp_attn = build_attn_inp_no_cache();

    // TEMPORARY: Disable inp_out_ids to test if it's causing layer skipping
    // ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * inp_out_ids = nullptr;

    // DEBUG: Check if inp_out_ids is set
    if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
        if (inp_out_ids) {
            LLAMA_LOG_INFO("[DEBUG] inp_out_ids is SET (will apply get_rows to last layer)\n");
        } else {
            LLAMA_LOG_INFO("[DEBUG] inp_out_ids is NULL (no special last layer handling)\n");
        }
    }

    // ModernBERT: Check if we need alternating attention pattern
    const bool use_alternating_attn = (model.arch == LLM_ARCH_MODERNBERT &&
                                       hparams.global_attn_every_n_layers > 0);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur = inpL;

        // CLS TRACK: Mark layer input
        track_cls_if_enabled(inpL, "layer_input", il);

        // DEBUG: Track inpL at start of layer
        if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
            char name[64];
            snprintf(name, sizeof(name), "layer_%d_inpL_start", il);
            g_debug_tensors.push_back({inpL, std::string(name), il});
        }

        // PRE-NORM: Apply attn_norm BEFORE attention computation
        // ModernBERT layer 0 has no attn_norm (acts as identity), layers 1-21 have it
        ggml_tensor * attn_residual_base = cur;  // Save unnormalized input for residual add
        if (model.arch == LLM_ARCH_MODERNBERT) {
            if (il == 0) {
                // Layer 0: No attn_norm tensor (identity/no-op in HuggingFace)
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: Skipping attn_norm, input shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
                // VALUE DEBUG: Print first 5 values of layer input
                if (il < 2) {
                    ggml_set_output(cur);  // Mark for value inspection
                    LLAMA_LOG_INFO("[VALUE_DEBUG] Layer %d input marked for inspection\n", il);
                }
            } else {
                // Layers 1-21: Apply normalization BEFORE attention
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: Applying attn_norm BEFORE attention, input shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
                if (il < 2) {
                    ggml_set_output(cur);  // Mark input before norm
                    LLAMA_LOG_INFO("[VALUE_DEBUG] Layer %d pre-attn_norm input marked\n", il);
                }
                cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);
                cb(cur, "attn_norm", il);

                // CLS TRACK: Mark after attn_norm
                track_cls_if_enabled(cur, "after_attn_norm", il);

                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: After attn_norm, output shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
                if (il < 2) {
                    ggml_set_output(cur);  // Mark output after norm
                    LLAMA_LOG_INFO("[VALUE_DEBUG] Layer %d post-attn_norm output marked\n", il);
                }
                if (g_debug_layers_enabled) {
                    char name[64];
                    snprintf(name, sizeof(name), "layer_%d_attn_norm_out", il);
                    g_debug_tensors.push_back({cur, std::string(name), il});
                }
            }
        }

        {
            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;

            // self-attention
            // Check for ModernBERT that critical tensors exist
            if (model.arch == LLM_ARCH_MODERNBERT) {
                if (!model.layers[il].wqkv && (!model.layers[il].wq || !model.layers[il].wk || !model.layers[il].wv)) {
                    throw std::runtime_error("ModernBERT layer " + std::to_string(il) + " missing attention weight tensors");
                }
                if (!model.layers[il].wo) {
                    throw std::runtime_error("ModernBERT layer " + std::to_string(il) + " missing attention output tensor (wo)");
                }
                // Only check for attn_out_norm in non-ModernBERT or layer != 0
                if (model.arch != LLM_ARCH_MODERNBERT && !model.layers[il].attn_out_norm) {
                    throw std::runtime_error("Layer " + std::to_string(il) + " missing attention output norm tensor");
                }
            }

            if (model.layers[il].wqkv) {
                // DEBUG: Mark input to Wqkv for layer 0
                if (il == 0 && g_dump_tensor_values) {
                    ggml_set_output(cur);
                    dump_tensor_values("layer0_input_to_wqkv", cur, il);
                    LLAMA_LOG_INFO("[TENSOR_DUMP] Layer 0: Input to Wqkv marked for inspection\n");
                }

                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                // DEBUG: Mark Wqkv output for layer 0
                if (il == 0 && g_dump_tensor_values) {
                    ggml_set_output(cur);
                    dump_tensor_values("layer0_wqkv_output", cur, il);
                    LLAMA_LOG_INFO("[TENSOR_DUMP] Layer 0: Wqkv output marked for inspection\n");
                }

                if (model.layers[il].bqkv) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }

                // DEBUG: Print QKV offset calculations
                if (il == 0) {
                    const size_t q_offset = 0 * sizeof(float) * (n_embd);
                    const size_t k_offset = 1 * sizeof(float) * (n_embd);
                    const size_t v_offset = 1 * sizeof(float) * (n_embd + n_embd_gqa);
                    LLAMA_LOG_INFO("[QKV_OFFSET_DEBUG] Layer 0:\n");
                    LLAMA_LOG_INFO("  n_embd=%u, n_embd_gqa=%u, n_embd_head=%u, n_head=%u, n_head_kv=%u\n",
                        n_embd, n_embd_gqa, n_embd_head, n_head, n_head_kv);
                    LLAMA_LOG_INFO("  cur shape: ne[0]=%lld, ne[1]=%lld, nb[0]=%zu, nb[1]=%zu\n",
                        cur->ne[0], cur->ne[1], cur->nb[0], cur->nb[1]);
                    LLAMA_LOG_INFO("  Q offset: %zu bytes (element %zu)\n", q_offset, q_offset / sizeof(float));
                    LLAMA_LOG_INFO("  K offset: %zu bytes (element %zu)\n", k_offset, k_offset / sizeof(float));
                    LLAMA_LOG_INFO("  V offset: %zu bytes (element %zu)\n", v_offset, v_offset / sizeof(float));
                    LLAMA_LOG_INFO("  Expected V offset: %zu bytes (element 1536)\n", 2 * n_embd * sizeof(float));
                }

                Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, n_embd_head * sizeof(float), cur->nb[1],
                                    0 * sizeof(float) * (n_embd));
                Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd));
                Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));
            } else {
                Qcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wq, cur), model.layers[il].bq);
                Kcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wk, cur), model.layers[il].bk);
                Vcur = ggml_add(ctx0, build_lora_mm(model.layers[il].wv, cur), model.layers[il].bv);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            }

            if (model.layers[il].attn_q_norm) {
                Qcur = ggml_reshape_2d(ctx0, Qcur, n_embd_head * n_head, n_tokens);

                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, model.layers[il].attn_q_norm_b, LLM_NORM, il);

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            }

            if (model.layers[il].attn_k_norm) {
                Kcur = ggml_reshape_2d(ctx0, Kcur, n_embd_head * n_head_kv, n_tokens);

                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, model.layers[il].attn_k_norm_b, LLM_NORM, il);

                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            }

            // RoPE
            if (model.arch == LLM_ARCH_NOMIC_BERT || model.arch == LLM_ARCH_NOMIC_BERT_MOE ||
                model.arch == LLM_ARCH_JINA_BERT_V3 || model.arch == LLM_ARCH_MODERNBERT) {

                // Get per-layer RoPE frequency for ModernBERT (global vs local)
                const float freq_base_l  = model.arch == LLM_ARCH_MODERNBERT ? model.get_rope_freq_base(cparams, il)  : freq_base;
                const float freq_scale_l = model.arch == LLM_ARCH_MODERNBERT ? model.get_rope_freq_scale(cparams, il) : freq_scale;

                // TOKEN 20778 DEBUG: Log Q/K/V before RoPE
                if (g_debug_token_20778 && model.arch == LLM_ARCH_MODERNBERT && il == 0) {
                    LLAMA_LOG_INFO("[TOKEN_20778] ===== LAYER 0 BEFORE RoPE =====\n");
                    LLAMA_LOG_INFO("[TOKEN_20778]   Qcur: shape=[%ld,%ld,%ld], type=%d, data=%p\n",
                                   (long)Qcur->ne[0], (long)Qcur->ne[1], (long)Qcur->ne[2],
                                   (int)Qcur->type, (void*)Qcur->data);
                    LLAMA_LOG_INFO("[TOKEN_20778]   Kcur: shape=[%ld,%ld,%ld], type=%d, data=%p\n",
                                   (long)Kcur->ne[0], (long)Kcur->ne[1], (long)Kcur->ne[2],
                                   (int)Kcur->type, (void*)Kcur->data);
                    LLAMA_LOG_INFO("[TOKEN_20778]   Vcur: shape=[%ld,%ld,%ld], type=%d, data=%p\n",
                                   (long)Vcur->ne[0], (long)Vcur->ne[1], (long)Vcur->ne[2],
                                   (int)Vcur->type, (void*)Vcur->data);
                    LLAMA_LOG_INFO("[TOKEN_20778]   RoPE: freq_base=%.1f, freq_scale=%.4f, n_rot=%d\n",
                                   freq_base_l, freq_scale_l, n_rot);

                    // Force Q/K/V as outputs for inspection
                    ggml_set_output(Qcur);
                    ggml_set_output(Kcur);
                    ggml_set_output(Vcur);
                }

                // DEBUG: Log RoPE parameters and Q/K/V values for ModernBERT layer 0
                if (model.arch == LLM_ARCH_MODERNBERT && il == 0) {
                    LLAMA_LOG_INFO("[QKV_DEBUG] Layer %d: n_rot=%d, rope_type=%d, freq_base_l=%.1f, freq_scale_l=%.4f, is_swa=%d\n",
                                   il, n_rot, rope_type, freq_base_l, freq_scale_l, hparams.is_swa(il));
                    LLAMA_LOG_INFO("[QKV_DEBUG] Layer 0: Qcur shape=[%ld,%ld,%ld], Kcur shape=[%ld,%ld,%ld], Vcur shape=[%ld,%ld,%ld]\n",
                                   Qcur->ne[0], Qcur->ne[1], Qcur->ne[2],
                                   Kcur->ne[0], Kcur->ne[1], Kcur->ne[2],
                                   Vcur->ne[0], Vcur->ne[1], Vcur->ne[2]);

                    // Mark Q/K/V before RoPE as outputs so we can inspect them
                    ggml_set_output(Qcur);
                    ggml_set_output(Kcur);
                    ggml_set_output(Vcur);
                }

                // Apply RoPE (unless bypassed for debugging)
                if (!g_disable_rope) {
                    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                         ext_factor, attn_factor, beta_fast, beta_slow);

                    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                         ext_factor, attn_factor, beta_fast, beta_slow);

                    // DEBUG: Mark Q/K after RoPE for layer 0
                    if (il == 0 && g_dump_tensor_values) {
                        ggml_set_output(Qcur);
                        ggml_set_output(Kcur);
                        dump_tensor_values("layer0_qcur_after_rope", Qcur, il);
                        dump_tensor_values("layer0_kcur_after_rope", Kcur, il);
                        LLAMA_LOG_INFO("[TENSOR_DUMP] Layer 0: Q/K after RoPE marked for inspection\n");
                    }
                } else {
                    LLAMA_LOG_INFO("[ROPE_BYPASS] Layer %d: Skipping RoPE application\n", il);
                }

                // TOKEN 20778 DEBUG: Log Q/K after RoPE
                if (g_debug_token_20778 && model.arch == LLM_ARCH_MODERNBERT && il == 0) {
                    LLAMA_LOG_INFO("[TOKEN_20778] ===== LAYER 0 AFTER RoPE =====\n");
                    LLAMA_LOG_INFO("[TOKEN_20778]   Qcur: shape=[%ld,%ld,%ld], type=%d, data=%p\n",
                                   (long)Qcur->ne[0], (long)Qcur->ne[1], (long)Qcur->ne[2],
                                   (int)Qcur->type, (void*)Qcur->data);
                    LLAMA_LOG_INFO("[TOKEN_20778]   Kcur: shape=[%ld,%ld,%ld], type=%d, data=%p\n",
                                   (long)Kcur->ne[0], (long)Kcur->ne[1], (long)Kcur->ne[2],
                                   (int)Kcur->type, (void*)Kcur->data);

                    // Force Q/K as outputs for inspection
                    ggml_set_output(Qcur);
                    ggml_set_output(Kcur);

                    // Check for buffer aliasing
                    if (Qcur->data == Kcur->data) {
                        LLAMA_LOG_ERROR("[TOKEN_20778]   ⚠️  WARNING: Q AND K SHARE THE SAME BUFFER!\n");
                    }
                }

                // DEBUG: Mark Q/K after RoPE as outputs for inspection
                if (model.arch == LLM_ARCH_MODERNBERT && il == 0) {
                    ggml_set_output(Qcur);
                    ggml_set_output(Kcur);
                }
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // INSTRUMENTATION: Dump Q/K/V values for layer 0 to compare with HF
            if (il == 0 && model.arch == LLM_ARCH_MODERNBERT) {
                // This is during graph construction, data won't be available yet
                // Mark for output so we can inspect after execution
                LLAMA_LOG_INFO("[TENSOR_DUMP] Layer 0 Q/K/V marked for inspection\n");
            }

            // ModernBERT: Sliding window attention is automatically applied to local layers
            // Global layers (il % global_attn_every_n_layers == 0) use full attention
            // Local layers use bidirectional sliding window attention (SYMMETRIC)
            // The build_attn function selects the appropriate mask based on hparams.is_swa(il)

            // DEBUG: Log SWA status for first few layers
            if (model.arch == LLM_ARCH_MODERNBERT && il < 3) {
                const bool is_swa = hparams.is_swa(il);
                const bool expected_global = (il % hparams.global_attn_every_n_layers == 0);
                LLAMA_LOG_INFO("[MODERNBERT_ATTN] Layer %d: is_swa=%d, expected_global=%d, pattern=%u\n",
                               il, is_swa, expected_global, hparams.global_attn_every_n_layers);
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            // CLS TRACK: Mark after attention
            track_cls_if_enabled(cur, "after_attention", il);

            // DEBUG: Track attention output
            if (model.arch == LLM_ARCH_MODERNBERT && il < 3) {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: After attention, output shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
            }
            if (model.arch == LLM_ARCH_MODERNBERT && il < 2) {
                ggml_set_output(cur);
                LLAMA_LOG_INFO("[VALUE_DEBUG] Layer %d attention output marked\n", il);
            }
            if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
                char attn_name[64];
                snprintf(attn_name, sizeof(attn_name), "layer_%d_attn_out", il);
                g_debug_tensors.push_back({cur, std::string(attn_name), il});
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
                LLAMA_LOG_INFO("[DEBUG] Layer %d: Applying get_rows for last layer\n", il);
            }
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // DEBUG: Track both operands before residual add
        if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
            char name1[64], name2[64];
            snprintf(name1, sizeof(name1), "layer_%d_cur_before_add", il);
            snprintf(name2, sizeof(name2), "layer_%d_attn_residual_base_before_add", il);
            g_debug_tensors.push_back({cur, std::string(name1), il});
            g_debug_tensors.push_back({attn_residual_base, std::string(name2), il});
        }

        // PRE-NORM: Add attention output to UNNORMALIZED input
        // This matches HuggingFace: hidden_states = attention_output + hidden_states
        // where hidden_states is the value BEFORE attn_norm
        if (model.arch == LLM_ARCH_MODERNBERT) {
            if (il < 3) {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: Before attn residual add - attn_out shape=[%lld,%lld,%lld,%lld], base shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3],
                    attn_residual_base->ne[0], attn_residual_base->ne[1], attn_residual_base->ne[2], attn_residual_base->ne[3]);
            }
            ggml_set_output(cur);                   // Protect attention output
            ggml_set_output(attn_residual_base);    // Protect unnormalized input
            cur = ggml_add(ctx0, cur, attn_residual_base);
            ggml_set_output(cur);  // Protect residual add result

            // CLS TRACK: Mark after attention residual add
            track_cls_if_enabled(cur, "after_attn_residual", il);

            if (il < 3) {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: After attn residual add, shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
            }
            if (il < 2) {
                LLAMA_LOG_INFO("[VALUE_DEBUG] Layer %d post-attn-residual marked\n", il);
            }
        } else {
            cur = ggml_add(ctx0, cur, inpL);
        }

        // DEBUG: Track after attention residual add
        if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
            char name[64];
            snprintf(name, sizeof(name), "layer_%d_attn_residual_add", il);
            g_debug_tensors.push_back({cur, std::string(name), il});
        }

        // PRE-NORM: For ModernBERT, save the value BEFORE mlp_norm for FFN residual add
        // HuggingFace does: hidden_states = hidden_states + mlp_output
        // where hidden_states is the value BEFORE mlp_norm, not after
        ggml_tensor * ffn_residual_base = cur;  // Save unnormalized value for residual add

        // PRE-NORM: Apply mlp_norm BEFORE FFN computation (for ModernBERT)
        // Note: IN POST-NORM architectures, this normalization happened after the residual add
        // In PRE-NORM, we apply it here before the FFN
        // The converter maps HF's 'mlp_norm' to llama.cpp's 'layer_out_norm'
        if (model.arch == LLM_ARCH_MODERNBERT) {
            // All layers (0-21) should have mlp_norm (layer_out_norm in llama.cpp naming)
            if (model.layers[il].layer_out_norm) {
                if (il < 3) {
                    LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: Before mlp_norm, input shape=[%lld,%lld,%lld,%lld]\n",
                        il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
                }
                cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);
                cb(cur, "mlp_norm", il);
                if (il < 3) {
                    LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: After mlp_norm, output shape=[%lld,%lld,%lld,%lld]\n",
                        il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
                }
                if (g_debug_layers_enabled) {
                    char name[64];
                    snprintf(name, sizeof(name), "layer_%d_mlp_norm_out", il);
                    g_debug_tensors.push_back({cur, std::string(name), il});
                }
            } else {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: WARNING - No mlp_norm (layer_out_norm) tensor!\n", il);
            }
        }

        // DEBUG: Track FFN input (after mlp_norm for ModernBERT)
        if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
            char name[64];
            snprintf(name, sizeof(name), "layer_%d_ffn_input", il);
            g_debug_tensors.push_back({cur, std::string(name), il});
        }

        if (model.layers[il].attn_norm_2 != nullptr) {
            cur = ggml_add(ctx0, cur, inpL);  // re-add the layer input
            cur = build_norm(cur, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, il);
        }

        // For ModernBERT, use the normalized cur as FFN input, but use ffn_residual_base for residual add
        // For other models, use cur for both (preserves existing behavior)
        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        if (hparams.moe_every_n_layers > 0 && il % hparams.moe_every_n_layers == 1) {
            // MoE branch
            cur = build_moe_ffn(cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps, nullptr,
                                model.layers[il].ffn_down_exps, nullptr, hparams.n_expert, hparams.n_expert_used,
                                LLM_FFN_GELU, false, false, 0.0f, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
            cb(cur, "ffn_moe_out", il);
        } else if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_NOMIC_BERT_MOE ||
                   model.arch == LLM_ARCH_JINA_BERT_V3) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);
        } else if (model.arch == LLM_ARCH_MODERNBERT) {
            // ModernBERT uses GeGLU (Gated GELU) activation with no bias terms
            // FFN flow: x -> [gate, up] -> GELU(gate) * up -> down
            // Note: We use LLM_FFN_GELU + LLM_FFN_PAR because gate and up are separate tensors
            // (LLM_FFN_GEGLU is for fused tensors only)
            if (model.layers[il].ffn_gate == nullptr || model.layers[il].ffn_up == nullptr || model.layers[il].ffn_down == nullptr) {
                throw std::runtime_error("ModernBERT layer " + std::to_string(il) + " missing required FFN tensors");
            }
            if (il < 3) {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: Before FFN, input shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
            }
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL, NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);

            // CLS TRACK: Mark after FFN
            track_cls_if_enabled(cur, "after_ffn", il);

            if (il < 3) {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: After FFN, output shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
            }
        } else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                    model.layers[il].ffn_gate ? LLM_FFN_GELU : LLM_FFN_GEGLU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        // DEBUG: Track FFN output before residual
        if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
            char ffn_pre_name[64];
            snprintf(ffn_pre_name, sizeof(ffn_pre_name), "layer_%d_ffn_pre_residual", il);
            g_debug_tensors.push_back({cur, std::string(ffn_pre_name), il});
        }

        // DEBUG: Log FFN output
        if (model.arch == LLM_ARCH_MODERNBERT && il == 0) {
            LLAMA_LOG_INFO("[NaN_DEBUG] Layer %d: FFN output cur=%p, ffn_inp=%p, ffn_residual_base=%p\n",
                           il, (void*)cur, (void*)ffn_inp, (void*)ffn_residual_base);
        }

        // CRITICAL FIX: Protect FFN residual add operands and result
        if (model.arch == LLM_ARCH_MODERNBERT) {
            if (il < 3) {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: Before FFN residual add - ffn_out shape=[%lld,%lld,%lld,%lld], base shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3],
                    ffn_residual_base->ne[0], ffn_residual_base->ne[1], ffn_residual_base->ne[2], ffn_residual_base->ne[3]);
            }
            ggml_set_output(cur);                // Protect FFN output
            ggml_set_output(ffn_residual_base);  // Protect residual base (unnormalized attention output)
        }
        // CRITICAL: For ModernBERT, add FFN output to the UNNORMALIZED value (ffn_residual_base)
        // This matches HuggingFace: hidden_states = hidden_states + mlp_output
        // where hidden_states is the value BEFORE mlp_norm
        if (model.arch == LLM_ARCH_MODERNBERT) {
            cur = ggml_add(ctx0, cur, ffn_residual_base);
        } else {
            cur = ggml_add(ctx0, cur, ffn_inp);
        }
        if (model.arch == LLM_ARCH_MODERNBERT) {
            ggml_set_output(cur);  // Protect FFN residual add result

            // CLS TRACK: Mark after FFN residual add
            track_cls_if_enabled(cur, "after_ffn_residual", il);

            if (il < 3) {
                LLAMA_LOG_INFO("[PRE_NORM_DEBUG] Layer %d: After FFN residual add, shape=[%lld,%lld,%lld,%lld]\n",
                    il, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
            }
        }

        // DEBUG: Log after FFN residual add
        if (model.arch == LLM_ARCH_MODERNBERT && il == 0) {
            LLAMA_LOG_INFO("[NaN_DEBUG] Layer %d: After FFN residual add cur=%p\n", il, (void*)cur);
        }

        // DEBUG: Track after FFN residual add
        if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
            char ffn_post_name[64];
            snprintf(ffn_post_name, sizeof(ffn_post_name), "layer_%d_ffn_post_residual", il);
            g_debug_tensors.push_back({cur, std::string(ffn_post_name), il});
        }

        // output layer norm
        // CRITICAL: ModernBERT uses PRE-NORM architecture
        // - layer_out_norm is applied BEFORE FFN (as mlp_norm), not here after FFN
        // - The layer output is the FFN residual add result (no POST-normalization)
        // Other architectures use POST-NORM and apply layer_out_norm here
        if (model.arch != LLM_ARCH_MODERNBERT) {
            cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);
        }

        // input for next layer
        inpL = cur;

        // DEBUG: Log final layer output
        if (model.arch == LLM_ARCH_MODERNBERT && il == 0) {
            LLAMA_LOG_INFO("[NaN_DEBUG] Layer %d: Final output inpL=%p\n", il, (void*)inpL);
        }

        // CRITICAL FIX: Protect layer outputs for ModernBERT
        if (model.arch == LLM_ARCH_MODERNBERT) {
            ggml_set_output(inpL);
        }

        // DEBUG: Track layer output
        if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
            char layer_name[64];
            snprintf(layer_name, sizeof(layer_name), "layer_%d", il);
            g_debug_tensors.push_back({inpL, std::string(layer_name), il});
        }
    }

    cur = inpL;

    // ModernBERT applies final_norm (output_norm) after all encoder layers
    // This is a learned LayerNorm, not an identity transform
    if (model.output_norm) {
        // CLS TRACK: Mark before final norm
        track_cls_if_enabled(cur, "before_final_norm", -1);

        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
        cb(cur, "result_norm", -1);

        // CLS TRACK: Mark after final norm
        track_cls_if_enabled(cur, "after_final_norm", -1);

        if (model.arch == LLM_ARCH_MODERNBERT) {
            LLAMA_LOG_INFO("[MODERNBERT_DEBUG] Applied final_norm (output_norm)\n");
        }

        // DEBUG: Track final norm output
        if (g_debug_layers_enabled) {
            g_debug_tensors.push_back({cur, "final_norm", -2});
        }
    }

    // Apply L2 normalization if requested (for sentence-transformers models)
    // This is separate from LayerNorm and applies to the final embeddings
    if (model.arch == LLM_ARCH_MODERNBERT && hparams.normalize_embeddings) {
        // FIX: Use ggml_l2_norm for L2 vector normalization, not ggml_norm (layer norm)
        cur = ggml_l2_norm(ctx0, cur, 1e-12f);
        cb(cur, "result_l2_norm", -1);
    }

    // DEBUG: Mark final output for layer-by-layer debugging
    if (g_debug_layers_enabled && model.arch == LLM_ARCH_MODERNBERT) {
        g_debug_tensors.push_back({cur, "final_output", -2});
    }

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}

// DEBUG: Function to retrieve debug tensors for inspection after computation
std::vector<debug_tensor_info> & llm_get_debug_tensors() {
    return g_debug_tensors;
}

bool llm_debug_layers_enabled() {
    return g_debug_layers_enabled;
}
