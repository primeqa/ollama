#include "models.h"
#include "../llama-impl.h"
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstring>

// DEBUG: Global storage for intermediate tensors
static std::vector<debug_tensor_info> g_debug_tensors;
static bool g_debug_layers_enabled = false;

llm_build_modernbert::llm_build_modernbert(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    // Check if layer debugging is enabled
    const char* debug_env = std::getenv("OLLAMA_DEBUG_LAYERS");
    g_debug_layers_enabled = (debug_env != nullptr && std::strcmp(debug_env, "1") == 0);

    if (g_debug_layers_enabled) {
        g_debug_tensors.clear();
        LLAMA_LOG_INFO("[DEBUG] Layer debugging enabled for ModernBERT\n");
        LLAMA_LOG_INFO("[DEBUG] model.arch = %d, LLM_ARCH_MODERNBERT = %d\n", (int)model.arch, (int)LLM_ARCH_MODERNBERT);
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
                LLAMA_LOG_INFO("[PRE_NORM_V2] Layer %d: Skipping attn_norm (layer 0 has no attn_norm tensor)\n", il);
            } else {
                // Layers 1-21: Apply normalization BEFORE attention
                LLAMA_LOG_INFO("[PRE_NORM_V2] Layer %d: Applying attn_norm BEFORE attention\n", il);
                cur = build_norm(cur, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, il);
                cb(cur, "attn_norm", il);
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
                cur = build_lora_mm(model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                if (model.layers[il].bqkv) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
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

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                     ext_factor, attn_factor, beta_fast, beta_slow);

                // DEBUG: Mark Q/K after RoPE as outputs for inspection
                if (model.arch == LLM_ARCH_MODERNBERT && il == 0) {
                    ggml_set_output(Qcur);
                    ggml_set_output(Kcur);
                }
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

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

            // DEBUG: Track attention output
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
            ggml_set_output(cur);                   // Protect attention output
            ggml_set_output(attn_residual_base);    // Protect unnormalized input
            cur = ggml_add(ctx0, cur, attn_residual_base);
            ggml_set_output(cur);  // Protect residual add result
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
        // Note: In POST-NORM architectures, this normalization happened after the residual add
        // In PRE-NORM, we apply it here before the FFN
        // The converter maps HF's 'mlp_norm' to llama.cpp's 'layer_out_norm'
        if (model.arch == LLM_ARCH_MODERNBERT) {
            // All layers (0-21) should have mlp_norm (layer_out_norm in llama.cpp naming)
            if (model.layers[il].layer_out_norm) {
                LLAMA_LOG_INFO("[PRE_NORM_V2] Layer %d: Applying mlp_norm BEFORE FFN\n", il);
                cur = build_norm(cur, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, il);
                cb(cur, "mlp_norm", il);
                if (g_debug_layers_enabled) {
                    char name[64];
                    snprintf(name, sizeof(name), "layer_%d_mlp_norm_out", il);
                    g_debug_tensors.push_back({cur, std::string(name), il});
                }
            } else {
                LLAMA_LOG_INFO("[PRE_NORM_V2] Layer %d: WARNING - No mlp_norm (layer_out_norm) tensor!\n", il);
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
            cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL, NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
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

    // CRITICAL: ModernBERT does NOT have output_norm!
    // The final output is directly from the last layer (no additional normalization)
    // Only apply output_norm for other BERT models
    if (model.arch != LLM_ARCH_MODERNBERT && model.output_norm) {
        cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
        cb(cur, "result_norm", -1);

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
