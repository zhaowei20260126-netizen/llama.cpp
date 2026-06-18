#include "models.h"

void llama_model_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool pipeline_brick = params.pipeline_brick_enabled;
    const bool pipeline_first = pipeline_brick && params.pipeline_brick_layer_start == 0;
    const bool pipeline_last  = pipeline_brick && params.pipeline_brick_layer_end == (int32_t) hparams.n_layer();
    const int32_t tp_rank = params.pipeline_brick_tp_rank;
    const int32_t tp_size = params.pipeline_brick_tp_size;
    const bool tp_enabled = tp_size > 1;
    std::string tp_layout;
    int32_t tp_meta_rank = 0;
    int32_t tp_meta_size = 0;
    int32_t tp_meta_q_heads = 0;
    int32_t tp_meta_kv_heads = 0;
    int32_t tp_meta_ffn = 0;
    const bool tp_has_layout = tp_enabled && ml.get_key("pipeline_brick.tp_layout", tp_layout, false);

    if (pipeline_brick) {
        if (hparams.n_layer() != 36 || hparams.n_embd != 2560) {
            throw std::runtime_error("pipeline-brick prototype only supports dense Qwen3-4B");
        }
        if (params.pipeline_brick_role == LLAMA_PIPELINE_BRICK_ROLE_NONE) {
            throw std::runtime_error("pipeline-brick requires a role");
        }
        if (params.pipeline_brick_layer_start < 0 ||
                params.pipeline_brick_layer_end > (int32_t) hparams.n_layer() ||
                params.pipeline_brick_layer_start >= params.pipeline_brick_layer_end) {
            throw std::runtime_error("invalid pipeline-brick layer range");
        }
        if (tp_enabled) {
            if (tp_rank < 0 || tp_rank >= tp_size) {
                throw std::runtime_error("invalid pipeline-brick TP rank");
            }
            if (tp_size != 2 && tp_size != 4) {
                throw std::runtime_error("pipeline-brick TP prototype requires --tp-size 2 or 4");
            }
            if ((n_embd_head_k * n_head) % tp_size != 0 ||
                    n_embd_gqa % tp_size != 0 ||
                    n_ff % tp_size != 0) {
                throw std::runtime_error("Qwen3-4B dimensions are not divisible by pipeline-brick TP size");
            }
            if (tp_has_layout) {
                if (!ml.get_key("pipeline_brick.tp_rank", tp_meta_rank, false) ||
                        !ml.get_key("pipeline_brick.tp_size", tp_meta_size, false) ||
                        !ml.get_key("pipeline_brick.tp_q_heads", tp_meta_q_heads, false) ||
                        !ml.get_key("pipeline_brick.tp_kv_heads", tp_meta_kv_heads, false) ||
                        !ml.get_key("pipeline_brick.tp_ffn", tp_meta_ffn, false)) {
                    throw std::runtime_error("pipeline-brick asymmetric TP metadata is incomplete");
                }
                if (tp_layout != "rank0-small") {
                    throw std::runtime_error("unsupported pipeline-brick TP layout: " + tp_layout);
                }
                if (tp_meta_rank != tp_rank || tp_meta_size != tp_size) {
                    throw std::runtime_error("pipeline-brick TP metadata does not match requested rank/size");
                }
                if (tp_meta_q_heads <= 0 || tp_meta_kv_heads <= 0 || tp_meta_ffn <= 0) {
                    throw std::runtime_error("pipeline-brick TP metadata contains non-positive local dimensions");
                }
                if (tp_meta_q_heads > (int32_t) n_head ||
                        tp_meta_kv_heads > (int32_t) n_head_kv ||
                        tp_meta_ffn > (int32_t) n_ff) {
                    throw std::runtime_error("pipeline-brick TP metadata exceeds Qwen3-4B dimensions");
                }
                if (tp_meta_q_heads % tp_meta_kv_heads != 0 ||
                        tp_meta_q_heads / tp_meta_kv_heads != (int32_t) (n_head / n_head_kv)) {
                    throw std::runtime_error("pipeline-brick TP metadata breaks Qwen3 GQA grouping");
                }
            }
        }
    } else if (tp_enabled) {
        throw std::runtime_error("pipeline-brick TP requires pipeline-brick mode");
    }

    const int64_t n_head_local    = tp_has_layout ? tp_meta_q_heads  : (tp_enabled ? n_head    / tp_size : n_head);
    const int64_t n_head_kv_local = tp_has_layout ? tp_meta_kv_heads : (tp_enabled ? n_head_kv / tp_size : n_head_kv);
    const int64_t n_embd_q_local  = n_embd_head_k * n_head_local;
    const int64_t n_embd_kv_local = n_embd_head_k * n_head_kv_local;
    const int64_t n_ff_local      = tp_has_layout ? tp_meta_ffn      : (tp_enabled ? n_ff / tp_size : n_ff);

    pipeline_brick_tp_q_heads_local = (int32_t) n_head_local;
    pipeline_brick_tp_kv_heads_local = (int32_t) n_head_kv_local;
    pipeline_brick_tp_ffn_local = (int32_t) n_ff_local;

    if (!pipeline_brick || pipeline_first) {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    } else {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | TENSOR_SKIP);
    }

    // output
    const int output_flags = pipeline_brick && !pipeline_last ? TENSOR_NOT_REQUIRED | TENSOR_SKIP : 0;
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, output_flags);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | output_flags);
    // if output is NULL, init from the input tok embed
    if (output == NULL && (!pipeline_brick || pipeline_last)) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // output rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED | output_flags);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool active_layer = !pipeline_brick ||
            (i >= params.pipeline_brick_layer_start && i < params.pipeline_brick_layer_end);
        const int layer_flags = active_layer ? 0 : TENSOR_NOT_REQUIRED | TENSOR_SKIP;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, layer_flags);

        create_tensor_qkv(layer, i, n_embd, n_embd_q_local, n_embd_kv_local, n_embd_kv_local, layer_flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q_local, n_embd}, layer_flags);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, layer_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, layer_flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, layer_flags);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,      n_ff_local}, layer_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_local,  n_embd}, layer_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,      n_ff_local}, layer_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const bool pipeline_brick = model.pipeline_brick_enabled();
    const int layer_start = pipeline_brick ? model.pipeline_brick_layer_start() : 0;
    const int layer_end   = pipeline_brick ? model.pipeline_brick_layer_end()   : n_layer;
    const bool pipeline_first = pipeline_brick && layer_start == 0;
    const bool pipeline_last  = pipeline_brick && layer_end == n_layer;
    const int32_t tp_size = model.pipeline_brick_tp_size();
    const bool tp_enabled = tp_size > 1;
    const int64_t n_head_local    = tp_enabled ? model.pipeline_brick_tp_q_heads()  : n_head;
    const int64_t n_head_kv_local = tp_enabled ? model.pipeline_brick_tp_kv_heads() : n_head_kv;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(!tp_enabled || n_head_local > 0);
    GGML_ASSERT(!tp_enabled || n_head_kv_local > 0);
    GGML_ASSERT(!tp_enabled || n_head_local % n_head_kv_local == 0);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    if (!pipeline_brick || pipeline_first) {
        inpL = build_inp_embd(model.tok_embd);
    } else {
        inpL = build_inp_hidden();
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = cparams.ema_kv_active ? nullptr : build_attn_inp_kv();

    ggml_tensor * inp_out_ids = (!pipeline_brick || pipeline_last) ? build_inp_out_ids() : nullptr;

    for (int il = layer_start; il < layer_end; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head_local, n_head_kv_local, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            auto * inp_attn_layer = cparams.ema_kv_active ? build_attn_inp_kv(il) : inp_attn;
            cur = build_attn(inp_attn_layer,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            if (tp_enabled) {
                cur = ggml_all_reduce_sum(ctx0, cur);
                cb(cur, "attn_tp_all_reduce", il);
            }
        }
        if (il == layer_end - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        if (tp_enabled) {
            cur = ggml_all_reduce_sum(ctx0, cur);
            cb(cur, "ffn_tp_all_reduce", il);
        }
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    if (pipeline_brick && !pipeline_last) {
        res->t_embd = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
