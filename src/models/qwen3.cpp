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

void llama_model_qwen3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const bool pipeline_brick = params.pipeline_brick_enabled;
    const bool pipeline_head  = params.pipeline_brick_role == LLAMA_PIPELINE_BRICK_ROLE_HEAD;
    const bool pipeline_tail  = params.pipeline_brick_role == LLAMA_PIPELINE_BRICK_ROLE_TAIL;

    if (pipeline_brick) {
        if (hparams.n_layer() != 36 || hparams.n_embd != 2560) {
            throw std::runtime_error("pipeline-brick prototype only supports dense Qwen3-4B");
        }
        if (!pipeline_head && !pipeline_tail) {
            throw std::runtime_error("pipeline-brick requires head or tail role");
        }
        if (params.pipeline_brick_layer_start < 0 ||
                params.pipeline_brick_layer_end > (int32_t) hparams.n_layer() ||
                params.pipeline_brick_layer_start >= params.pipeline_brick_layer_end) {
            throw std::runtime_error("invalid pipeline-brick layer range");
        }
    }

    if (!pipeline_brick || pipeline_head) {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    } else {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | TENSOR_SKIP);
    }

    // output
    const int output_flags = pipeline_brick && pipeline_head ? TENSOR_NOT_REQUIRED | TENSOR_SKIP : 0;
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, output_flags);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED | output_flags);
    // if output is NULL, init from the input tok embed
    if (output == NULL && (!pipeline_brick || pipeline_tail)) {
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

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, layer_flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, layer_flags);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, layer_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, layer_flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, layer_flags);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, layer_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, layer_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, layer_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const bool pipeline_brick = model.pipeline_brick_enabled();
    const bool pipeline_head  = model.pipeline_brick_role() == LLAMA_PIPELINE_BRICK_ROLE_HEAD;
    const bool pipeline_tail  = model.pipeline_brick_role() == LLAMA_PIPELINE_BRICK_ROLE_TAIL;
    const int layer_start = pipeline_brick ? model.pipeline_brick_layer_start() : 0;
    const int layer_end   = pipeline_brick ? model.pipeline_brick_layer_end()   : n_layer;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    if (pipeline_tail) {
        inpL = build_inp_hidden();
    } else {
        inpL = build_inp_embd(model.tok_embd);
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = cparams.ema_kv_active ? nullptr : build_attn_inp_kv();

    ggml_tensor * inp_out_ids = pipeline_head ? nullptr : build_inp_out_ids();

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
                    n_embd_head, n_head, n_head_kv, il);

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
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    if (pipeline_head) {
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
