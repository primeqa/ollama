package convert

import (
	"cmp"
	"encoding/json"
	"io/fs"
	"log/slog"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type modernBertModel struct {
	ModelParameters
	NumHiddenLayers        uint32  `json:"num_hidden_layers"`
	MaxPositionEmbeddings  uint32  `json:"max_position_embeddings"`
	HiddenSize             uint32  `json:"hidden_size"`
	IntermediateSize       uint32  `json:"intermediate_size"`
	NumAttentionHeads      uint32  `json:"num_attention_heads"`
	LayerNormEPS           float32 `json:"norm_eps"`
	GlobalAttnEveryNLayers uint32  `json:"global_attn_every_n_layers"`
	LocalAttention         uint32  `json:"local_attention"`
	LocalRopeTheta         float32 `json:"local_rope_theta"`
	GlobalRopeTheta        float32 `json:"global_rope_theta"`
	HiddenActivation       string  `json:"hidden_activation"`
	ClassifierPooling      string  `json:"classifier_pooling"`
	normalizeEmbeddings    bool
	PoolingType            uint32
}

var (
	_ ModelConverter = (*modernBertModel)(nil)
	_ moreParser     = (*modernBertModel)(nil)
)

func (p *modernBertModel) parseMore(fsys fs.FS) error {
	// Parse sentence_transformers module config if present
	var hasPoolingModule bool
	bts, err := fs.ReadFile(fsys, "modules.json")
	if err == nil {
		var modules []struct {
			Type string `json:"type"`
			Path string `json:"path"`
		}

		if err := json.Unmarshal(bts, &modules); err != nil {
			return err
		}

		for _, m := range modules {
			switch m.Type {
			case "sentence_transformers.models.Pooling":
				hasPoolingModule = true
			case "sentence_transformers.models.Normalize":
				p.normalizeEmbeddings = true
			}
		}
	}

	// Set pooling type based on available information
	if hasPoolingModule {
		slog.Debug("modern-bert: detected sentence-transformers Pooling module, using CLS pooling")
		p.PoolingType = 2 // CLS pooling for embedding models
	} else {
		slog.Debug("modern-bert pooling config", "classifier_pooling", p.ClassifierPooling)
		if p.ClassifierPooling == "mean" {
			p.PoolingType = 1 // Mean pooling
		} else if p.ClassifierPooling == "cls" {
			p.PoolingType = 2 // CLS pooling
		} else {
			slog.Warn("modern-bert: unknown classifier_pooling value, defaulting to CLS", "value", p.ClassifierPooling)
			p.PoolingType = 2
		}
	}

	return nil
}

func (p *modernBertModel) KV(t *Tokenizer) KV {
	kv := p.ModelParameters.KV(t)

	// Architecture name aligned with upstream llama.cpp
	kv["general.architecture"] = "modern-bert"

	// Model parameters
	kv["modern-bert.block_count"] = p.NumHiddenLayers
	kv["modern-bert.context_length"] = p.MaxPositionEmbeddings
	kv["modern-bert.embedding_length"] = p.HiddenSize
	kv["modern-bert.feed_forward_length"] = p.IntermediateSize
	kv["modern-bert.attention.head_count"] = p.NumAttentionHeads
	kv["modern-bert.attention.layer_norm_epsilon"] = p.LayerNormEPS
	kv["modern-bert.attention.causal"] = false
	kv["modern-bert.pooling_type"] = p.PoolingType

	// Sliding window attention parameters
	kv["modern-bert.attention.global_attn_every_n_layers"] = cmp.Or(p.GlobalAttnEveryNLayers, uint32(3))
	kv["modern-bert.attention.local_attn_window"] = cmp.Or(p.LocalAttention, uint32(128))

	// RoPE parameters
	kv["modern-bert.rope.freq_base_local"] = cmp.Or(p.LocalRopeTheta, 10000.0)
	kv["modern-bert.rope.freq_base_global"] = cmp.Or(p.GlobalRopeTheta, 80000.0)

	// Embedding normalization
	kv["modern-bert.pooling.normalize_embeddings"] = p.normalizeEmbeddings

	// ModernBERT uses GPT2/BPE tokenizer (like RoBERTa)
	kv["tokenizer.ggml.model"] = "gpt2"
	kv["tokenizer.ggml.pre"] = "gpt-2"
	kv["tokenizer.ggml.token_type_count"] = uint32(2)

	// BERT-like models need CLS (as BOS) and SEP (as EOS) tokens added automatically
	kv["tokenizer.ggml.bos_token_id"] = uint32(50281) // CLS token
	kv["tokenizer.ggml.eos_token_id"] = uint32(50282) // SEP token
	kv["tokenizer.ggml.add_bos_token"] = true
	kv["tokenizer.ggml.add_eos_token"] = true
	kv["tokenizer.ggml.add_sep_token"] = true

	return kv
}

func (p *modernBertModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor

	for _, t := range ts {
		// Skip pooler layers, position IDs, and decoder (MLM head)
		if slices.Contains([]string{
			"embeddings.position_ids",
			"pooler.dense.weight",
			"pooler.dense.bias",
		}, t.Name()) || strings.HasPrefix(t.Name(), "decoder.") {
			continue
		}

		name := t.Name()

		// Strip "model." prefix if present
		if strings.HasPrefix(name, "model.") {
			name = strings.TrimPrefix(name, "model.")
		}

		// ModernBERT uses GeGLU - keep mlp.Wi as combined tensor (don't split)
		// Upstream llama.cpp uses LLM_FFN_GEGLU which expects combined tensor
		out = append(out, &ggml.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (modernBertModel) Replacements() []string {
	return []string{
		// Layer prefix replacements
		"layers", "blk",
		"encoder.layer", "blk",
		"encoder.layers", "blk",
		// Embeddings
		"embeddings.tok_embeddings", "token_embd",
		"embeddings.word_embeddings", "token_embd",
		"embeddings.norm", "token_embd_norm",
		// Final norm
		"final_norm", "output_norm",
		// Attention (upstream tensor names)
		"attn.Wqkv", "attn_qkv",
		"attn.Wo", "attn_output",
		"attn_norm", "attn_norm", // Pre-attention norm
		// FFN (upstream tensor names)
		"mlp.Wi", "ffn_up",    // Combined gate+up tensor for GEGLU
		"mlp.Wo", "ffn_down",
		"mlp_norm", "ffn_norm", // Pre-FFN norm
	}
}

func (modernBertModel) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}
