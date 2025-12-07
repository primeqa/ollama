package convert

import (
	"cmp"
	"encoding/json"
	"io/fs"
	"log/slog"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

type modernBertModel struct {
	ModelParameters
	NumHiddenLayers           uint32  `json:"num_hidden_layers"`
	MaxPositionEmbeddings     uint32  `json:"max_position_embeddings"`
	HiddenSize                uint32  `json:"hidden_size"`
	IntermediateSize          uint32  `json:"intermediate_size"`
	NumAttentionHeads         uint32  `json:"num_attention_heads"`
	LayerNormEPS              float32 `json:"layer_norm_eps"`
	GlobalAttnEveryNLayers    uint32  `json:"global_attn_every_n_layers"`
	LocalAttention            uint32  `json:"local_attention"`
	LocalRopeTheta            float32 `json:"local_rope_theta"`
	GlobalRopeTheta           float32 `json:"global_rope_theta"`
	HiddenActivation          string  `json:"hidden_activation"`
	ClassifierPooling         string  `json:"classifier_pooling"`
	normalizeEmbeddings       bool
	PoolingType               uint32
}

var (
	_ ModelConverter = (*modernBertModel)(nil)
	_ moreParser     = (*modernBertModel)(nil)
)

func (p *modernBertModel) parseMore(fsys fs.FS) error {
	// Parse sentence_transformers module config if present
	bts, err := fs.ReadFile(fsys, "modules.json")
	if err != nil {
		// Not all models have this, return nil if missing
		return nil
	}

	var modules []struct {
		Type string `json:"type"`
		Path string `json:"path"`
	}

	if err := json.Unmarshal(bts, &modules); err != nil {
		return err
	}

	var pooling string
	for _, m := range modules {
		switch m.Type {
		case "sentence_transformers.models.Pooling":
			pooling = m.Path
		case "sentence_transformers.models.Normalize":
			p.normalizeEmbeddings = true
		}
	}

	if pooling != "" {
		bts, err := fs.ReadFile(fsys, filepath.Join(pooling, "config.json"))
		if err == nil {
			var pc struct {
				PoolingModeCLSToken   bool `json:"pooling_mode_cls_token"`
				PoolingModeMeanTokens bool `json:"pooling_mode_mean_tokens"`
			}

			if err := json.Unmarshal(bts, &pc); err == nil {
				if pc.PoolingModeMeanTokens {
					p.PoolingType = 1 // Mean pooling
					return nil
				} else if pc.PoolingModeCLSToken {
					p.PoolingType = 2 // CLS pooling
					return nil
				}
			}
		}
		// If pooling config file missing or invalid, fall through to default
	}

	// ModernBERT uses CLS pooling by default based on classifier_pooling
	// Debug: log what we're seeing
	slog.Debug("modernbert pooling config", "classifier_pooling", p.ClassifierPooling, "pooling_path", pooling)
	if p.ClassifierPooling == "mean" {
		p.PoolingType = 1 // Mean pooling
	} else {
		p.PoolingType = 2 // CLS pooling (default)
	}

	return nil
}

func (p *modernBertModel) KV(t *Tokenizer) ggml.KV {
	kv := p.ModelParameters.KV(t)

	kv["general.architecture"] = "modernbert"
	kv["modernbert.attention.causal"] = false
	kv["modernbert.pooling_type"] = p.PoolingType
	kv["modernbert.normalize_embeddings"] = p.normalizeEmbeddings

	kv["modernbert.block_count"] = p.NumHiddenLayers
	kv["modernbert.context_length"] = p.MaxPositionEmbeddings
	kv["modernbert.embedding_length"] = p.HiddenSize
	kv["modernbert.feed_forward_length"] = p.IntermediateSize
	kv["modernbert.attention.head_count"] = p.NumAttentionHeads
	kv["modernbert.attention.layer_norm_epsilon"] = p.LayerNormEPS

	// ModernBERT-specific parameters for alternating attention
	kv["modernbert.attention.global_attn_every_n_layers"] = cmp.Or(p.GlobalAttnEveryNLayers, 3)
	kv["modernbert.attention.local_attn_window"] = cmp.Or(p.LocalAttention, 128)
	kv["modernbert.rope.freq_base_local"] = cmp.Or(p.LocalRopeTheta, 10000.0)
	kv["modernbert.rope.freq_base_global"] = cmp.Or(p.GlobalRopeTheta, 80000.0)

	kv["tokenizer.ggml.model"] = "bert"
	kv["tokenizer.ggml.token_type_count"] = uint32(2)

	// Convert to phantom space tokens (like BERT/NomicBERT)
	for i, e := range t.Tokens {
		if strings.HasPrefix(e, "[") && strings.HasSuffix(e, "]") {
			// Keep special tokens as-is
		} else if strings.HasPrefix(e, "##") {
			t.Tokens[i] = e[2:]
		} else {
			t.Tokens[i] = "\u2581" + e
		}
	}

	kv["tokenizer.ggml.tokens"] = t.Tokens

	return kv
}

func (p *modernBertModel) Tensors(ts []Tensor) []*ggml.Tensor {
	var out []*ggml.Tensor
	for _, t := range ts {
		// Skip pooler layers and position IDs (we do pooling in the runtime)
		if slices.Contains([]string{
			"embeddings.position_ids",
			"pooler.dense.weight",
			"pooler.dense.bias",
		}, t.Name()) {
			continue
		}

		out = append(out, &ggml.Tensor{
			Name:     t.Name(),
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (modernBertModel) Replacements() []string {
	return []string{
		// ModernBERT uses "layers.N" not "encoder.layers.N"
		"layers", "blk",
		"encoder.layer", "blk",
		"encoder.layers", "blk",
		"embeddings.tok_embeddings", "token_embd",
		"embeddings.word_embeddings", "token_embd",
		"embeddings.norm", "token_embd_norm",
		"embeddings.LayerNorm", "token_embd_norm",
		"final_norm", "output_norm",
		"attn.Wqkv", "attn_qkv",
		"attn.Wo", "attn_output",
		"attention.self.query", "attn_q",
		"attention.self.key", "attn_k",
		"attention.self.value", "attn_v",
		"attention.output.dense", "attn_output",
		"attention.output.LayerNorm", "attn_output_norm",
		"attn_norm", "attn_output_norm",
		"mlp.Wi", "ffn_up",
		"mlp.Wo", "ffn_down",
		"intermediate.dense", "ffn_up",
		"output.dense", "ffn_down",
		"output.LayerNorm", "layer_output_norm",
		"mlp_norm", "layer_output_norm",
	}
}

func (modernBertModel) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}
