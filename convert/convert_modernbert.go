package convert

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
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

	// ModernBERT embedding models use CLS pooling by default
	// Note: classifier_pooling is for the classification head, not for embeddings
	// So we should not use it to determine the embedding pooling type
	p.PoolingType = 2 // CLS pooling (default for embeddings)

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
	skippedCount := 0
	mlpWiCount := 0

	for _, t := range ts {
		// Skip pooler layers and position IDs (we do pooling in the runtime)
		if slices.Contains([]string{
			"embeddings.position_ids",
			"pooler.dense.weight",
			"pooler.dense.bias",
		}, t.Name()) {
			slog.Debug("skipping pooler/position tensor", "name", t.Name())
			skippedCount++
			continue
		}

		name := t.Name()

		// Skip attention projection bias tensors for global attention layers
		// Full attention layers don't have attention biases while local attention layers do
		// Don't skip normalization biases (attn_output_norm.bias)
		if strings.Contains(name, ".bias") && !strings.Contains(name, "norm") && (strings.Contains(name, "attn") || strings.Contains(name, "attention")) {
			// Apply layer prefix replacements to parse the layer number correctly
			layerName := name
			layerName = strings.Replace(layerName, "encoder.layer.", "blk.", 1)
			layerName = strings.Replace(layerName, "encoder.layers.", "blk.", 1)
			layerName = strings.Replace(layerName, "layers.", "blk.", 1)

			var layer int
			if _, err := fmt.Sscanf(layerName, "blk.%d.", &layer); err == nil {
				globalAttnEveryN := cmp.Or(p.GlobalAttnEveryNLayers, 3)
				// Skip if it's a global layer (multiple of N) - this includes layer 0
				if layer%int(globalAttnEveryN) == 0 {
					slog.Debug("skipping attention bias for global layer", "layer", layer, "name", name)
					skippedCount++
					continue
				}
			}
		}

		// ModernBERT uses gated FFN (GeGLU) - the mlp.Wi tensor contains both gate and up weights fused
		// We need to split it into two separate tensors
		if strings.Contains(name, "mlp.Wi") {
			mlpWiCount++
			// Get the fused tensor data
			shape := t.Shape()
			slog.Debug("splitting fused mlp.Wi tensor", "name", name, "shape", shape)
			if len(shape) != 2 {
				// Unexpected shape, just pass through
				slog.Debug("unexpected tensor shape for mlp.Wi", "name", name, "shape", shape)
				out = append(out, &ggml.Tensor{
					Name:     name,
					Kind:     t.Kind(),
					Shape:    shape,
					WriterTo: t,
				})
				continue
			}

			// PyTorch stores linear weights as [out_features, in_features]
			// So shape is [2*intermediate_size, hidden_size] = [2304, 768]
			// We need to split along dim 0 into two tensors of [intermediate_size, hidden_size]
			dim0 := shape[0]
			dim1 := shape[1]
			halfDim0 := dim0 / 2
			slog.Debug("split dimensions", "dim0", dim0, "dim1", dim1, "halfDim0", halfDim0)

			// Create ffn_gate tensor (first half of rows)
			// Apply the replacement directly: mlp.Wi -> ffn_gate
			gateName := strings.Replace(name, "mlp.Wi", "ffn_gate", 1)
			out = append(out, &ggml.Tensor{
				Name:     gateName,
				Kind:     t.Kind(),
				Shape:    []uint64{halfDim0, dim1},
				WriterTo: &splitTensorRows{source: t, offset: 0, rows: halfDim0},
			})

			// Create ffn_up tensor (second half of rows)
			// Apply the replacement directly: mlp.Wi -> ffn_up
			upName := strings.Replace(name, "mlp.Wi", "ffn_up", 1)
			out = append(out, &ggml.Tensor{
				Name:     upName,
				Kind:     t.Kind(),
				Shape:    []uint64{halfDim0, dim1},
				WriterTo: &splitTensorRows{source: t, offset: halfDim0, rows: halfDim0},
			})
		} else {
			out = append(out, &ggml.Tensor{
				Name:     name,
				Kind:     t.Kind(),
				Shape:    t.Shape(),
				WriterTo: t,
			})
		}
	}

	// Special case: Layer 0 in ModernBERT doesn't have attn_output_norm in the source model
	// because input is already normalized from embeddings.norm. But llama.cpp expects it,
	// so we create a synthetic one by copying embeddings.norm
	hasLayer0AttnNorm := false
	var embeddingsNormTensor *ggml.Tensor

	for _, t := range out {
		if t.Name == "blk.0.attn_output_norm.weight" {
			hasLayer0AttnNorm = true
		}
		if t.Name == "token_embd_norm.weight" {
			embeddingsNormTensor = t
		}
	}

	if !hasLayer0AttnNorm && embeddingsNormTensor != nil {
		slog.Debug("creating blk.0.attn_output_norm from token_embd_norm")
		out = append(out, &ggml.Tensor{
			Name:     "blk.0.attn_output_norm.weight",
			Kind:     embeddingsNormTensor.Kind,
			Shape:    embeddingsNormTensor.Shape,
			WriterTo: embeddingsNormTensor.WriterTo,
		})
	}

	return out
}

// getTensorKindSize returns the size in bytes for each element type
func getTensorKindSize(kind uint32) uint64 {
	switch kind {
	case tensorKindFP32: // 0
		return 4
	case tensorKindFP16: // 1
		return 2
	case tensorKindBF16: // 30
		return 2
	default:
		// Unknown kind, assume FP32
		return 4
	}
}

// splitTensorRows handles splitting a fused tensor along dimension 0 (rows)
type splitTensorRows struct {
	source Tensor
	offset uint64 // starting row
	rows   uint64 // number of rows to extract
}

func (st *splitTensorRows) WriteTo(w io.Writer) (n int64, err error) {
	shape := st.source.Shape()
	if len(shape) != 2 {
		return 0, fmt.Errorf("splitTensorRows only works with 2D tensors")
	}

	dim1 := shape[1]  // columns
	elemSize := getTensorKindSize(st.source.Kind())

	// Read the entire source tensor
	var buf bytes.Buffer
	if _, err := st.source.WriteTo(&buf); err != nil {
		return 0, err
	}
	data := buf.Bytes()

	// Calculate byte offsets
	// Each row is dim1 elements
	rowSizeBytes := dim1 * elemSize
	startByte := st.offset * rowSizeBytes
	endByte := (st.offset + st.rows) * rowSizeBytes

	// Write the contiguous block of rows
	nn, err := w.Write(data[startByte:endByte])
	return int64(nn), err
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
		// ModernBERT uses gated FFN - tensors are split in Tensors() method
		"mlp.Wgate", "ffn_gate",
		"mlp.Wup", "ffn_up",
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
