package convert

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
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
	LayerNormEPS              float32 `json:"norm_eps"`
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
	var hasPoolingModule bool
	bts, err := fs.ReadFile(fsys, "modules.json")
	if err == nil {
		// modules.json exists, parse it
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
	// Priority: modules.json Pooling module > classifier_pooling config
	if hasPoolingModule {
		// ModernBERT embedding models use CLS pooling (first token)
		// The modules.json indicates this is a sentence-transformers model with a Pooling module
		slog.Debug("modernbert: detected sentence-transformers Pooling module, using CLS pooling")
		p.PoolingType = 2 // CLS pooling for embedding models
	} else {
		// No pooling module - fall back to classifier_pooling setting from config.json
		slog.Debug("modernbert pooling config", "classifier_pooling", p.ClassifierPooling)
		if p.ClassifierPooling == "mean" {
			p.PoolingType = 1 // Mean pooling
		} else if p.ClassifierPooling == "cls" {
			p.PoolingType = 2 // CLS pooling
		} else {
			// Default to CLS pooling for ModernBERT
			slog.Warn("modernbert: unknown classifier_pooling value, defaulting to CLS", "value", p.ClassifierPooling)
			p.PoolingType = 2
		}
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

	// Set general rope.freq_base to the global value (used as default by llama.cpp)
	kv["general.rope.freq_base"] = cmp.Or(p.GlobalRopeTheta, 80000.0)

	// ModernBERT uses GPT2/BPE tokenizer (like RoBERTa), not BERT WordPiece
	kv["tokenizer.ggml.model"] = "gpt2"
	kv["tokenizer.ggml.token_type_count"] = uint32(2)

	// Tokens are already set by ModelParameters.KV(t) - don't overwrite

	// BERT-like models need CLS (as BOS) and SEP (as EOS) tokens added automatically
	// llama.cpp uses add_bos_token and add_eos_token with bos_token_id and eos_token_id
	kv["tokenizer.ggml.bos_token_id"] = uint32(50281) // CLS token
	kv["tokenizer.ggml.eos_token_id"] = uint32(50282) // SEP token
	kv["tokenizer.ggml.add_bos_token"] = true
	kv["tokenizer.ggml.add_eos_token"] = true

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
					continue
				}
			}
		}

		// ModernBERT uses GeGLU (Gated GELU) - the mlp.Wi tensor contains both gate and up weights fused
		// We need to split it into two separate tensors
		if strings.Contains(name, "mlp.Wi") {
			shape := t.Shape()
			if len(shape) != 2 {
				// Unexpected shape, just pass through
				out = append(out, &ggml.Tensor{
					Name:     name,
					Kind:     t.Kind(),
					Shape:    shape,
					WriterTo: t,
				})
				continue
			}

			// PyTorch stores linear weights as [out_features, in_features]
			// For GeGLU, shape is [2*intermediate_size, hidden_size]
			// We need to split along dim 0 into two tensors of [intermediate_size, hidden_size]
			dim0 := shape[0]
			dim1 := shape[1]
			halfDim0 := dim0 / 2

			// Create ffn_gate tensor (first half of rows)
			// ModernBERT's mlp.Wi is organized as [gate; up] (concatenated along dim 0)
			gateName := strings.Replace(name, "mlp.Wi", "ffn_gate", 1)
			out = append(out, &ggml.Tensor{
				Name:     gateName,
				Kind:     t.Kind(),
				Shape:    []uint64{halfDim0, dim1},
				WriterTo: &splitTensorRows{source: t, offset: 0, rows: halfDim0},
			})

			// Create ffn_up tensor (second half of rows)
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
