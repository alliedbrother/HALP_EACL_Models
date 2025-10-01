# Qwen2.5-VL-7B-Instruct: Detailed Architecture & Embedding Extraction

## 📊 Complete Architecture Overview

```
Qwen2_5_VLForConditionalGeneration (Top-level model)
│
├── visual: Qwen2_5_VisionTransformerPretrainedModel
│   └── Vision Encoder (ViT)
│       ├── Input: pixel_values [B, C, H, W] + grid_thw [B, 3]
│       └── Output: vision_features [B, num_vision_tokens, 3584]
│
└── model: Qwen2_5_VLModel
    └── language_model: Qwen2_5_VLTextModel
        ├── Layer 0:  Qwen2_5_VLDecoderLayer  ← EXTRACT HERE
        ├── Layer 1:  Qwen2_5_VLDecoderLayer
        ├── ...
        ├── Layer 7:  Qwen2_5_VLDecoderLayer  ← EXTRACT HERE
        ├── ...
        ├── Layer 14: Qwen2_5_VLDecoderLayer  ← EXTRACT HERE
        ├── ...
        ├── Layer 21: Qwen2_5_VLDecoderLayer  ← EXTRACT HERE
        ├── ...
        └── Layer 27: Qwen2_5_VLDecoderLayer  ← EXTRACT HERE (last)
```

**Key Parameters:**
- Total layers: 28
- Hidden size: 3584
- Attention heads: 28
- Precision: BFloat16

---

## 🔍 Three Types of Embeddings Extracted

### 1️⃣ **Vision-Only Representation** (Before Language Processing)

**Purpose:** Capture pure visual understanding before any text influence

**Code Location:** `_extract_vision_representation()` (lines 187-262)

**Architecture Path:**
```python
model.visual(pixel_values, grid_thw)
```

**Detailed Flow:**

```
Input Image → Processor
    ↓
pixel_values: [1, 3, H, W]  (RGB image tensor)
grid_thw: [1, 3]            (temporal, height, width grid)
    ↓
model.visual (Vision Transformer)
    ├── Patch Embedding
    ├── Position Embedding
    ├── ViT Layers (self-attention on image patches)
    └── Output: [1, num_vision_tokens, 3584]
    ↓
Mean Pooling (dim=1)
    ↓
Final: [3584] ← EXTRACTED AS vision_only_representation
```

**What's Happening:**
1. Image is divided into patches (e.g., 24×24 grid)
2. Each patch becomes a "vision token"
3. ViT processes these tokens with self-attention
4. We **pool** (average) across all vision tokens
5. Result: Single 3584-dim vector representing the entire image

**Key Code:**
```python
# Line 232-233: Pass through vision encoder
vision_outputs = self.model.visual(pixel_values, grid_thw=grid_thw)

# Line 249-252: Pool over sequence dimension
if vision_features.dim() == 3:
    vision_rep = vision_features.mean(dim=1)  # [B, seq, hidden] -> [B, hidden]
```

**Extraction Point:** ⭐ **PRE-FUSION** - Before vision and text interact

---

### 2️⃣ **Vision Token Representation** (At Image-Text Boundary)

**Purpose:** Capture how vision information is represented at the boundary where image tokens end and text begins

**Code Location:** `_extract_decoder_embeddings()` (lines 264-330)

**Architecture Path:**
```python
model.model.language_model.layers[i]
```

**Detailed Flow:**

```
Forward Pass Through Language Model:
────────────────────────────────────

Input Sequence: [vision_tokens | text_tokens]
                 ↑              ↑
                 701 tokens     remaining
                 (from image)   (from text)

Layer 0 (Decoder Layer)
    Input:  [1, 1402, 3584]  (full sequence)
    ↓ Self-Attention + FFN
    Output: [1, 1402, 3584]
           ↓
    EXTRACT at position 701: hidden_states[0, 701, :] → [3584]
    ↑
    This is the LAST VISION TOKEN representation

Layer 7 (n/4)
    Same process...
    EXTRACT at position 701

Layer 14 (n/2)
    Same process...
    EXTRACT at position 701

Layer 21 (3n/4)
    Same process...
    EXTRACT at position 701

Layer 27 (last)
    Same process...
    EXTRACT at position 701
```

**What's Happening:**

1. **Token Sequence Construction:**
   ```
   [<vision_token_0>, <vision_token_1>, ..., <vision_token_700>,
    <text_token_0>, <text_token_1>, ..., <text_token_N>]
   ```

2. **Why position 701?**
   - Image is encoded into 701 vision tokens (varies by image size)
   - Position 701 is the **boundary** between vision and text
   - It's the "last representation of visual information" before text takes over

3. **Layer-by-layer transformation:**
   - **Layer 0:** Raw combined representation (vision + text minimally processed)
   - **Layer 7:** Early fusion starting
   - **Layer 14:** Mid-level fusion (vision-text interaction)
   - **Layer 21:** Late fusion (high-level semantic integration)
   - **Layer 27:** Final representation (fully integrated)

**Key Code:**
```python
# Line 291: Access language model layers
language_layers = self.model.model.language_model.layers

# Line 294-298: Register hooks on specific layers
for layer_idx in target_layers:  # [0, 7, 14, 21, 27]
    hook = language_layers[layer_idx].register_forward_hook(
        create_hook(layer_idx)
    )

# Line 318: Extract at vision boundary
if image_boundary < hidden_states.shape[1]:
    vision_token_reps[f'layer_{layer_idx}'] = hidden_states[0, image_boundary, :].cpu()
```

**Token Boundary Detection:**
```python
# Line 346-351: From image_grid_thw
grid = inputs['image_grid_thw']  # e.g., [1, 23, 23]
vision_token_count = grid[0].prod().item()  # 1 * 23 * 23 = 529
vision_token_boundary = vision_token_count
```

**Extraction Point:** ⭐ **POST-FUSION** - After vision-language interaction in each layer

---

### 3️⃣ **Query Token Representation** (At Question End)

**Purpose:** Capture the complete question understanding at the point where the model is about to generate the answer

**Code Location:** `_extract_decoder_embeddings()` (lines 264-330)

**Architecture Path:**
```python
model.model.language_model.layers[i]
```

**Detailed Flow:**

```
Forward Pass Through Language Model:
────────────────────────────────────

Input Sequence: [vision_tokens | question_tokens]
                                  ↑
                                  position 1402 (last token)

Layer 0 (Decoder Layer)
    Input:  [1, 1402, 3584]
    ↓ Self-Attention + FFN
    Output: [1, 1402, 3584]
           ↓
    EXTRACT at position 1402: hidden_states[0, 1402, :] → [3584]
    ↑
    This is the representation at the END OF QUESTION
    (before answer generation starts)

[Same for layers 7, 14, 21, 27]
```

**What's Happening:**

1. **Token Sequence:**
   ```
   Position:  0...700  |  701...1402
   Content:   [vision] | [question: "Is the statue on water or platform?"]
                                                                          ↑
                                                                    query_end
   ```

2. **Why position 1402?**
   - It's the **last token of the question**
   - Represents the model's complete understanding of:
     - The image (from vision tokens)
     - The question (from text tokens)
   - This is the "thought" right before generating the answer

3. **Layer-wise evolution:**
   - **Layer 0:** Basic understanding (vision + question text)
   - **Layer 7:** Forming multimodal understanding
   - **Layer 14:** Reasoning about the question wrt image
   - **Layer 21:** High-level semantic integration
   - **Layer 27:** Final representation used for answer generation

**Key Code:**
```python
# Line 320-322: Extract at query boundary
if query_boundary < hidden_states.shape[1]:
    query_token_reps[f'layer_{layer_idx}'] = hidden_states[0, query_boundary, :].cpu()
```

**Boundary Detection:**
```python
# Line 355: Query boundary is last token
seq_len = input_ids.shape[1]  # Total sequence length
query_token_boundary = seq_len - 1  # Last token position
```

**Extraction Point:** ⭐ **PRE-GENERATION** - Right before the model generates the answer

---

## 🔄 Complete Processing Flow

### Step-by-Step Execution

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INPUT PREPARATION                                             │
└─────────────────────────────────────────────────────────────────┘

Image: haloquest_1236.png
Question: "Is the forest lit by a full moonlight?"

    ↓ Processor

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text", "text": "Is the forest lit..."}
        ]
    }
]

    ↓ apply_chat_template + process_vision_info

Outputs:
- pixel_values: [1, 3, H, W]        (image tensor)
- image_grid_thw: [1, 1, 23, 23]    (grid dimensions)
- input_ids: [1, 1402]              (tokenized text)
- attention_mask: [1, 1402]


┌─────────────────────────────────────────────────────────────────┐
│ 2. VISION-ONLY EXTRACTION                                        │
└─────────────────────────────────────────────────────────────────┘

    pixel_values ──→ model.visual(pixel_values, grid_thw)
                           │
                           ├─ Patch Embedding
                           ├─ ViT Self-Attention Layers
                           └─ Output: [1, 529, 3584]
                                        ↓ mean(dim=1)
                                   vision_only_rep: [3584] ✓


┌─────────────────────────────────────────────────────────────────┐
│ 3. ANSWER GENERATION (with hooks active)                         │
└─────────────────────────────────────────────────────────────────┘

inputs = {
    'input_ids': [1, 1402],
    'pixel_values': [1, 3, H, W],
    'image_grid_thw': [1, 1, 23, 23],
    'attention_mask': [1, 1402]
}

    ↓ model.generate(**inputs)

Internal Forward Pass:
────────────────────

1. Vision Encoding:
   model.visual → vision_embeds: [1, 529, 3584]

2. Text Embedding:
   model.model.language_model.embed_tokens → text_embeds: [1, 873, 3584]

3. Sequence Concatenation:
   combined = [vision_embeds | text_embeds]
   shape: [1, 1402, 3584]
          ↑    ↑     ↑
          │    │     └─ hidden dimension
          │    └─ 529 vision + 873 text tokens
          └─ batch size

4. Through Decoder Layers:

   Layer 0:  [1, 1402, 3584]
        ↓ Self-Attention (vision + text attend to each other)
        ↓ Feed-Forward
       [1, 1402, 3584]
        │
        ├─ HOOK captures output
        ├─ Extract [0, 529, :] → vision_token_rep_layer_0 ✓
        └─ Extract [0, 1402, :] → query_token_rep_layer_0 ✓

   Layer 7:  [1, 1402, 3584]
        ↓ Self-Attention
        ↓ Feed-Forward
       [1, 1402, 3584]
        │
        ├─ HOOK captures output
        ├─ Extract [0, 529, :] → vision_token_rep_layer_7 ✓
        └─ Extract [0, 1402, :] → query_token_rep_layer_7 ✓

   [... Layer 14, 21, 27 same process ...]

5. Final Layer Output → LM Head → Logits → Generated Text


┌─────────────────────────────────────────────────────────────────┐
│ 4. SAVED TO HDF5                                                 │
└─────────────────────────────────────────────────────────────────┘

question_comb_6253/
├── question: "Is the statue on water or platform?"
├── image_id: "haloquest_2016.png"
├── ground_truth_answer: "On a platform"
├── answer: "The statue is standing on a platform..."
│
├── vision_only_representation: [3584]          ← From model.visual
│
├── vision_token_representation/
│   ├── layer_0:  [3584]  ← From decoder layer 0, pos 529
│   ├── layer_7:  [3584]  ← From decoder layer 7, pos 529
│   ├── layer_14: [3584]  ← From decoder layer 14, pos 529
│   ├── layer_21: [3584]  ← From decoder layer 21, pos 529
│   └── layer_27: [3584]  ← From decoder layer 27, pos 529
│
└── query_token_representation/
    ├── layer_0:  [3584]  ← From decoder layer 0, pos 1402
    ├── layer_7:  [3584]  ← From decoder layer 7, pos 1402
    ├── layer_14: [3584]  ← From decoder layer 14, pos 1402
    ├── layer_21: [3584]  ← From decoder layer 21, pos 1402
    └── layer_27: [3584]  ← From decoder layer 27, pos 1402
```

---

## 🧠 Why These Specific Extraction Points?

### **Vision-Only Representation**
- **What:** Pure visual encoding without text influence
- **Why:** Establishes baseline of what the model "sees"
- **HALP Use:** Detect if hallucinations stem from vision misunderstanding

### **Vision Token Representation (at boundary)**
- **What:** How vision is represented AFTER text interaction
- **Why:** Shows how visual info transforms through reasoning layers
- **HALP Use:** Detect if vision info is being "overwritten" or "confused" by text

### **Query Token Representation (at end)**
- **What:** Complete multimodal understanding before generation
- **Why:** Captures the model's "final thought" before answering
- **HALP Use:** Predict hallucination by analyzing this pre-answer state

### **5 Strategic Layers (0, 7, 14, 21, 27)**
- **Layer 0:** Initial fusion
- **Layer n/4 (7):** Early reasoning
- **Layer n/2 (14):** Mid-level integration
- **Layer 3n/4 (21):** Advanced reasoning
- **Layer n-1 (27):** Final representation

**Why not all 28 layers?**
- Storage efficiency (5 layers vs 28 = 5.6x less data)
- Captures key transition points in reasoning
- Research shows intermediate layers most informative

---

## 📐 Dimensionality Summary

| Embedding Type | Shape | Source | Extraction Point |
|----------------|-------|--------|------------------|
| Vision-Only | `[3584]` | `model.visual` | After ViT, before LM |
| Vision Token (×5) | `[3584]` each | `language_model.layers[i]` | Position: last_image_token |
| Query Token (×5) | `[3584]` each | `language_model.layers[i]` | Position: last_query_token |

**Total per sample:** 1 + 5 + 5 = **11 embeddings** of dimension 3584

---

## 🎯 HALP (Hallucination Prediction) Strategy

```
Vision-Only → "What the model sees"
       ↓
Vision Token (layer progression) → "How vision info evolves during reasoning"
       ↓
Query Token (layer progression) → "Complete understanding before answer"
       ↓
Train Classifier → Predict if answer will be hallucinated
```

**Key Insight:** By comparing:
1. What the model initially sees (vision-only)
2. How that vision transforms through layers (vision-token)
3. What the model "thinks" before answering (query-token)

We can detect patterns that correlate with hallucinations!

---

## 🔧 Technical Implementation Details

### Hook Mechanism
```python
def create_hook(layer_idx):
    def hook(module, input, output):
        # Capture output of this layer
        layer_outputs[layer_idx] = output[0].detach()
    return hook

# Register on specific layers
for layer_idx in [0, 7, 14, 21, 27]:
    hook = language_layers[layer_idx].register_forward_hook(create_hook(layer_idx))
```

**How hooks work:**
1. Hooks are "callback functions" that PyTorch calls during forward pass
2. When data flows through `language_layers[i]`, our hook captures the output
3. We store it in `layer_outputs[i]`
4. After forward pass, we extract specific token positions from these captured outputs

### Token Boundary Detection
```python
# From image_grid_thw tensor
grid = inputs['image_grid_thw']  # Shape: [batch, 3] = [t, h, w]
vision_tokens = grid[0].prod().item()  # t × h × w = total vision tokens

# Vision boundary: right after vision tokens
vision_boundary = vision_tokens

# Query boundary: last token in sequence
query_boundary = input_ids.shape[1] - 1
```

### Memory Management
```python
# After each sample:
torch.cuda.empty_cache()  # Free GPU memory

# Hooks are removed after each extraction:
for hook in hooks:
    hook.remove()
```

---

## 📊 Real Example from Test Output

```yaml
Question: "Is the forest lit by a full moonlight?"
Image: haloquest_1236.png

Token Breakdown:
  - Vision tokens: 0-700 (701 tokens from 1×23×23 grid + padding)
  - Text tokens: 701-1402 (702 tokens for the question)
  - Vision boundary: 701
  - Query boundary: 1402

Extracted Embeddings:
  vision_only_representation:
    shape: (1452, 3584)  # Raw ViT output (all vision tokens)
    pooled: (3584,)       # Mean-pooled to single vector

  vision_token_representation:
    layer_0: (3584,)   # hidden_states[0, 701, :]
    layer_7: (3584,)   # hidden_states[0, 701, :]
    layer_14: (3584,)  # hidden_states[0, 701, :]
    layer_21: (3584,)  # hidden_states[0, 701, :]
    layer_27: (3584,)  # hidden_states[0, 701, :]

  query_token_representation:
    layer_0: (3584,)   # hidden_states[0, 1402, :]
    layer_7: (3584,)   # hidden_states[0, 1402, :]
    layer_14: (3584,)  # hidden_states[0, 1402, :]
    layer_21: (3584,)  # hidden_states[0, 1402, :]
    layer_27: (3584,)  # hidden_states[0, 1402, :]

Generated Answer: "The image depicts a magical forest scene with glowing mushrooms..."
Ground Truth: "No; The forest isn't brightened by full moonlight"
```

---

## 🎓 Summary

**We extract embeddings from 3 critical points in Qwen2.5-VL's processing pipeline:**

1. **Vision Encoder Output** → Pure visual understanding
2. **Decoder Layers at Vision Boundary** → Vision-text fusion evolution
3. **Decoder Layers at Query End** → Complete multimodal reasoning

**These embeddings capture the model's internal reasoning process, enabling us to predict hallucinations BEFORE the model generates incorrect answers.**
