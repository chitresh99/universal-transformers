# Universal Transformers

**Paper:** Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Lukasz Kaiser. *Universal Transformers.* ICLR 2019.
**arXiv:** [1807.03819](https://arxiv.org/abs/1807.03819) (v1 July 2018, v3 March 2019)
**Affiliations:** Google Brain, DeepMind, University of Amsterdam
**Code:** released in [tensor2tensor](https://github.com/tensorflow/tensor2tensor)

This document explains the paper with intuition first and technical detail after. Everything marked "paper says" is grounded in the paper text. Anything the paper does not state is labeled as such, so it is clear where the paper ends and interpretation begins.

---

## Table of contents

1. [One paragraph summary](#1-one-paragraph-summary)
2. [Intuition first](#2-intuition-first)
3. [The problem the paper is attacking](#3-the-problem-the-paper-is-attacking)
4. [What the paper builds on](#4-what-the-paper-builds-on)
5. [Architecture](#5-architecture)
6. [The math](#6-the-math)
7. [Tensor shapes through the whole model](#7-tensor-shapes-through-the-whole-model)
8. [Dynamic halting (ACT) in detail](#8-dynamic-halting-act-in-detail)
9. [RNN vs LSTM vs Transformer vs Universal Transformer](#9-rnn-vs-lstm-vs-transformer-vs-universal-transformer)
10. [Transformer vs Universal Transformer, precisely](#10-transformer-vs-universal-transformer-precisely)
11. [Why it is called "Universal"](#11-why-it-is-called-universal)
12. [Training details](#12-training-details)
13. [Inference details](#13-inference-details)
14. [Datasets, experiments, and results](#14-datasets-experiments-and-results)
15. [Argued-for choices vs control variables](#15-argued-for-choices-vs-control-variables)
16. [Weakest links](#16-weakest-links)
17. [Why this paper did not make the same noise](#17-why-this-paper-did-not-make-the-same-noise)
18. [What is used today from this paper](#18-what-is-used-today-from-this-paper)
19. [Interview explanations](#19-interview-explanations)
20. [Questions to test your understanding](#20-questions-to-test-your-understanding)
21. [Implementation notes and pitfalls](#21-implementation-notes-and-pitfalls)
22. [Glossary](#22-glossary)
23. [References](#23-references)

---

## 1. One paragraph summary

A standard Transformer stacks a fixed number of layers (for example 6), and each layer has its own weights. The Universal Transformer (UT) keeps one Transformer layer and applies it repeatedly, with the same weights every time, so every token's representation is refined over several passes. This turns the Transformer into a recurrent model where the recurrence runs over depth (revisions of the representation), not over sequence positions. A timestep embedding tells the shared layer which pass it is on. An optional per-token halting mechanism (Adaptive Computation Time) lets each token decide how many passes it needs. The paper argues this adds the inductive bias of RNNs (learning repeatable procedures) to the Transformer, makes the model computationally universal under certain assumptions, and shows gains on algorithmic tasks, bAbI, subject-verb agreement, LAMBADA, and WMT14 English-German translation.

---

## 2. Intuition first

### 2.1 Assembly line vs repeated editing

**Transformer:** an assembly line with 6 different workers. Each token passes station 1, then 2, up to 6. Every worker has their own tools (weights). There are always exactly 6 stations.

**Universal Transformer:** one editor revising the same draft several times. The editor reads the whole draft, improves every word at once, then reads the improved draft and improves it again. Same skill set on every pass. You can do 4 passes or 12.

**With dynamic halting:** words that are already clear stop being edited early. Confusing words keep getting revised.

### 2.2 Each pass has two parts

1. **Talk to everyone (self-attention).** Every token looks at every other token's current state and gathers what it needs.
2. **Think privately (transition function).** Every token processes what it gathered, independently, with a small network.

Nothing is processed in isolation, because step 1 mixes information across all positions on every pass.

### 2.3 Where the RNN is

Look at one token across passes: `word^0 -> word^1 -> word^2 -> word^3`. That is a hidden state updated repeatedly by the same function, which is exactly an RNN. The UT is therefore a bank of RNNs, one per token, with shared weights, running in parallel, and they read each other's states through attention at every tick.

| | Recurrence axis | Example |
|---|---|---|
| RNN | Across positions | read word 1, then word 2, then word 3 |
| Universal Transformer | Across depth (revisions) | refine all words, then refine all words again |

### 2.4 A concrete example

Sentence: "The keys to the cabinet are ..."

| Pass | What plausibly happens |
|---|---|
| 1 | "keys" picks up that it is a plural noun near "cabinet" |
| 2 | "are" attends to richer versions of every word and starts linking to "keys" instead of "cabinet" |
| 3 | The link sharpens; "the" may already have halted |

This is an illustration of the idea, not a trace from the paper. The paper does show something similar on bAbI, where attention starts nearly uniform and gets sharper on the relevant facts over later steps.

### 2.5 The one-line difference

A Transformer does not share weights across layers. A Universal Transformer does. Everything else in the paper follows from that.

---

## 3. The problem the paper is attacking

**Paper says:**

1. RNNs process tokens sequentially, which makes them slow to train.
2. Transformers process all tokens in parallel and have a global receptive field, and they beat RNNs on machine translation.
3. But Transformers fail on simple tasks that recurrent models handle, such as copying strings or simple logical inference, especially when test inputs are longer than training inputs.
4. The authors attribute this to the Transformer's fixed stack of distinct layers giving up the RNN's inductive bias toward iterative or recursive transformations.

**The gap:** no model at the time combined (a) Transformer-style parallelism and global attention, (b) an RNN-style bias toward repeated procedures, and (c) competitive results on real language tasks. Neural Turing Machines and Neural GPUs could learn algorithms but were not competitive on natural language.

---

## 4. What the paper builds on

| Source | What UT takes from it |
|---|---|
| Transformer (Vaswani et al., 2017) | Multi-head scaled dot-product attention, residual connections, layer norm, sinusoidal encodings, encoder-decoder layout, autoregressive decoding, MT training setup |
| RNNs (general) | Applying the same function repeatedly to an evolving state |
| Adaptive Computation Time (Graves, 2016) | Learned per-step halting probability, so the number of steps is decided by the model |
| Depthwise separable convolutions (Chollet, 2016; Kaiser et al., 2017) | One of the two transition function options |
| Neural GPU (Kaiser and Sutskever, 2016) | The universality argument: UT reduces to a Neural GPU under specific settings |
| Neural Turing Machine (Graves et al., 2014) | Related model with memory access in the recurrent step |
| Seq2seq (Sutskever et al., 2014; Cho et al., 2014) | Encoder-decoder framing |
| Bahdanau attention (2014) | Encoder-decoder cross attention is "akin to" it |

**Models used as components:** multi-head self-attention, a position-wise feed-forward network or a depthwise separable convolution as the transition, layer norm, dropout, ACT.

**Models used as baselines:** LSTM, standard Transformer, Weighted Transformer, Neural GPU, Stack-RNN, attention LSTM, and per-task prior state of the art (QRNet, Sparse DNC, GA+MAGE, MemN2N on bAbI; Neural Cache and Dhingra et al. on LAMBADA).

---

## 5. Architecture

### 5.1 Encoder

1. Embed the input tokens to get `H^0`.
2. For `t = 1 .. T`:
   1. Add the position-and-timestep embedding `P^t` to the current state.
   2. Multi-head self-attention, then residual, dropout, layer norm. Output `A^t`.
   3. Transition function (FFN or separable conv), then residual, dropout, layer norm. Output `H^t`.
3. Output `H^T`.

The same attention weights and the same transition weights are used at every step `t`.

### 5.2 Decoder

Same recurrent structure, with three sub-blocks per step instead of two:

1. Masked multi-head self-attention (a position can only see positions to its left).
2. Multi-head cross attention: queries from the decoder state, keys and values from the final encoder output `H^T`.
3. Transition function.

Each sub-block has residual, dropout, and layer norm (paper Figure 4). After `T` steps, a linear projection to the vocabulary and a softmax give next-token probabilities.

### 5.3 Transition function

Paper says: depending on the task, either

- a **position-wise fully connected network**: affine, ReLU, affine, applied to each position separately, or
- a **depthwise separable convolution**.

The MT result uses the fully connected version. The paper does not give a systematic ablation of which transition to use per task, and it does not report kernel sizes or hidden sizes in the main text.

### 5.4 Variants in the paper

| Variant | Steps |
|---|---|
| UT (fixed) | Same fixed `T` for every token |
| UT with dynamic halting | Each token halts on its own via ACT, up to a maximum number of steps |

### 5.5 Layer norm placement

The equations are **post-LN**: `LayerNorm(x + Sublayer(x))`, as in the original Transformer. This matters for implementation because many modern Transformer and looped-model implementations use pre-LN for stability.

---

## 6. The math

Notation from the paper: `m` is sequence length, `d` is model dimension, `k` is number of heads. The paper uses row-vector convention (`H W`).

### 6.1 Scaled dot-product attention (Eq. 1)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

The paper says `d` here is the number of columns of `Q`, `K`, `V`. Inside a head that is `d/k`, not the full model dimension.

### 6.2 Multi-head self-attention (Eqs. 2 and 3)

$$
\text{MultiHeadSelfAttention}(H^t) = \text{Concat}(\text{head}_1, \dots, \text{head}_k)\, W^O
$$

$$
\text{head}_i = \text{Attention}(H^t W^Q_i,\; H^t W^K_i,\; H^t W^V_i)
$$

with $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d \times d/k}$ and $W^O \in \mathbb{R}^{d \times d}$.

### 6.3 One recurrent step (Eqs. 4 and 5)

$$
A^t = \text{LayerNorm}\big((H^{t-1} + P^t) + \text{MultiHeadSelfAttention}(H^{t-1} + P^t)\big)
$$

$$
H^t = \text{LayerNorm}\big(A^t + \text{Transition}(A^t)\big)
$$

Note that `P^t` is added before attention **and** is part of the residual path.

### 6.4 Fully connected transition

$$
\text{Transition}(x) = \max(0,\; x W_1 + b_1)\, W_2 + b_2
$$

applied independently to each row (position) of `A^t`.

### 6.5 Position and timestep embeddings (Eqs. 6 and 7)

For position `i` (1 to `m`), step `t` (1 to `T`), dimension index `j`:

$$
P^t_{i,2j} = \sin\left(\frac{i}{10000^{2j/d}}\right) + \sin\left(\frac{t}{10000^{2j/d}}\right)
$$

$$
P^t_{i,2j+1} = \cos\left(\frac{i}{10000^{2j/d}}\right) + \cos\left(\frac{t}{10000^{2j/d}}\right)
$$

This is the Transformer's sinusoidal encoding computed twice (once for position, once for step) and summed. It is fixed, not learned. Its job: since the weights are shared, the layer needs some signal telling it which pass it is on.

### 6.6 Decoder cross attention

Same as Eq. 2, but `Q` comes from the decoder state and `K`, `V` come from the encoder output `H^T`.

### 6.7 Output distribution (Eq. 8)

$$
p\big(y_{pos} \mid y_{[1:pos-1]}, H^T\big) = \text{softmax}(O H^T)
$$

with $O \in \mathbb{R}^{d \times V}$. Here `H^T` is the final **decoder** state, and `T` is the step count, not a transpose (paper footnote 1).

### 6.8 ACT halting probability (from Appendix C)

At each step, per position:

$$
p^t_i = \sigma(w^\top h^t_i + b)
$$

Accumulated halting probability grows until it crosses a threshold; the leftover mass (remainder) is used as the final step's weight. Full walkthrough in [Section 8](#8-dynamic-halting-act-in-detail).

### 6.9 Compute cost

Per step, self-attention costs $O(m^2 d)$ and the transition costs $O(m\, d\, d_{ff})$. Over `T` steps:

$$
\text{cost} \approx T \cdot \big(O(m^2 d) + O(m\, d\, d_{ff})\big)
$$

Parameters do **not** grow with `T`. Compute does.

---

## 7. Tensor shapes through the whole model

The paper gives matrix shapes without a batch dimension and without concrete sizes. The table below adds a batch dimension and uses **illustrative** sizes to make shapes concrete. These sizes are not the paper's UT configuration, which the paper text does not report.

| Symbol | Meaning | Illustrative value |
|---|---|---|
| `B` | batch size | 32 |
| `m` | source length | 20 |
| `n` | target length | 22 |
| `d` | model dim | 512 |
| `k` | heads | 8 |
| `d_h` | head dim = `d/k` | 64 |
| `d_ff` | FFN hidden dim | 2048 |
| `V` | vocab size | 32000 |
| `T` | recurrent steps | 6 |

### 7.1 Encoder, one step

| Stage | Shape (symbolic) | Shape (illustrative) |
|---|---|---|
| Input token ids | `B x m` | `32 x 20` |
| Embeddings `H^0` | `B x m x d` | `32 x 20 x 512` |
| `P^t` (broadcast over batch) | `m x d` | `20 x 512` |
| `X = H^{t-1} + P^t` | `B x m x d` | `32 x 20 x 512` |
| `W^Q_i`, `W^K_i`, `W^V_i` | `d x d_h` | `512 x 64` |
| `Q`, `K`, `V` (heads stacked) | `B x k x m x d_h` | `32 x 8 x 20 x 64` |
| Scores `QK^T / sqrt(d_h)` | `B x k x m x m` | `32 x 8 x 20 x 20` |
| Softmax weights | `B x k x m x m` | `32 x 8 x 20 x 20` |
| Weighted values | `B x k x m x d_h` | `32 x 8 x 20 x 64` |
| Concat heads | `B x m x d` | `32 x 20 x 512` |
| After `W^O` (`d x d`) | `B x m x d` | `32 x 20 x 512` |
| `A^t` after residual + LN | `B x m x d` | `32 x 20 x 512` |
| FFN first layer (`W_1`: `d x d_ff`) | `B x m x d_ff` | `32 x 20 x 2048` |
| FFN second layer (`W_2`: `d_ff x d`) | `B x m x d` | `32 x 20 x 512` |
| `H^t` after residual + LN | `B x m x d` | `32 x 20 x 512` |

After `T` steps: `H^T` has shape `B x m x d`. **The shape never changes across steps.** This is what makes recurrence over depth possible: output of step `t` is a valid input to step `t+1`.

### 7.2 Decoder, one step

| Stage | Shape (symbolic) | Shape (illustrative) |
|---|---|---|
| Target ids, right-shifted | `B x n` | `32 x 22` |
| Target embeddings | `B x n x d` | `32 x 22 x 512` |
| Causal mask | `n x n` | `22 x 22` |
| Self-attention scores | `B x k x n x n` | `32 x 8 x 22 x 22` |
| Self-attention output | `B x n x d` | `32 x 22 x 512` |
| Cross attention `Q` (from decoder) | `B x k x n x d_h` | `32 x 8 x 22 x 64` |
| Cross attention `K`, `V` (from `H^T_enc`) | `B x k x m x d_h` | `32 x 8 x 20 x 64` |
| Cross attention scores | `B x k x n x m` | `32 x 8 x 22 x 20` |
| Cross attention output | `B x n x d` | `32 x 22 x 512` |
| Transition output | `B x n x d` | `32 x 22 x 512` |

### 7.3 Output head

| Stage | Shape (symbolic) | Shape (illustrative) |
|---|---|---|
| Final decoder state | `B x n x d` | `32 x 22 x 512` |
| `O` | `d x V` | `512 x 32000` |
| Logits | `B x n x V` | `32 x 22 x 32000` |
| Softmax over last dim | `B x n x V` | `32 x 22 x 32000` |

The paper writes the output matrix as `m x V`. For the decoder this should be target length, so `n x V`. Minor notation slip.

### 7.4 ACT state (Appendix C, encoder)

| Tensor | Shape | Meaning |
|---|---|---|
| `p` (halting prob this step) | `B x m x 1` | dense layer to 1 unit, sigmoid |
| `halting_probability` | `B x m` | cumulative halting mass |
| `remainders` | `B x m` | leftover mass at the halting step |
| `n_updates` | `B x m` | how many steps this position used |
| `still_running`, `new_halted` | `B x m` | 0/1 masks |
| `update_weights` | `B x m x 1` | interpolation weight for the new state |
| `transformed_state`, `new_state` | `B x m x d` | state tensors |

Shapes mix `B x m` and `B x m x 1` in the listing. In a from-scratch implementation, pick one convention and be explicit about broadcasting.

### 7.5 bAbI input encoding (Section 3.1)

Paper says: each fact is encoded by applying a learned multiplicative positional mask to each word embedding and summing; the question is encoded the same way; the UT consumes the fact and question vectors.

| Stage | Shape |
|---|---|
| Story word ids (`f` facts, `L` words max) | `B x f x L` |
| Word embeddings | `B x f x L x d` |
| Positional mask | `L x d` |
| Masked and summed over words | `B x f x d` |
| Question vector | `B x 1 x d` |
| UT input sequence | `B x (f+1) x d` |

### 7.6 Parameter count of one shared layer

Using the illustrative sizes, ignoring biases and layer norm:

| Part | Formula | Count |
|---|---|---|
| Attention projections | `4 d^2` | 1,048,576 |
| FFN | `2 d d_ff` | 2,097,152 |
| One encoder layer | | about 3.15M |
| 6-layer Transformer encoder | `6 x 3.15M` | about 18.9M |
| UT encoder, any `T`, same `d` | `1 x 3.15M` | about 3.15M |

So a UT with the same width has roughly `1/T` the layer parameters of a `T`-layer Transformer, and the same compute. The paper's MT comparison instead matches **parameters**, which means the UT must be wider. See [Weakest links](#16-weakest-links) for why that matters.

### 7.7 Decoder KV cache at inference (interpretation, not in paper)

- **Self-attention cache:** one cache per recurrent step, like a `T`-layer model: `T x B x k x n x d_h` for keys and the same for values.
- **Cross attention cache:** the encoder output `H^T` is fixed during decoding, and the cross attention projection weights are shared across steps. So if implemented exactly as described, cross attention keys and values are **identical at every step** and can be computed once: `B x k x m x d_h`, not `T` copies. A standard Transformer cannot do this because each layer has different cross attention weights.
- **With ACT in the decoder:** a position that halted at step 3 has no fresh step 5 state. The cache has to reuse the copied state, which complicates batching.

---

## 8. Dynamic halting (ACT) in detail

### 8.1 Intuition

Some tokens need more thinking than others. Each token gets a "done?" meter. Every step adds a bit to the meter. When the meter passes a threshold, the token stops updating. Its last state is copied forward, and other tokens can still attend to it.

### 8.2 Algorithm as written in Appendix C

State carried per position: `halting_probability`, `remainders`, `n_updates`, `previous_state`, all initialized to zero. A scalar `threshold` in (0, 1) is a hyperparameter.

Per step:

1. Compute `p = sigmoid(dense(state))`.
2. `still_running = halting_probability < 1.0`.
3. `new_halted = (halting_probability + p * still_running > threshold) * still_running`.
4. `still_running = (halting_probability + p * still_running <= threshold) * still_running`.
5. `halting_probability += p * still_running`.
6. `remainders += new_halted * (1 - halting_probability)`.
7. `halting_probability += new_halted * remainders` (this makes it exactly 1.0 for halted positions).
8. `n_updates += still_running + new_halted`.
9. `update_weights = p * still_running + new_halted * remainders` (0 for already halted positions).
10. `transformed_state = transition(self_attention(state))`.
11. `new_state = transformed_state * update_weights + previous_state * (1 - update_weights)`.

Loop until no position is still running or `max_steps` is reached.

### 8.3 Worked example for one token

Threshold = 0.99 (illustrative).

| Step | `p` | Cum before | Cum + p | Status | `update_weight` | `new_state` | Cum after |
|---|---|---|---|---|---|---|---|
| 1 | 0.30 | 0.00 | 0.30 | running | 0.30 | 0.30 S1 + 0.70 prev | 0.30 |
| 2 | 0.50 | 0.30 | 0.80 | running | 0.50 | 0.50 S2 + 0.50 prev | 0.80 |
| 3 | 0.40 | 0.80 | 1.20 | halts (>0.99) | remainder = 0.20 | 0.20 S3 + 0.80 prev | 1.00 |
| 4 | any | 1.00 | | halted | 0.00 | copy prev | 1.00 |

`n_updates = 3`. Here `S_t` is the transformed state at step `t` and `prev` is the previous `new_state`.

### 8.4 Things the listing does not make obvious

- **The ponder cost.** Graves's ACT adds a penalty on the number of steps so the model does not always use the maximum. The paper text does not describe this term. Check the tensor2tensor hyperparameters for an ACT loss weight before assuming it is absent.
- **Which state feeds the next step.** Listing 1 passes the returned `transformed_state` as the next `state`, and `new_state` as `previous_state`. Read literally, attention at the next step operates on the transformed (not interpolated) state, and halted positions still have their `transformed_state` recomputed, while their output `new_state` is frozen. The prose says halted states are "simply copied." Verify against the tensor2tensor source before implementing, because this changes what other tokens attend to.
- **Where `P^t`, dropout, and layer norm go.** The listing collapses these into `transition(self_attention(state))`.

---

## 9. RNN vs LSTM vs Transformer vs Universal Transformer

| Property | RNN | LSTM | Transformer | Universal Transformer |
|---|---|---|---|---|
| How tokens are read | One at a time | One at a time | All at once | All at once |
| Token-to-token path | Through hidden state chain | Through cell/hidden state chain | Direct, one attention hop | Direct, one attention hop per step |
| Memory in the recurrent step | One fixed-size vector | Cell + hidden vectors | Not recurrent | Every position's state (attention over whole previous step) |
| Recurrence axis | Positions | Positions | None | Depth (revisions) |
| Weight sharing across positions | Yes | Yes | Yes | Yes |
| Weight sharing across depth | N/A (one step type) | N/A | No | Yes |
| Sequential steps | Grows with sequence length | Grows with sequence length | Fixed by number of layers | `T`, can be chosen or learned per token |
| Parallel training across sequence | No | No | Yes | Yes |
| Long-range dependencies | Hard (vanishing gradients) | Better, still hard | Easy | Easy |
| Cost vs sequence length | Linear | Linear | Quadratic | Quadratic, times `T` |
| Order information | Built in | Built in | Position embeddings | Position and timestep embeddings |
| Adaptive computation | Possible with ACT | Possible with ACT | No | Yes, per token |

**The crucial contrast the paper draws (Section 4):** a time-recurrent RNN only sees a fixed-size state vector inside its recurrent step, which makes it computationally closer to a finite automaton. The UT's recurrent step can attend to the entire previous step, so it has real memory access inside the loop.

---

## 10. Transformer vs Universal Transformer, precisely

| Aspect | Transformer | Universal Transformer |
|---|---|---|
| Layers | `N` distinct layers | 1 layer applied `T` times |
| Weights across depth | Different per layer | Shared (both attention and transition) |
| Depth at inference | Fixed at `N` | Any `T`; per token with ACT |
| Depth signal | Implicit (each layer is different) | Explicit timestep embedding `P^t` added every step |
| Position encoding | Added once at the input | Added (with timestep) at every step |
| Transition | FFN | FFN or depthwise separable conv |
| Parameters vs compute | Both grow with `N` | Parameters fixed; compute grows with `T` |
| Universality (paper's framing) | Not universal with finite precision and fixed depth | Universal if `T` can grow with input and memory suffices |
| If `T = 1` | | Reduces to a 1-layer Transformer (plus timestep embedding) |
| If `T` fixed | | Equivalent to an `T`-layer Transformer with tied weights |

---

## 11. Why it is called "Universal"

### 11.1 The argument (Section 4 and Appendix B)

1. A standard Transformer does many operations in total, but its number of **sequential** steps is fixed by its number of layers, independent of input size.
2. Assuming finite precision, this means it cannot be computationally universal. For any depth `T`, you can build an input of length `N > T` for a function that needs one sequential step per input element, and the Transformer cannot compute it.
3. A UT's weights are shared, so its number of steps is not tied to its parameters. Choose `T` as a function of input length and this limit disappears. This holds with or without ACT, as long as the number of steps is not constant.

### 11.2 The reduction to a Neural GPU

1. Ignore the decoder.
2. Set self-attention (with its residual) to the identity.
3. Use a convolution as the transition.
4. Set `T` equal to the input length.

The result is exactly a Neural GPU, which is known to be universal. A similar correspondence is sketched with the Neural Turing Machine.

### 11.3 Caveats

- This is a construction sketch, not a formal proof in the paper.
- It depends on `T` growing with input and on sufficient memory. In practice the experiments cap steps at a maximum.
- Expressivity is not learnability. The model being able to represent an algorithm does not mean gradient descent finds it.
- Universality claims for standard Transformers depend on assumptions. Perez et al. (ICLR 2019) argue Transformers can be Turing complete under arbitrary precision with unbounded decoding. The UT paper's claim is specifically under finite precision with fixed depth.

---

## 12. Training details

### 12.1 What the paper says

- **Framework:** tensor2tensor (TensorFlow).
- **Objective:** autoregressive next-token prediction with teacher forcing. Decoder input is the target shifted right by one. Decoder self-attention is causally masked.
- **Regularization:** dropout and layer norm around each sub-block, residual connections.
- **Gradient flow:** backpropagation through all `T` steps. Because weights are shared, each weight receives the sum of its gradients from all steps, as in backprop through time.
- **bAbI:** 10 runs with different initializations, best model selected on validation.
- **Algorithmic tasks:** trained on length 40, evaluated on length 400, with randomized position offsets to encourage position-relative transformations. No curriculum.
- **Learning to Execute:** mix strategy from Zaremba and Sutskever (2015), no curriculum, no target sequences at test time.
- **Subject-verb agreement:** language modeling objective, evaluated by ranking accuracy of the correct verb form.
- **LAMBADA:** trained for next-word prediction on all words (LM setting); also a reading comprehension setting.
- **MT:** same setup as Vaswani et al. (2017), 8 P100 GPUs, UT with fully connected transition and no ACT, parameter count matched to Transformer base.

### 12.2 What the paper does not state in the text

- Exact `d`, `d_ff`, number of heads, or `T` for the UT base MT model.
- ACT threshold and maximum steps per task.
- Whether a ponder cost is used and its weight.
- Optimizer and learning-rate schedule per task (MT inherits Vaswani's setup; the others are unspecified).
- Separable convolution kernel sizes.
- Whether `T` was scaled with input length on the length-generalization tasks.

These live, if anywhere, in tensor2tensor hyperparameter sets. For a replication, list them explicitly as open choices.

---

## 13. Inference details

**Paper says:**

1. Run the encoder once on the input.
2. Run the decoder repeatedly. Each iteration consumes all tokens generated so far and produces a distribution for the next position.
3. Sample from it or take the highest-probability token.

**Cost implications (interpretation):**

| Aspect | Transformer (N layers) | UT (T steps) |
|---|---|---|
| Encoder cost | N layer applications | T applications of one layer |
| Decoder cost per generated token | N layer applications | T applications of one layer |
| Self-attention KV cache | N sets | T sets |
| Cross attention KV cache | N sets | 1 set suffices (shared weights, fixed encoder output) |
| Weight memory | N layers | 1 layer |
| Batching | Uniform depth | Uniform with fixed T; ragged with ACT |

With ACT, different tokens in a batch use different depths. Accelerators prefer static shapes, so in practice all positions run until the last one halts, and halted positions are masked. This saves little wall-clock time even when many tokens halt early.

---

## 14. Datasets, experiments, and results

All numbers below are from the paper's tables.

### 14.1 bAbI question answering

**Dataset:** 20 synthetic reasoning tasks (Weston et al., 2015). Short stories of facts plus a question. Two sizes: 1K and 10K training examples per task. Settings: train single (per task) and train joint (one model for all tasks).

**Metric:** average error, and number of failed tasks (error above 5%).

| Model | 10K single | 10K joint | 1K single | 1K joint |
|---|---|---|---|---|
| Transformer | 15.2 (10/20) | 22.1 (12/20) | 21.8 (5/20) | 26.8 (14/20) |
| UT | 0.23 (0/20) | 0.47 (0/20) | 5.31 (5/20) | 8.50 (8/20) |
| UT + ACT | 0.21 (0/20) | 0.29 (0/20) | 4.55 (3/20) | 7.78 (5/20) |

**Analysis the paper reports:**

- Attention starts almost uniform and becomes sharp on the supporting facts in later steps.
- Average ponder time grows with task difficulty: 2.3 steps (one supporting fact), 3.1 (two), 3.8 (three).
- On three-fact tasks many positions halt at step 1 or 2, and only a few get more steps, which looks like the model ignoring irrelevant facts.

### 14.2 Subject-verb agreement

**Dataset:** Linzen et al. (2016), English sentences where the model must predict the verb's number. Difficulty is measured by agreement attractors (intervening nouns with the opposite number).

| Model | 0 | 1 | 2 | 3 | 4 | 5 | Total |
|---|---|---|---|---|---|---|---|
| Best LSTM (prior) | 0.993 | 0.972 | 0.950 | 0.922 | 0.900 | 0.842 | 0.991 |
| Transformer | 0.973 | 0.941 | 0.932 | 0.917 | 0.901 | 0.883 | 0.962 |
| UT | 0.993 | 0.971 | 0.969 | 0.940 | 0.921 | 0.892 | 0.992 |
| UT + ACT | 0.994 | 0.969 | 0.967 | 0.944 | 0.932 | 0.907 | 0.992 |

Overall results are comparable to the prior best. The UT's advantage grows as attractors increase.

### 14.3 LAMBADA

**Dataset:** passages from novels (Paperno et al., 2016). Predict the last word, which humans can guess from the full passage but not from the last sentence alone. A control set of unfiltered passages checks general LM ability.

| Model | LM ppl (acc) control | dev | test | RC acc test |
|---|---|---|---|---|
| Transformer | 142 (0.19) | 5122 (0.0) | 7321 (0.0) | 0.3988 |
| LSTM | 138 (0.23) | 4966 (0.0) | 5174 (0.0) | 0.2007 |
| UT, 6 steps fixed | 131 (0.32) | 279 (0.18) | 319 (0.17) | 0.5216 |
| UT + ACT | 130 (0.32) | 134 (0.22) | 142 (0.19) | 0.5625 |
| UT, 8 steps fixed | 129 (0.32) | 192 (0.21) | 202 (0.18) | not reported |
| UT, 9 steps fixed | 129 (0.33) | 214 (0.21) | 239 (0.17) | not reported |

**The control experiment:** UT + ACT averaged 8.2 plus or minus 2.1 steps. To check whether ACT won only by using more steps, the authors trained fixed UTs with 8 and 9 steps. Both were better than 6 steps but worse than ACT. The authors suggest ACT may act as a regularizer. This is a hypothesis, not something they test further.

### 14.4 Algorithmic tasks

**Tasks:** copy, reverse, integer addition on decimal digit strings. Train on length 40, test on length 400.

| Model | Copy char | Copy seq | Reverse char | Reverse seq | Addition char | Addition seq |
|---|---|---|---|---|---|---|
| LSTM | 0.45 | 0.09 | 0.66 | 0.11 | 0.08 | 0.0 |
| Transformer | 0.53 | 0.03 | 0.13 | 0.06 | 0.07 | 0.0 |
| UT | 0.91 | 0.35 | 0.96 | 0.46 | 0.34 | 0.02 |
| Neural GPU (with curriculum) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Large improvement, but far from solved. Addition sequence accuracy is 2%.

### 14.5 Learning to Execute

**Tasks:** Zaremba and Sutskever (2015). Memorization (copy, double, reverse; max length 55) and program evaluation (program, control, addition; length 5, nesting 2).

| Model | Copy seq | Double seq | Reverse seq | Program seq | Control seq | Addition seq |
|---|---|---|---|---|---|---|
| LSTM | 0.11 | 0.047 | 0.32 | 0.12 | 0.21 | 0.11 |
| Transformer | 0.63 | 0.55 | 0.26 | 0.29 | 0.66 | 1.0 |
| UT | 1.0 | 1.0 | 1.0 | 0.63 | 1.0 | 1.0 |

### 14.6 Machine translation

**Dataset:** WMT 2014 English-German, same setup as Vaswani et al. (2017).

| Model | BLEU |
|---|---|
| UT small | 26.8 |
| Transformer base | 28.0 |
| Weighted Transformer base | 28.4 |
| UT base | 28.9 |

All base models have the same number of parameters. The UT uses the fully connected transition and **no ACT**. The introduction notes ACT marginally degraded MT results.

### 14.7 Summary of where ACT helps

| Task | ACT effect |
|---|---|
| bAbI | Small improvement |
| Subject-verb agreement | Better at high attractor counts |
| LAMBADA | Clear improvement, beats more fixed steps |
| MT | Slightly worse |

---

## 15. Argued-for choices vs control variables

### 15.1 Argued for (the paper's contributions)

| Choice | Evidence given |
|---|---|
| Weight sharing across depth | Every UT vs Transformer comparison |
| Recurrence over depth | Same as above; the two are inseparable in this paper |
| Per-position ACT | UT vs UT + ACT rows; LAMBADA 8 and 9 step control |
| Universality | Construction sketch (Neural GPU reduction) |

### 15.2 Introduced but not ablated

| Choice | What is missing |
|---|---|
| Timestep embedding added every step | No run without it |
| Re-adding position embedding every step | No run adding it only once |
| FFN vs separable conv transition | Chosen per task, no side-by-side |
| ACT threshold and max steps | No sweep reported |

### 15.3 Held fixed (control variables inherited from the Transformer)

- Multi-head scaled dot-product attention
- Post-LN residual blocks with dropout
- Sinusoidal encodings
- Encoder-decoder layout, teacher forcing, autoregressive decoding
- MT data, preprocessing, and training setup from Vaswani et al.

---

## 16. Weakest links

1. **Parameter-matched, not compute-matched.** "Same number of parameters" means the UT base does roughly `T` times more computation per parameter than a Transformer base, and must be wider to match parameter count. Illustrative arithmetic: if a 6-layer encoder has about 18.9M layer parameters, a param-matched single shared layer also has about 18.9M, and running it 6 times costs about 6 times the Transformer encoder's layer compute. The MT gain could come from extra compute, extra width, or the recurrence bias. The paper does not separate these.

2. **UT base configuration not reported in the text.** Without `d`, `d_ff`, heads, and `T`, the MT comparison is hard to interpret and replicate from the paper alone.

3. **Test-time depth extrapolation is claimed as a benefit but not demonstrated.** The paper stresses that UT depth can vary, and the universality argument requires `T` to grow with input length. No experiment trains at one `T` and evaluates at a larger `T`, and the length-generalization tasks do not say `T` was scaled with length.

4. **Length generalization is improved, not solved.** Copy at length 400 reaches 35% sequence accuracy; addition reaches 2%. The Neural GPU, which UT supposedly generalizes, gets 100% (with curriculum).

5. **Weak baselines on some tasks.** The Transformer overfits on bAbI (footnote 2 says it always overfit across hyperparameters). The Transformer and LSTM LAMBADA dev and test perplexities (thousands) suggest broken or badly overfit baselines, which inflates the relative gain.

6. **Transformer base number differs from the original paper.** Table 7 lists Transformer base at 28.0 BLEU; Vaswani et al. reported 27.3 for base (and 28.4 for big). The paper says setups are comparable, which suggests this is a rerun, but the discrepancy is not discussed. Single runs, no variance.

7. **ACT underspecified.** No ponder cost description, no threshold values, and the appendix listing's state handling (see [Section 8.4](#84-things-the-listing-does-not-make-obvious)) does not obviously match the prose.

8. **Seed selection on bAbI.** Best of 10 runs chosen on validation. Appendix E shows averages across seeds are notably worse (for example 10K single: 0.21 best vs 0.73 average), and the Transformer baseline is not reported the same way.

9. **Universality is a construction, not evidence.** It is a theoretical capacity statement under assumptions (unbounded steps, sufficient memory) that the experiments do not satisfy.

10. **"ACT as regularizer" is speculation.** The LAMBADA 8/9-step control is good, but the regularizer explanation is not tested.

11. **Minor inconsistencies.** Output matrix written as `m x V` for the decoder; Section 3.5 text says Tables 5 and 6 are program evaluation and memorization "respectively," but Table 5 is memorization and Table 6 is program evaluation.

12. **Recurrence and weight sharing are confounded with each other.** A shared-weight model run for fixed `T` is both. The paper cannot say whether the gains come from the bias of reusing a function or from regularization through fewer free parameters per unit of compute.

---

## 17. Why this paper did not make the same noise

This section is analysis, not something the paper states.

1. **Timing.** It appeared in July 2018, between GPT (June 2018) and BERT (October 2018). The field's attention moved to large-scale pretraining, where the question was "how big and how much data," not "which architecture."

2. **Parameters became cheap and compute became expensive.** Weight sharing gives more compute per parameter. In the scaling era the bottleneck was compute and throughput, so trading compute for fewer parameters was the wrong trade for most practitioners. At a fixed compute budget, unshared layers usually use that compute better.

3. **The flagship result was incremental.** 0.9 BLEU on WMT14 En-De with matched parameters is solid, not a step change like the Transformer's original speed and quality jump.

4. **The biggest wins were on small or synthetic tasks.** bAbI, Learning to Execute, and digit copying were viewed as toy benchmarks, so dramatic gains there did not change how people built production models.

5. **ACT is hard to use in practice.** It adds hyperparameters (threshold, ponder cost), it hurt MT, and variable per-token depth is awkward for batched accelerator execution with static shapes, which removes most of its real-world speed benefit.

6. **Inference is not cheaper.** Decoding cost per token scales with `T` like a `T`-layer model. Memory for weights shrinks, but weights were not the main serving bottleneck for most users at the time.

7. **The theory did not map to practice.** Turing completeness is appealing but practitioners care about benchmark scores per unit compute, and the paper's experiments cap steps anyway.

8. **Later scaling studies were mixed.** Tay et al. (2022), with Dehghani as a co-author, compared scaling behavior of ten architectures including Universal Transformers and found the best architecture changes with scale. The vanilla Transformer remained the default choice.

9. **No single killer demo.** BERT had GLUE, GPT-2 had generation samples. UT had a collection of moderate wins across many small tasks.

---

## 18. What is used today from this paper

### 18.1 Directly descended ideas

| Idea from UT | Where it shows up |
|---|---|
| Weight sharing across layers | ALBERT (Lan et al., 2020) for parameter efficiency; recursive and looped Transformers |
| One block applied repeatedly | Looped Transformers (Giannou et al., 2023, "Looped transformers as programmable computers"); Saunshi et al. (2025), "Reasoning with latent thoughts: on the power of looped transformers" |
| Depth as a test-time compute knob | Geiping et al. (2025), "Scaling up test-time compute with latent reasoning: a recurrent depth approach": Huginn-3.5B, pretrained on 800B tokens, runs a recurrent block many times at inference instead of generating chain-of-thought tokens |
| Looped language models at scale | Zhu et al. (2025), "Scaling latent reasoning via looped language models" |
| Per-token adaptive depth | PonderNet (Banino et al., 2021); Mixture-of-Recursions (2025), which routes tokens to different recursion depths |
| Telling a shared layer which step it is on | Depth conditioning in recent looped models, for example FiLM depth conditioning in looped ASR encoders (2026), which plays the role of UT's timestep embedding |
| Recurrent-depth models for compositional reasoning | 2026 work on implicit reasoning in recurrent-depth Transformers studies depth extrapolation and "overthinking" when too many loops are used |

### 18.2 What changed relative to the 2018 design

- **Pre-norm instead of post-norm** in most modern looped models, for stable deep unrolling.
- **Input injection:** Geiping et al. use a prelude, recurrent block, and coda structure, and feed the embedded input into the recurrent block at every iteration, instead of UT's "add position and timestep embedding."
- **Training with randomized loop counts** so the model works at many depths, which addresses UT's untested depth extrapolation.
- **Framing:** UT sold recurrence as inductive bias and universality; modern work sells it as latent reasoning and test-time compute scaling.

### 18.3 What is not used

- Mainstream published LLM families (for example GPT and Llama style models) still use unshared layers.
- The original ACT formulation is rarely used directly; newer work prefers routers, early exit, or convergence-based stopping.

---

## 19. Interview explanations

### 19.1 Thirty seconds

"The Universal Transformer takes a standard Transformer layer and applies it repeatedly with shared weights, instead of stacking different layers. That makes it recurrent over depth: every token's representation is refined in parallel, pass after pass. A timestep embedding tells the layer which pass it is on, and adaptive computation time lets each token stop when it is done. It combines Transformer parallelism with an RNN's bias toward repeatable procedures, generalizes better on algorithmic tasks, and is Turing complete under assumptions."

### 19.2 Two minutes

1. **Problem:** Transformers beat RNNs on translation but fail at simple algorithmic tasks like copying, especially on longer inputs. The fixed stack of distinct layers means a fixed number of sequential steps and no bias toward repeated procedures.
2. **Idea:** share weights across layers. One layer (self-attention then transition) is applied `T` times. The output of step `t` feeds step `t+1`.
3. **View it as RNNs:** one RNN per token, sharing weights, updating in parallel, and reading each other's states through attention every step.
4. **Details:** sinusoidal position plus timestep embedding added each step; post-LN residual blocks; FFN or separable conv transition; ACT for per-token halting.
5. **Theory:** because depth is decoupled from parameters, `T` can grow with input length, which gives computational universality; it reduces to a Neural GPU as a special case.
6. **Results:** big gains on bAbI, Learning to Execute, and length generalization; state of the art on LAMBADA with ACT; 0.9 BLEU over Transformer base on WMT14 En-De at matched parameters.
7. **Caveats:** matched parameters, not compute; depth extrapolation not shown; ACT hurt MT.

### 19.3 Likely follow-up questions with answers

**Q: Is a UT with fixed T just a Transformer?**
A: It is a `T`-layer Transformer with all layer weights tied, plus a timestep embedding each step. The difference matters because tied weights let you change `T` without new parameters and give an iterative inductive bias.

**Q: Why do you need the timestep embedding?**
A: With identical weights at every step, the layer has no other way to know which pass it is doing. In a normal Transformer each layer's unique weights implicitly encode its depth.

**Q: How is UT recurrence different from RNN recurrence?**
A: RNNs recur over positions and the recurrent step only sees a fixed-size state. UT recurs over depth and the recurrent step attends to all positions' previous states, so it has memory access inside the loop.

**Q: Does UT save compute?**
A: No. It saves parameters. Compute grows with `T` like a `T`-layer model. In the paper's MT comparison it uses more compute than the Transformer at matched parameters.

**Q: Why does the standard Transformer fail to be universal in the paper's argument?**
A: With finite precision, its number of sequential operations is constant (number of layers), so there are inputs longer than that depth requiring more sequential steps than it has.

**Q: What does ACT do to a halted token?**
A: Its output state is frozen and copied forward. Other tokens can still attend to it. Computation continues until all tokens halt or the step cap is reached.

**Q: Why did ACT hurt MT?**
A: The paper does not explain. Plausible reasons: translation benefits from uniform depth, halting adds optimization noise, or hyperparameters were not tuned for MT.

**Q: How would you make UT practical today?**
A: Pre-norm, input injection each loop, training with random loop counts, compute-matched evaluation, and batching-friendly halting (or fixed depth with an early-exit criterion at inference).

**Q: What is the KV cache difference at inference?**
A: Self-attention needs one cache per step, like a `T`-layer model. Cross attention keys and values can be computed once, because the encoder output is fixed and the projection weights are shared across steps.

---

## 20. Questions to test your understanding

Try answering these before checking back.

1. What exactly is shared across what in a Transformer, and what additionally is shared in a UT?
2. Why does the tensor shape of `H^t` need to equal the shape of `H^{t-1}` for the model to work?
3. Write `A^t` and `H^t` from memory. Where does `P^t` enter?
4. What is `d` inside Eq. 1 for a model with `d = 512` and `k = 8`?
5. What happens to the UT when `T = 1`?
6. Why can a UT, but not a standard Transformer, choose depth at inference time?
7. Why does the Neural GPU reduction require `T` equal to input length?
8. In the ACT worked example, why is the weight at the halting step the remainder, not `p`?
9. Why do fixed 8 and 9 step UTs on LAMBADA matter for interpreting the ACT result?
10. Name two confounds in the MT comparison.
11. Which experiment would most directly test the claim that shared depth enables longer computation at test time?
12. How many attention score tensors of shape `B x k x n x m` are computed per generated token in the decoder, and how many unique cross attention key/value tensors are needed?
13. Why might post-LN be a problem for large `T`?
14. Is a UT still Turing complete if ACT has a maximum step count?
15. What does "recurrence over depth" mean for gradient flow to a single weight matrix?
16. Why would weight sharing improve length generalization rather than hurt capacity?
17. What would a compute-matched baseline for UT base on WMT14 look like?
18. How does input injection in modern looped models relate to UT's per-step `P^t`?

---

## 21. Implementation notes and pitfalls

### 21.1 Minimal encoder loop (PyTorch-style sketch, fixed T)

```python
def encode(x_ids):
    h = embed(x_ids)                                  # B x m x d
    for t in range(1, T + 1):
        x = h + coord_embedding(m, t)                 # P^t, m x d, broadcast
        a = norm1(x + dropout(self_attn(x, x, x)))    # Eq. 5
        h = norm2(a + dropout(transition(a)))         # Eq. 4
    return h                                          # B x m x d
```

`self_attn`, `transition`, `norm1`, `norm2` are created once, outside the loop. That single fact is the Universal Transformer.

### 21.2 Pitfalls

1. **Creating layers inside the loop.** This silently turns the model back into a normal Transformer.
2. **Adding `P^t` only once.** The paper adds it every step, and it contains the timestep term.
3. **Wrong scaling in attention.** Scale by `sqrt(d/k)`, the per-head dim.
4. **Post-LN instability at large `T`.** Reproduce with post-LN for fidelity; treat pre-LN as a documented ablation.
5. **ACT masks and broadcasting.** Keep `B x m` and `B x m x 1` straight.
6. **ACT state handoff.** Decide explicitly whether the next step attends to `transformed_state` or `new_state`, and document why, after checking tensor2tensor.
7. **Ponder cost.** Without it, ACT tends to use the maximum steps. Log `n_updates` distributions.
8. **Comparisons.** Report both parameter-matched and compute-matched baselines.
9. **Depth extrapolation.** If claiming it, train at one `T` and test at larger `T`.

### 21.3 Suggested ablation axes for a replication

| Axis | Values |
|---|---|
| Weight sharing | shared vs unshared, same `T` |
| Steps | 1, 2, 4, 6, 8, 12 |
| Test-time steps | train at 6, test at 4 to 16 |
| Timestep embedding | on vs off |
| Norm placement | post-LN vs pre-LN |
| Transition | FFN vs separable conv |
| ACT | off, on with threshold sweep, ponder cost sweep |
| Baseline matching | parameter-matched vs compute-matched |
| Task | copy and reverse with train length 40, test length 400 |

---

## 22. Glossary

| Term | Meaning |
|---|---|
| Recurrence | Applying the same function repeatedly, each output feeding the next input |
| Recurrence over depth | Repeating a layer across refinement steps instead of across sequence positions |
| Weight sharing (tying) | Using the same parameters in multiple places |
| Transition function | Per-position network applied after attention (FFN or separable conv) |
| Timestep embedding | Sinusoidal signal for the step index, summed with the position embedding |
| ACT | Adaptive Computation Time (Graves, 2016): learned per-step halting |
| Ponder time | Number of steps a position uses before halting |
| Remainder | Leftover halting mass used as the weight at the halting step |
| Inductive bias | Assumptions built into an architecture that make some solutions easier to learn |
| Turing complete / computationally universal | Able to simulate any Turing machine given enough resources |
| Neural GPU | Convolutional recurrent model that learns algorithms; UT reduces to it in a special case |
| Teacher forcing | Feeding ground-truth previous tokens to the decoder during training |
| Agreement attractor | Noun between subject and verb with the opposite number |
| Post-LN / pre-LN | Layer norm after or before the residual sub-block |

---

## 23. References

- Dehghani et al. (2019). Universal Transformers. ICLR. arXiv:1807.03819
- Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762
- Graves (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv:1603.08983
- Kaiser and Sutskever (2016). Neural GPUs Learn Algorithms. arXiv:1511.08228
- Graves, Wayne, Danihelka (2014). Neural Turing Machines. arXiv:1410.5401
- Zaremba and Sutskever (2015). Learning to Execute. arXiv:1410.4615
- Weston et al. (2015). Towards AI-Complete Question Answering (bAbI). arXiv:1502.05698
- Linzen, Dupoux, Goldberg (2016). Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies. TACL
- Paperno et al. (2016). The LAMBADA Dataset. ACL
- Chollet (2016). Xception: Deep Learning with Depthwise Separable Convolutions. arXiv:1610.02357
- Perez, Marinkovic, Barcelo (2019). On the Turing Completeness of Modern Neural Network Architectures. ICLR
- Lan et al. (2020). ALBERT. ICLR
- Banino, Balaguer, Blundell (2021). PonderNet. arXiv:2107.05407
- Tay et al. (2022). Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling? arXiv:2207.10551
- Giannou et al. (2023). Looped Transformers as Programmable Computers. ICML
- Geiping et al. (2025). Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv:2502.05171
- Saunshi et al. (2025). Reasoning with Latent Thoughts: On the Power of Looped Transformers
- Zhu et al. (2025). Scaling Latent Reasoning via Looped Language Models. arXiv:2510.25741