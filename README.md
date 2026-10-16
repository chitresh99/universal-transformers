# Implementing Dehghani et al. (2019): Universal Transformers

Paper: [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)

## Quick start

```bash
uv sync
cd src/universal_transformers
python3 ut.py
```

## Goal

Implement the Universal Transformer from scratch in PyTorch: one encoder block and one decoder block applied repeatedly instead of a stack of distinct layers, a coordinate embedding that encodes both position and refinement step, and Adaptive Computation Time (ACT), which lets each position decide how many refinement steps it needs. The goal was a correct small-scale implementation that runs on a CPU, where every component is checked and the halting behaviour can be inspected, not a reproduction of the paper's results.

## The synthetic tasks

The model is trained on three algorithmic tasks at once.

- The source is a random sequence of 4 to 10 symbols drawn from 12 (`s0` to `s11`), so repeated symbols are common.
- **copy**: the target is the source unchanged.
- **reverse**: the target is the source in reverse order.
- **sort**: the target is the source in ascending order.
- Each example draws its task at random, and a task token (`<copy>`, `<reverse>` or `<sort>`) is placed at the start of the source so the model knows which one to perform.
- There is exactly one correct output per input.
- 20000 training examples and 500 held-out examples, generated from separately seeded random generators.
- Vocabulary: 12 symbols, 3 task tokens and 3 special tokens (`<pad>`, `<sos>`, `<eos>`), 18 in total.

The tasks differ in kind. Copy is the identity mapping. Reverse needs each position to locate its mirror position, which depends on the sequence length. Sort cannot be solved by position at all: it needs comparisons between symbols and counting of repeats. Training them together in one model makes it possible to see whether ACT spends different amounts of computation on tasks of different difficulty.

## What follows the paper

| Component | Paper | This implementation |
|---|---|---|
| Recurrence in depth | One block applied repeatedly, weights shared across steps | Same, one encoder block and one decoder block |
| Block structure | Self-attention, then a transition function, each with a residual connection, dropout and layer normalization | Same |
| Decoder | Adds attention over the encoder's final state, with causal self-attention | Same |
| Transition function | Position-wise fully connected layers (or a separable convolution) | Fully connected: two linear layers with ReLU between |
| Coordinate embedding | Sinusoidal position term plus sinusoidal timestep term, both over the same frequency bank, added at every step | Same |
| Dynamic halting | ACT per position (Appendix C): a halting probability each step, halt when the running sum passes a threshold, use the remainder on the halting step | Same, threshold 1 - 0.01 |
| Halted positions | State frozen once a position halts | Same, through the ACT update weights |
| Ponder cost | From ACT (Graves, 2016); the weight is not given in the paper | Number of updates plus remainders, weighted by 1e-3 |
| Algorithmic task metrics | Character accuracy and sequence accuracy | Same |

## Deviations

| | Paper | This implementation | Reason |
|---|---|---|---|
| Scale | Full-size models | d_model 128, 4 heads, d_ff 256, 338,452 parameters | CPU |
| Maximum steps | No single value; the fixed-step model on LAMBADA used 6 | 6 | Matches the paper's fixed-step setting; enough for short sequences |
| Algorithmic tasks | Copy, reverse and addition on decimal digits (0 to 9), results reported per task | Copy, reverse and sort on 12 symbols, trained together in one model | Sort tests content-based reasoning; one model allows comparing ponder times across tasks |
| Task signal | No task token described | Task token at the start of the source | With the tasks mixed, the same source would otherwise have more than one correct target |
| Sequence lengths | Trained on length 40, evaluated on length 400 | 4 to 10, same range for training and evaluation | CPU; length generalization is not tested |
| Position offsets | Training positions start at randomized offsets, to encourage position-relative transformations | Positions always start at 0 | Not needed without a length generalization test |
| Position index | Positions counted from 1 in the coordinate embedding | Positions counted from 0 (steps still start at 1) | Follows the Transformer convention; only shifts the encoding by one position |
| Halting unit input | The halting probability is computed from the state alone (Appendix C, Listing 2) | Computed from the state plus the coordinate embedding | Lets the halting unit see the step number; a small departure |
| Other tasks | bAbI, LAMBADA, subject-verb agreement, machine translation | Not implemented | Out of scope |
| Hardware | GPUs | CPU only | Compute |

### Why these defaults

The values in `Config` are sized so that a full run finishes on a CPU while the hardest task, sort, still reaches 100%. Training converges well before the last epoch (the loss is already 0.021 at epoch 10), so `epochs = 30` leaves a margin rather than being a tuned number.

## Code decisions

These are choices the paper does not dictate, made in this implementation.

- **Task tokens instead of separate models.** One model learns all three tasks, so ponder times can be compared on equal terms. The task token is part of the source and is visible to every encoder position through self-attention.
- **Embeddings initialised with standard deviation d_model^-0.5.** They are multiplied by sqrt(d_model) before use, so this gives values of about 1, the same size as the coordinate embedding. With PyTorch's default initialisation the embeddings would be about 11 times larger and drown out the position signal (see the bugs table).
- **Separate source and target embeddings, and an untied output layer.** The task tokens only ever appear on the source side, so sharing gives little here.
- **The coordinate embedding is added at every step,** including step 1, before self-attention, as in equation 5 of the paper. It is also part of the input to the halting unit, which the paper does not do (see Deviations). There is no separate positional encoding before the first step.
- **The halting unit is a single linear layer per block,** created in `UTBlock.__init__` so it is always registered before the optimizer is built.
- **The ponder cost is averaged over all positions, padding included.** The penalty is small (1e-3), and padding positions behave consistently across batches, so this does not distort training noticeably.
- **No label smoothing.** The loss can go all the way to 0, which makes the overfit check easy to read.
- **Gradient norms are clipped to 1.0,** and the learning rate warms up over 400 steps to a peak of 3e-4, then decays in proportion to 1 / sqrt(step).
- **Batches of 64 examples, sorted by source length,** with the batch order reshuffled every epoch from a seeded generator.
- **Greedy decoding only.** Each input has exactly one correct output, so beam search adds nothing here.
- **The overfit check has its own settings:** 32 examples, batch size 8, 300 epochs (1200 steps), warmup 100, peak learning rate 1e-3, and dropout and the ponder penalty both off. It measures pure memorisation.
- **Ponder statistics are reported per task on content positions only.** The task token, `<eos>` and padding are excluded, because the edge tokens behave differently from symbols and would blur the comparison.

## Pipeline overview

1. **Data generation.** Seeded generators produce examples, each with a randomly chosen task.
2. **Vocabulary.** Integer ids for the special tokens, the task tokens and the symbols.
3. **Tensorization.** The source becomes `<task>` + symbols + `<eos>`. The target becomes `<sos>` + symbols + `<eos>`.
4. **Batching.** Examples are sorted by source length, cut into batches of 64 and padded.
5. **Encoder.** The scaled embeddings go through the encoder block repeatedly under ACT. At each step the coordinate embedding is added, each position computes a halting probability, and positions that have halted keep their state.
6. **Decoder.** The same process with the decoder block, which also attends to the encoder's final state. A linear layer maps each position to scores over the vocabulary.
7. **Training.** Teacher forcing with cross-entropy plus the ponder penalty, Adam, gradient clipping and the warmup schedule.
8. **Evaluation.** Greedy decoding on the 500 held-out examples, character and sequence accuracy per task, ponder statistics per task, and three samples with per-position ponder times.

## Checks

| Check | What it verifies |
|---|---|
| Shapes | Logits and per-position ponder counts have the expected shapes |
| Ponder bound | No position ever takes more than `max_steps` steps, so a miscounting step counter fails loudly |
| Overfit | The model can memorise 32 training examples to 100% sequence accuracy with dropout and the ponder penalty off |

The overfit check is the one that mattered most. Every bug in the table below was surfaced by it failing, not by the full training run. If the model cannot memorise 32 examples, something is structurally broken, and training on 20000 examples will not fix it.

### Parameter count

| Part | Calculation | Parameters |
|---|---|---|
| Source and target embeddings | 2 * 18 * 128 | 4,608 |
| One attention block | 4 * (128 * 128 + 128) | 66,048 |
| Transition function | 128 * 256 + 256 + 256 * 128 + 128 | 65,920 |
| One LayerNorm | 2 * 128 | 256 |
| Halting unit | 128 + 1 | 129 |
| Encoder block | 1 attention + transition + 2 LayerNorms + halting unit | 132,609 |
| Decoder block | 2 attention + transition + 3 LayerNorms + halting unit | 198,913 |
| Output layer | 128 * 18 + 18 | 2,322 |
| Total | 4,608 + 132,609 + 198,913 + 2,322 | 338,452 |

The encoder and decoder each have one block regardless of how many steps they take, so the parameter count does not grow with `max_steps`.

## Bugs found during implementation

| Bug | Symptom | Fix |
|---|---|---|
| Timestep term in the coordinate embedding was a scalar `sin(step)` | Found by inspection; every dimension got the same time signal | Use the same frequency bank as the position term, `sin(t / 10000^(2j/d))` |
| ACT computed a weighted accumulator but fed the raw transformed state to the next step | Overfit check at about 3% character accuracy; halting had no effect on the forward pass | The weighted state is both carried forward and returned: `state = transformed * w + state * (1 - w)` |
| Ponder penalty applied to a detached integer step counter, which also double-counted the halting step | Average ponder above `max_steps` (7.12 with a maximum of 6); the penalty had no effect | Differentiable cost `n_updates + remainders`; detached count kept for reporting only; assertion that ponder never exceeds `max_steps` |
| Halting layer created lazily inside the halting loop | No symptom yet; any code that built the optimizer before the first forward pass would have left it untrained | Moved into `UTBlock.__init__` |
| No task signal in the source while copy and reverse were mixed | The same input had two correct targets; held-out sequence accuracy 3% on copy and 0.4% on reverse | Task token prepended to the source |
| Embeddings initialised with standard deviation 1, then multiplied by sqrt(128) | Embeddings about 11 times larger than the coordinate embedding, so the model was nearly blind to position; outputs had the right symbols in a scrambled order | Initialise with standard deviation d_model^-0.5 |
| Overfit check ran 200 steps against a 400-step warmup | The learning rate never passed half its peak; 54% character and 16% sequence accuracy on 32 examples | Batch size 8, 300 epochs, warmup 100, peak 1e-3 |

The most instructive of these was a wrong diagnosis. After the first four fixes, the overfit check still failed, and copy outputs looked right on some positions and scrambled on others. That pattern looked like a weak positional signal, and the suspected cause was position and timestep colliding in the shared sinusoidal frequency bank at this small scale. The coordinate embedding was never changed, though. The real cause was the embedding scale burying the positional signal, and once that and the task token were fixed, the paper's summed embedding worked as specified.

The two quietest bugs were the missing task token and the embedding scale. Neither raised an error, and each made the task close to impossible.

## Results

All runs on CPU with seed 0.

| | copy | reverse | sort | overall |
|---|---|---|---|---|
| Held-out character accuracy | 100.0% | 100.0% | 100.0% | 100.0% |
| Held-out sequence accuracy | 100.0% | 100.0% | 100.0% | 100.0% |

- **Overfit check:** 100% character and sequence accuracy on 32 examples, with the loss reaching 0.0000.
- **Training:** cross-entropy falls from 2.22 in epoch 1 to 0.021 by epoch 10 and 0.0027 by epoch 30. Perplexity ends at 1.003.
- **Repeated symbols:** sort is solved exactly even with several repeats, for example a source with `s0` three times and `s4`, `s10` and `s11` twice each.

### Ponder times by task

Held-out examples, symbol positions only, maximum of 6 steps.

| Task | Encoder steps | Decoder steps | Encoder positions at the cap |
|---|---|---|---|
| copy | 5.14 | 4.58 | 19% |
| reverse | 5.80 | 4.66 | 87% |
| sort | 5.87 | 5.07 | 90% |

### Samples

```
task            sort
source          s4 s0 s11 s1 s11 s0 s10 s10 s0 s4
target          s0 s0 s0 s1 s4 s4 s10 s10 s11 s11
predicted       s0 s0 s0 s1 s4 s4 s10 s10 s11 s11
enc ponder/pos  [5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 2]

task            reverse
source          s9 s3 s0 s5
target          s5 s0 s3 s9
predicted       s5 s0 s3 s9
enc ponder/pos  [5, 6, 6, 6, 6, 3]
```

The ponder lists include the task token (first) and `<eos>` (last).

## Analysis

### ACT spends less on the easiest task

Copy gets clearly less computation than the other two tasks: only 19% of its encoder positions use all 6 steps, against 87% for reverse and 90% for sort. Copy is the identity mapping, and the halting unit has learned to stop earlier for it. The model sees which task it is doing only through the task token, so this difference had to be learned through self-attention.

### Sort stands out in the decoder

In the encoder, reverse and sort look alike. Both are hard in their own way: reverse has to locate mirror positions, sort has to compare symbols. In the decoder, sort gets 5.07 steps against about 4.6 for the other two. That fits where sort's difficulty lies. Choosing the next smallest symbol and keeping count of repeats happens during generation, which is the decoder's job. This is a plausible reading, not a proven mechanism.

### The step limit hides part of the picture

With about 90% of reverse and sort encoder positions at 6 steps, the encoder is mostly being cut off rather than choosing to halt. The ordering copy < reverse <= sort is clear, but the size of the gap between reverse and sort is hidden by the ceiling. The decoder, which mostly stays below the cap, is where the difference shows. These figures come from one run with one seed, so small gaps such as 5.80 against 5.87 should not be read as meaningful.

### The paper sees the same pattern

On bAbI, the paper reports that average ponder time grows with task difficulty: about 2.3 steps for questions needing one supporting fact, 3.1 for two and 3.8 for three. The ordering here (copy below reverse and sort) is the same kind of effect on much simpler tasks: the halting unit learns to give harder problems more computation, without being told which problems are harder.

### Edge tokens halt early

In every sample, `<eos>` takes only 2 or 3 steps while symbol positions take 5 or 6. The end of the sequence is easy to identify, and ACT treats it as needing little refinement. This is the per-position adaptivity the method is designed for.

### Memorisation needs almost no depth

In the overfit check, the average number of steps falls from 3.48 to about 1.04, even though the ponder penalty is off. Memorising 32 examples does not need repeated refinement, and the halting unit learns to stop after one step. On the full data, the average rises instead, from 2.04 in epoch 1 to about 5.1, as the model learns that generalising to unseen sequences needs more steps.

## Glossary

**Universal Transformer.** A Transformer in which one block is applied repeatedly, with the same weights each time, instead of stacking distinct layers. Each application refines every position's representation further. The number of applications can be fixed or, with ACT, decided per position.

**Weight sharing across depth.** Every refinement step uses the same parameters. This makes the model recurrent in depth rather than in time, and keeps the parameter count independent of the number of steps.

**Coordinate embedding.** The signal added to every position at every step. It is the sum of a sinusoidal encoding of the position and a sinusoidal encoding of the current step, so the block can tell where a position is and how many refinement steps have happened.

**Adaptive Computation Time (ACT).** A mechanism from Graves (2016) that lets each position decide how many steps to take. At each step, a halting unit outputs a probability. When the running sum of these probabilities for a position passes 1 - epsilon, that position halts and its state stops changing.

**Halting probability.** The output of the halting unit, a sigmoid over a linear function of the position's current state plus its coordinate embedding.

**Remainder.** On the step where a position halts, the amount left to bring its total halting probability to exactly 1. It is used as that step's update weight, so each position's updates add up to a complete weighting.

**Ponder time.** The number of steps a position took before halting. Here it is reported per position in the samples and averaged per task in the ponder table.

**Ponder cost.** The number of updates plus the remainders, averaged over positions. Adding it to the loss, scaled by a small weight, encourages the model to halt sooner. The remainder is the part with a gradient, so it is what the optimizer actually adjusts.

**At the cap.** A position that used all `max_steps` steps. A high share at the cap means the step limit, not the halting unit, is deciding when positions stop.

**Cross-entropy (ce).** The loss: for each target position, -log(p), where p is the probability the model gives to the correct token, averaged over real target tokens. With no label smoothing it can reach 0, when the model is certain and correct everywhere.

**Perplexity (ppl).** e^(cross-entropy). It reads as the number of tokens the model is effectively choosing between at each step. A perplexity of 1 means certain and correct.

**Character accuracy.** The fraction of target positions where the prediction is correct, comparing position by position.

**Sequence accuracy.** The fraction of sequences where the whole prediction is identical to the target. One wrong or missing token makes the whole sequence wrong. It is the stricter of the two measures.

**Teacher forcing.** During training, the decoder input is the correct target shifted right by one, starting with `<sos>`, rather than the model's own predictions. A causal mask stops each position from seeing the answer.

**Warmup learning rate schedule.** The learning rate rises linearly for the first `warmup_steps` steps to its peak, then decays in proportion to 1 / sqrt(step). The warmup avoids large, unstable updates at the start of training.

**Greedy decoding.** At each step, take the single most likely next token and feed it back in, until `<eos>` or a length limit.

**Masking.** Setting some attention scores to negative infinity before the softmax so those positions get exactly zero weight. The padding mask hides `<pad>`, and the causal mask stops each decoder position from attending to later positions.

**Overfit check.** Training on a tiny set (32 examples here) until the model memorises it. A correct model and training loop should reach 100%. If it cannot, something is broken.

## Lessons

- **The overfit check found every bug.** Each failure was visible on 32 examples long before the full run would have made it clear, and it gave a fast loop for testing fixes.
- **A plausible diagnosis can still be wrong.** The scrambled copy outputs pointed convincingly at a collision in the coordinate embedding. The actual cause was the embedding scale, and the coordinate embedding never needed changing. Checking magnitudes before redesigning a component would have found it sooner.
- **Check that a task is learnable before tuning the model.** Mixing copy and reverse without a task token made the targets ambiguous, and no amount of training could have fixed that.
- **Initialisation scales interact.** Multiplying embeddings by sqrt(d_model) only makes sense if they start small. With PyTorch's default initialisation, the same line made the embeddings swamp everything added to them.
- **Read aggregate numbers, not just samples.** Three samples suggested ACT used the same number of steps for every task. The per-task averages over 500 examples showed copy clearly ponders less.
- **A cap can masquerade as a decision.** When most positions stop at `max_steps`, the halting unit is not choosing. Reporting the share at the cap makes that visible.

## Citations

```bibtex
@inproceedings{dehghani2019universal,
    title     = {Universal Transformers},
    author    = {Mostafa Dehghani and Stephan Gouws and Oriol Vinyals and Jakob Uszkoreit and Lukasz Kaiser},
    booktitle = {International Conference on Learning Representations},
    year      = {2019},
    url       = {https://arxiv.org/abs/1807.03819}
}

@misc{graves2016adaptive,
    title   = {Adaptive Computation Time for Recurrent Neural Networks},
    author  = {Alex Graves},
    year    = {2016},
    eprint  = {1603.08983},
    archivePrefix = {arXiv},
    primaryClass = {cs.NE},
    url     = {https://arxiv.org/abs/1603.08983}
}
```