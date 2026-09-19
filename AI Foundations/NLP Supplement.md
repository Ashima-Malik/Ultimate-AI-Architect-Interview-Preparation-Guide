# NLP — Deep Dive Supplement

### Everything that needs a simple example, worked through step by step

> This document supplements the main NLP guide. Every concept here has a worked example with real numbers.

---


---

# PART 1: WORD REPRESENTATIONS — WITH REAL EXAMPLES

---

## One-Hot Encoding — Full Worked Example

**Setup:** A tiny vocabulary of 5 words: `[cat, dog, fish, bird, run]`

Each word gets a position. Put a 1 there and 0 everywhere else.

```
         cat  dog  fish  bird  run
"cat" →  [ 1,   0,    0,    0,   0 ]
"dog" →  [ 0,   1,    0,    0,   0 ]
"fish"→  [ 0,   0,    1,    0,   0 ]
"bird"→  [ 0,   0,    0,    1,   0 ]
"run" →  [ 0,   0,    0,    0,   1 ]
```

**Encoding a sentence:** "cat and dog" → add the vectors:
```
cat  →  [1,0,0,0,0]
dog  →  [0,1,0,0,0]
Sum  →  [1,1,0,0,0]   ← this is the Bag-of-Words representation
```

**The problem visualised:**

How different is "cat" from "dog"?
```
Distance("cat","dog")  = sqrt((1-0)² + (0-1)² + 0 + 0 + 0) = √2 = 1.41
Distance("cat","fish") = sqrt((1-0)² + 0 + (0-1)² + 0 + 0) = √2 = 1.41
```

Cat is exactly as "different" from dog as it is from fish. The model has no idea that cat and dog are both animals. This is the core failure of one-hot encoding.

---

## Bag of Words — Full Worked Example

**Two movie reviews:**

```
Review 1: "The movie was great. Great acting, great story."
Review 2: "The movie was boring. Terrible acting."
```

**Build vocabulary:** `[the, movie, was, great, acting, story, boring, terrible]`

**Vectorise:**
```
           the  movie  was  great  acting  story  boring  terrible
Review 1:  [ 1,   1,    1,    3,     1,      1,      0,      0   ]
Review 2:  [ 1,   1,    1,    0,     1,      0,      1,      1   ]
```

A classifier sees Review 1 has `great=3` and `boring=0` → predicts positive.
Review 2 has `great=0` and `boring=1, terrible=1` → predicts negative. ✓

**What BoW ignores:**
"The movie was not great" → `great=1` (incorrectly suggests positive).
Order is completely lost. "Not great" and "great" look the same.

---

## TF-IDF — Full Worked Example


**Setup:** 3 job postings (our "documents"):

```
Doc 1: "machine learning engineer job"       (4 words)
Doc 2: "data science machine learning job"   (5 words)
Doc 3: "python developer job posting today"  (5 words)
```

**Step 1 — TF (Term Frequency)** = how often word appears in THIS doc / total words in doc

```
Word "job":
  TF(job, Doc1) = 1/4 = 0.25
  TF(job, Doc2) = 1/5 = 0.20
  TF(job, Doc3) = 1/5 = 0.20

Word "machine":
  TF(machine, Doc1) = 1/4 = 0.25
  TF(machine, Doc2) = 1/5 = 0.20
  TF(machine, Doc3) = 0/5 = 0.00   ← "machine" not in Doc3
```

**Step 2 — IDF (Inverse Document Frequency)** = log(N / df) where N = total docs, df = docs containing this word

```
N = 3 docs

"job"     → in all 3 docs → IDF = log(3/3) = log(1) = 0.00
"machine" → in 2 docs     → IDF = log(3/2) = 0.41
"python"  → in 1 doc      → IDF = log(3/1) = 1.10
"today"   → in 1 doc      → IDF = log(3/1) = 1.10
"data"    → in 1 doc      → IDF = log(3/1) = 1.10
```

**Step 3 — TF-IDF = TF × IDF**

```
TF-IDF("job", Doc1)     = 0.25 × 0.00 = 0.000  ← "job" tells us NOTHING about Doc1
TF-IDF("machine", Doc1) = 0.25 × 0.41 = 0.103  ← "machine" somewhat identifies Doc1
TF-IDF("python", Doc3)  = 0.20 × 1.10 = 0.220  ← "python" is THE defining word of Doc3
TF-IDF("today", Doc3)   = 0.20 × 1.10 = 0.220  ← "today" also distinctive but less meaningful
```

**What this means for product:**
If a recruiter searches "python", the TF-IDF score correctly ranks Doc 3 highest. "Job" would be useless as a search signal — it appears in every single posting, so IDF = 0.

**Where TF-IDF is used:**
LinkedIn job search (keyword matching layer), Google's early search ranking, document classification, keyword extraction for SEO.

---

## Word2Vec — Full Worked Example


**Training corpus (simplified):** These 6 sentences are shown to the model millions of times.

```
"I love my cat"
"I love my dog"
"cats eat fish"
"dogs eat bones"
"my pet cat is cute"
"my pet dog is cute"
```

**The training task (Skip-gram):**

For sentence "I love my cat", window size = 2:

```
Center word "love"  →  predict: ["I", "my"]
Center word "my"    →  predict: ["I", "love", "cat"]
Center word "cat"   →  predict: ["love", "my"]
```

**What the network learns:**

"cat" always appears near: `[love, my, eat, fish, pet, cute]`
"dog" always appears near: `[love, my, eat, bones, pet, cute]`

The context words are **almost identical**. So the network gives "cat" and "dog" nearly identical weight patterns.

**After training:**

```
"cat"   → [0.82, 0.14, 0.65, 0.91, 0.23]
"dog"   → [0.79, 0.18, 0.61, 0.88, 0.25]
"fish"  → [0.21, 0.76, 0.12, 0.33, 0.88]
```

"cat" and "dog" vectors are close together. "fish" is far away.

**Cosine Similarity (how close two vectors are):**

```
similarity("cat","dog")  = 0.97   ← almost identical!
similarity("cat","fish") = 0.31   ← very different
```

**The king−man+woman=queen arithmetic:**

```
"king"  → [0.82, 0.10, 0.90, 0.30]   (has: royalty, male)
"man"   → [0.10, 0.09, 0.88, 0.28]   (has: male, human)
"woman" → [0.11, 0.91, 0.12, 0.71]   (has: female, human)

king - man + woman:
= [0.82-0.10+0.11, 0.10-0.09+0.91, 0.90-0.88+0.12, 0.30-0.28+0.71]
= [0.83, 0.92, 0.14, 0.73]

Nearest vector to [0.83, 0.92, 0.14, 0.73] in the vocabulary = "queen" ✓
```

The model learned gender and royalty as geometric directions — with zero human supervision.

---

## GloVe — What's Different from Word2Vec

Word2Vec trains on local windows — it only sees pairs of nearby words.
GloVe trains on the global co-occurrence matrix — how often does word A appear near word B across the ENTIRE corpus?

```
Co-occurrence count (across 10 billion words):
  "ice" near "cold"    → 18,000 times
  "ice" near "water"   → 12,000 times
  "steam" near "hot"   → 14,000 times
  "steam" near "water" → 11,000 times
  "ice" near "hot"     → 100 times     ← rarely together
  "steam" near "cold"  → 80 times      ← rarely together
```

GloVe reads these global counts and learns:
- ice and steam are both related to water
- ice is linked to cold, steam to hot
- ice and steam are opposites along the temperature dimension

**Result:** GloVe embeddings often capture semantic relationships more consistently than Word2Vec because they use the whole picture, not just local windows.

---

## FastText — Handles Unknown Words

**Problem with Word2Vec:**
If "LinkedIn" wasn't in the training data, it gets no embedding. Unseen word = the model is completely blind.

**FastText solution:** Break every word into character n-grams:

```
"LinkedIn" → ["Lin", "ink", "nkd", "kdI", "dIn", "In", "nk", "kd", "dI"]
              + the whole word "LinkedIn"
```

Each n-gram gets its own vector. The word's embedding = average of all its n-gram vectors.

**Why this helps:**

If "LinkedIn" was never seen, but "link", "linked", and "In" were — the model can still build a reasonable embedding from the n-grams it knows.

"Unhappiness" → `["un","unhap","happi","happiness","ness"]` → even if this specific word wasn't in training, the model knows "happy", "un-", "-ness" and can compose something sensible.

**Where used:** LinkedIn search for typo-tolerant matching, languages with complex word forms (German compounds, Finnish), social media with invented words.

---

---

# PART 2: ALL ATTENTION MECHANISMS


## The Problem That Attention Solved

**The Seq2Seq bottleneck:**

An RNN encoder reads "The cat sat on the mat" word by word, compressing everything into ONE vector `c`:

```
"The" → h₁
"cat" → h₂
"sat" → h₃ 
"on"  → h₄
"the" → h₅
"mat" → h₆ = c   ← the ENTIRE sentence is compressed here
```

The decoder must reconstruct the French translation from `c` alone. For short sentences, fine. For long sentences — the model forgets "The" by the time it reaches "mat".

**Attention's fix:** Instead of one context vector `c`, give the decoder a way to look at ALL hidden states `[h₁, h₂, h₃, h₄, h₅, h₆]` at every output step, and decide which ones to focus on.

---

## 1. Bahdanau Attention (2015) — The Original

**Paper:** "Neural Machine Translation by Jointly Learning to Align and Translate"

**How it works step by step:**

When generating output word 1 (say, "Le"):

```
Step 1: Take decoder state s₀ (where the decoder is right now)

Step 2: Compute alignment score for each encoder hidden state:
  e(s₀, h₁) = vᵀ · tanh(W₁·s₀ + W₂·h₁)   → score for "The"
  e(s₀, h₂) = vᵀ · tanh(W₁·s₀ + W₂·h₂)   → score for "cat"
  ...
  e(s₀, h₆) = vᵀ · tanh(W₁·s₀ + W₂·h₆)   → score for "mat"

  These are learned parameters (v, W₁, W₂). This is why it's "additive" — adds the two states.

Step 3: Softmax the scores → attention weights αᵢ (sum to 1)
  α = softmax([e₁, e₂, e₃, e₄, e₅, e₆]) = [0.1, 0.5, 0.1, 0.1, 0.1, 0.1]
  (The model decides to focus on h₂ = "cat" when generating "Le chat")

Step 4: Weighted sum = context vector for this output step
  context = α₁·h₁ + α₂·h₂ + ... + α₆·h₆
  context ≈ 0.5 · h₂ + small contributions from others

Step 5: Decoder uses this context to generate "Le"
```

**Key insight:** Different output words use different attention distributions. "chat" focuses on "cat", "assis" focuses on "sat".

---

## 2. Luong Attention (2015) — The Simpler Version

**Three scoring variants:**

```
Dot product:    score(s, h) = s · h         (simplest, no learned params)
General:        score(s, h) = s · Wₐ · h    (learned weight matrix)
Concat:         score(s, h) = vᵀ · tanh(Wₐ[s;h])  (similar to Bahdanau)
```

**Difference from Bahdanau:**
- Bahdanau computes attention BEFORE the decoder produces its hidden state (forward-looking)
- Luong computes attention AFTER the decoder produces its hidden state (backward-looking)
- Luong's dot product requires no extra parameters → faster

**In practice:** Both work similarly. Luong's simplicity made it more popular and it led directly to the Transformer's scaled dot-product attention.

---

## 3. Scaled Dot-Product Attention — The Transformer Core

**The full formula worked through:**

```
Input: sequence of 4 words "The cat sat down"
Each word embedded as a 4-dimensional vector (in reality it's 512 or 768):

Word embeddings:
  "The"  → [1.0, 0.2, 0.8, 0.3]
  "cat"  → [0.9, 0.8, 0.1, 0.7]
  "sat"  → [0.2, 0.3, 0.9, 0.5]
  "down" → [0.3, 0.1, 0.7, 0.9]

These get multiplied by learned weight matrices to produce Q, K, V:
  Q = embedding × Wq   (query matrix)
  K = embedding × Wk   (key matrix)
  V = embedding × Wv   (value matrix)
```

**Computing attention for "cat":**

```
"cat" query vector q = [0.8, 0.6, 0.3, 0.5]

Dot with each key:
  q · k("The")  = 0.8×1.0 + 0.6×0.2 + 0.3×0.8 + 0.5×0.3 = 1.31
  q · k("cat")  = 0.8×0.9 + 0.6×0.8 + 0.3×0.1 + 0.5×0.7 = 1.70
  q · k("sat")  = 0.8×0.2 + 0.6×0.3 + 0.3×0.9 + 0.5×0.5 = 0.72
  q · k("down") = 0.8×0.3 + 0.6×0.1 + 0.3×0.7 + 0.5×0.9 = 0.87

Divide by √d = √4 = 2:
  [1.31/2, 1.70/2, 0.72/2, 0.87/2] = [0.655, 0.850, 0.360, 0.435]

Softmax → attention weights (sum to 1):
  softmax([0.655, 0.850, 0.360, 0.435]) ≈ [0.22, 0.32, 0.17, 0.29]
  (cat attends to itself most, then "down", then "The", then "sat")

Weighted sum of VALUE vectors:
  output = 0.22×v("The") + 0.32×v("cat") + 0.17×v("sat") + 0.29×v("down")
  
This output is "cat"'s new contextual representation — it now contains information
from all the words, weighted by relevance.
```

**Why scale by √d?**

Without scaling: with d=512 dimensions, dot products can be 100s or 1000s. After softmax, the biggest value gets weight ≈1.0 and everything else ≈0. The model essentially picks ONE word and ignores all others. Dividing by √512 ≈ 22.6 keeps the values in a reasonable range where softmax is more smooth and gradients flow better.

---

## 4. Multi-Head Attention — Why Multiple Heads

**Problem with single attention:** The model can only focus on one type of relationship at a time. Coreference ("it" = "dog") and syntactic role (subject-verb) both matter simultaneously.

**Multi-head solution:**

```
8 heads, each with dimension d/8 = 64 (instead of full 512):

Head 1 learns to look at: syntactic structure
  "the" → attends strongly to its noun "cat"

Head 2 learns to look at: semantic similarity
  "sat" → attends to "down" (both describe position)

Head 3 learns to look at: coreference
  "it" → attends strongly to "cat" or "dog"

Head 4 learns to look at: positional proximity
  Each word attends mostly to adjacent words

... and so on for heads 5-8
```

**Mechanically:**

```
For each head i:
  Qᵢ = X · Wᵢq    Kᵢ = X · Wᵢk    Vᵢ = X · Wᵢv

headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ)

MultiHead = Concat(head₁, head₂, ..., head₈) · Wo
```

`Concat` stacks all 8 head outputs (8 × 64 = 512 dimensions). `Wo` projects back to 512.

---

## 5. Self-Attention vs Cross-Attention

**Self-attention** (BERT encoder, GPT decoder):

```
Source of Q = same sentence
Source of K = same sentence
Source of V = same sentence

"The animal didn't cross the street because it was tired"
                                           ^
When processing "it":
  Self-attention weights: animal=0.72, street=0.08, tired=0.15, others=low
  The model learns "it" refers to "animal"
```

**Cross-attention** (Transformer decoder):

```
Source of Q = decoder's current state
Source of K = encoder's output
Source of V = encoder's output

English input: "The cat sat on the mat"
               ↑ encoded by encoder → [h₁, h₂, h₃, h₄, h₅, h₆]

Decoder generating "Le":
  Q = decoder state for position 1
  K, V = encoder outputs [h₁ ... h₆]
  
  Attention weights: h₂("cat")=0.5, h₁("The")=0.2, others=low
  → Decoder looks at "cat" most when generating "Le chat"
```

---

## 6. Causal (Masked) Self-Attention — How GPT Works

GPT generates text autoregressively — one token at a time. The crucial constraint: when generating word position i, the model cannot "see" words at positions i+1, i+2, ... (they don't exist yet).

**The mask in practice:**

```
Attention score matrix for "The cat sat":

           The    cat    sat
The    [  0.8,   -∞,    -∞  ]    ← "The" can only see itself
cat    [  0.3,   0.9,   -∞  ]    ← "cat" sees "The" and itself
sat    [  0.1,   0.4,   0.7 ]    ← "sat" sees all three

After softmax:   -∞ → 0 (those positions contribute nothing)
           The    cat    sat
The    [ 1.0,   0.0,   0.0 ]
cat    [ 0.25,  0.75,  0.0 ]
sat    [ 0.1,   0.4,   0.5 ]
```

This is why BERT (bidirectional) and GPT (causal/left-to-right) must be different architectures. You cannot use BERT to generate text because it looks at future words. You cannot use GPT for classification as efficiently because it never gets right-to-left context.

---

## 7. Flash Attention — Why It Matters

**Normal attention memory usage:**

```
Sequence length = 4,096 tokens
Attention matrix = 4,096 × 4,096 = 16.7 million values
At float32 → 67 MB of GPU memory — just for attention weights
For a 100K context → 100,000 × 100,000 = 10B values = 40 GB  ← impossible!
```

**Flash Attention's trick:**

Instead of computing the full attention matrix and storing it, Flash Attention breaks Q, K, V into small blocks that fit in fast SRAM (the GPU's L2 cache), processes each block, accumulates the results, and never materialises the full attention matrix.

```
Normal:    Q,K,V in HBM → compute full matrix → store in HBM → read again for V multiply
Flash:     Load block of Q,K,V into SRAM → compute partial attention → accumulate → next block
           No full matrix ever stored → memory = O(n) not O(n²)
```

**The result:** 2–4× faster. 10–20× less memory. Same mathematical output. Enables 128K context (GPT-4), 200K context (Claude 3), 1M context (Gemini 1.5).

---

## 8. Sparse and Sliding Window Attention

**Standard attention cost:** O(n²) — doubles the sequence, quadruples the compute.

For a 100-page legal document (≈50,000 tokens): 50,000² = 2.5 billion attention pairs.

**Sliding Window Attention (Mistral, Longformer):**

Each token only attends to its nearest W tokens (window size):

```
Token 50 can see: [tokens 48, 49, 50, 51, 52]   (W=2 each side)
Token 50 CANNOT see: token 1, token 200, token 50,000

Cost: O(n × W) — linear with sequence length
```

**Global tokens:** A few special tokens (like [CLS]) attend to everything and everything attends to them. This allows long-range information to flow through the global tokens.

**Where used:** Longformer (Hugging Face) for 4,096+ token documents, Mistral 7B (W=4,096), legal document analysis, processing entire codebases.

---

## 9. Grouped Query Attention (GQA) — LLaMA 2/3, Mistral

**Normal multi-head attention:**
Every head has its own Q, K, V matrices — very memory-intensive.

**Grouped Query Attention:**
Multiple query heads share ONE set of K, V matrices.

```
Standard: 32 heads × (Q, K, V each 128-dim) = 32 × 3 × 128 = 12,288 values per token
GQA:      32 Q heads, but only 8 K/V groups → 32×128 + 8×128 + 8×128 = 6,144 values
                                               ← half the KV cache!
```

**Why to care:** The KV cache (storing past keys and values during generation) is the main memory bottleneck when serving LLMs. GQA halves it → you can serve twice as many users simultaneously with the same hardware.

---


---

# PART 3: EVALUATION METRICS — WITH WORKED EXAMPLES

---

## Classification Metrics — Full Example

**Scenario:** You built a LinkedIn post quality classifier. Test it on 20 posts (8 are truly "high quality").

**Model's predictions:**

```
Post #  | Actually HQ? | Model says HQ?
--------|-------------|---------------
1       | Yes          | Yes            ← True Positive (TP)
2       | Yes          | Yes            ← True Positive (TP)
3       | Yes          | Yes            ← True Positive (TP)
4       | Yes          | Yes            ← True Positive (TP)
5       | Yes          | Yes            ← True Positive (TP)
6       | Yes          | No             ← False Negative (FN) — missed it
7       | Yes          | No             ← False Negative (FN) — missed it
8       | Yes          | No             ← False Negative (FN) — missed it
9       | No           | Yes            ← False Positive (FP) — wrongly flagged
10      | No           | Yes            ← False Positive (FP) — wrongly flagged
11–20   | No           | No             ← True Negative (TN) ×10
```

**Counting:**
```
TP = 5   (correctly identified as HQ)
FN = 3   (HQ posts we missed)
FP = 2   (low-quality posts we wrongly called HQ)
TN = 10  (correctly identified as not HQ)
```

**Accuracy:**
```
Accuracy = (TP + TN) / total = (5+10) / 20 = 15/20 = 0.75 = 75%
```

**Precision** = "Of all posts I called HQ, what fraction actually were?"
```
Precision = TP / (TP + FP) = 5 / (5+2) = 5/7 = 0.71 = 71%
```
3 out of every 10 posts I promote as "high quality" are actually low quality.

**Recall** = "Of all actual HQ posts, what fraction did I find?"
```
Recall = TP / (TP + FN) = 5 / (5+3) = 5/8 = 0.625 = 62.5%
```
I missed 37.5% of truly high-quality posts — they were buried.

**F1 Score** = harmonic mean (penalises big imbalances between P and R):
```
F1 = 2 × (0.71 × 0.625) / (0.71 + 0.625) = 2 × 0.444 / 1.335 = 0.666
```

**The trade-off — why you can't maximise both:**

If you lower the decision threshold (call MORE posts HQ):
→ Recall goes UP (catch more real HQ posts)
→ Precision goes DOWN (more low-quality posts slip through)

If you raise the threshold (call FEWER posts HQ):
→ Precision goes UP (what you do flag is almost certainly HQ)
→ Recall goes DOWN (you miss more real HQ posts)

**The Decision:**
For a quality feed: care more about Precision (don't promote bad content) than Recall.
For fraud detection: care more about Recall (catch every fraud, accept some false alarms).

---

## BLEU Score — Full Worked Example

**Task:** Evaluate machine translation quality.

```
Reference (correct): "The cat sat on the mat"
Model output:        "The cat is on the mat"
```

**Unigram (1-gram) precision:**
```
Output tokens:     The  cat  is  on  the  mat
Reference has:     The✓ cat✓ is✗ on✓ the✓ mat✓

Matches: 5 out of 6
1-gram precision = 5/6 = 0.833
```

**Bigram (2-gram) precision:**
```
Output bigrams:    [The cat] [cat is] [is on] [on the] [the mat]
Reference bigrams: [The cat] [cat sat] [sat on] [on the] [the mat]

[The cat]✓  [cat is]✗  [is on]✗  [on the]✓  [the mat]✓
Matches: 3 out of 5
2-gram precision = 3/5 = 0.60
```

**3-gram precision:**
```
Output:    [The cat is] [cat is on] [is on the] [on the mat]
Reference: [The cat sat] [cat sat on] [sat on the] [on the mat]

[on the mat]✓ only
3-gram precision = 1/4 = 0.25
```

**BLEU = geometric mean × brevity penalty**
```
BLEU = BP × exp(0.25 × log(0.833) + 0.25 × log(0.60) + 0.25 × log(0.25) + 0.25 × log(0))

In practice for this example ≈ 0.32 (below 0.4 threshold → not a great translation)
```

**Interpretation:**
- BLEU < 0.1 → almost useless translation
- BLEU 0.1–0.3 → gist understandable
- BLEU 0.3–0.5 → good quality
- BLEU > 0.5 → expert quality
- BLEU = 1.0 → exact match (never happens in practice)

**Why BLEU fails for LLMs:**
"The feline rested upon the carpet" = same meaning, BLEU ≈ 0 vs the reference.
That's why BERTScore is now preferred for evaluating modern LLM outputs.

---

## ROUGE Score — Summarisation Quality

ROUGE measures RECALL (did the summary cover the key content?) while BLEU measures PRECISION.

**ROUGE-1 (unigram recall):**
```
Reference summary: "The dog ate the bone quickly"
Model summary:     "The dog ate a bone"

Reference words in model: The✓ dog✓ ate✓ the✓(=a, partial) bone✓ quickly✗
Overlap = 4 out of 6 reference words
ROUGE-1 = 4/6 = 0.67
```

**ROUGE-L (Longest Common Subsequence):**
Finds the longest sequence of words that appear in both, in order:
```
Reference: The dog ate the bone quickly
Model:     The dog ate a bone

LCS = "The dog ate" + "bone" = 4 words
ROUGE-L = 4/6 = 0.67
```

**When to use ROUGE vs BLEU:**
- ROUGE for summarisation: did the summary include the key points from the reference?
- BLEU for translation: was the output word-for-word close to the reference?

---

## Perplexity — Full Example

**Setup:** You're comparing two language models on the sentence "The cat sat on the mat."

```
Model A predictions (after seeing each word):
  P("cat"   | "The")           = 0.15
  P("sat"   | "The cat")       = 0.20
  P("on"    | "The cat sat")   = 0.30
  P("the"   | "... sat on")    = 0.50
  P("mat"   | "... on the")    = 0.25

Average log probability = (log(0.15)+log(0.20)+log(0.30)+log(0.50)+log(0.25)) / 5
                        = (-1.897 + -1.609 + -1.204 + -0.693 + -1.386) / 5
                        = -6.789 / 5 = -1.358

Perplexity = 10^(1.358) = 22.8
```

```
Model B (better model, more training):
  Same words but P("cat" | "The") = 0.30, etc. (higher probabilities)
  Perplexity = 10.2
```

Model B has lower perplexity → less surprised by the text → better language model.

**Intuition:** If GPT-4 has perplexity of 5.4 and LLaMA-7B has perplexity of 12.1 on the same benchmark, GPT-4 has a much better internal model of language. But perplexity doesn't tell you which is more helpful — that needs human evaluation.

---

## NDCG — Ranking Quality Example

**Scenario:** LinkedIn job search returns 5 results. 3 are relevant with different quality levels.

```
Relevance scores:    0=irrelevant, 1=somewhat relevant, 2=very relevant

Perfect ranking:   pos 1=2, pos 2=2, pos 3=1, pos 4=0, pos 5=0
Our model output:  pos 1=2, pos 2=0, pos 3=2, pos 4=1, pos 5=0
```

**Computing DCG (Discounted Cumulative Gain) for our model:**
```
DCG = rel₁ + rel₂/log₂(2) + rel₃/log₂(3) + rel₄/log₂(4) + rel₅/log₂(5)
    = 2/1  + 0/1           + 2/1.585       + 1/2           + 0/2.322
    = 2.0  + 0.0           + 1.262         + 0.5           + 0.0
    = 3.762
```

**Computing Ideal DCG (best possible ranking):**
```
IDCG = 2/1 + 2/1 + 1/1.585 + 0 + 0
     = 2.0 + 2.0 + 0.631
     = 4.631
```

**NDCG = DCG / IDCG = 3.762 / 4.631 = 0.812**

Score of 0.812 means: our ranking is 81.2% as good as the perfect ranking.

We lost points by putting a score-0 result at position 2 instead of the score-2 result.

**Decision:** Is NDCG@5 = 0.81 good enough to ship? Compare to baseline (old search = 0.74). That's a 9.5% relative improvement — probably worth shipping.

---

## MRR — Mean Reciprocal Rank

Used when you care about "how quickly does the user find ONE good result?"

```
3 test queries:

Query 1: "machine learning job"
  Results: [irrelevant, irrelevant, relevant, ...]
  First relevant at position 3 → reciprocal rank = 1/3

Query 2: "python developer"
  Results: [relevant, ...]
  First relevant at position 1 → reciprocal rank = 1/1

Query 3: "product manager"
  Results: [irrelevant, relevant, ...]
  First relevant at position 2 → reciprocal rank = 1/2

MRR = (1/3 + 1/1 + 1/2) / 3 = (0.333 + 1.0 + 0.5) / 3 = 1.833/3 = 0.611
```

An MRR of 0.61 means on average users find a relevant result at approximately position 1.6 — not bad, but there's room to improve query 1.

---

---

# PART 4: MODEL ADAPTATION METHODS — SIMPLE EXPLANATIONS


---

## The Spectrum of Adaptation

From cheapest/fastest to most expensive/powerful:

```
Zero-shot prompting → Few-shot prompting → Prompt tuning → 
LoRA → Full fine-tuning → Pre-training from scratch
```

---

## Zero-Shot and Few-Shot Prompting

**Zero-shot:** Give the model no examples. Just describe the task.

```
Prompt: "Classify this LinkedIn post as professional or spam:
'Check out my FREE webinar on making $10K/month with no effort!'
Classification:"

GPT-4: "spam"
```

**Few-shot:** Give 2–5 examples to show the pattern.

```
Prompt: "Classify these posts:
Post: 'I just got promoted to Senior Engineer!' → professional
Post: 'Make $5000 from home, click here!' → spam
Post: 'Excited to share my first open-source project on GitHub' → professional
Post: 'Buy my course, use code FRIEND for 50% off' →"

GPT-4: "spam"
```

Few-shot works surprisingly well because GPT-4 already understands the concepts — you're just showing it the format and calibrating it.

**When to use:** Prototyping, when you have no labelled data, when the task is common enough that the base model already "gets it."

---

## Full Fine-Tuning — Step by Step

**What actually happens during fine-tuning:**

```
Step 1: Load pre-trained BERT (all 110 million weights)

Step 2: Add a new output layer for your task
  Original BERT output: 768-dimensional vector per token
  New layer: 768 → 2 (binary classification: spam/not-spam)

Step 3: Prepare your dataset
  500 LinkedIn posts you manually labelled as spam / not-spam
  Split: 400 training, 50 validation, 50 test

Step 4: Training loop
  For each batch of 32 posts:
    Forward pass → compute prediction
    Compare to label → compute loss
    Backward pass → compute gradients
    Update ALL 110M + 1,536 (new layer) weights

Step 5: Evaluate on validation set every epoch
  Epoch 1: F1 = 0.72
  Epoch 2: F1 = 0.84
  Epoch 3: F1 = 0.89   ← validation stops improving
  Stop here (early stopping) to avoid overfitting

Step 6: Test set evaluation: F1 = 0.87
```

**Hyperparameters that matter:**

Learning rate: how big each weight update is. Too high → overshoots, unlearns general knowledge. Too low → takes forever. Typical for fine-tuning: 2e-5 to 5e-5.

Batch size: how many examples per update. Larger = more stable but more memory.

Epochs: how many times to pass through the full dataset. Usually 2–4 for fine-tuning (more = overfitting risk).

Warmup steps: start with a very small learning rate for the first few batches, then increase to target. Prevents the first big gradient update from destroying pre-trained weights.

---

## LoRA — What It Actually Does

**The math made simple:**

A weight matrix W has shape 768×768 = 589,824 parameters.

LoRA adds two smaller matrices:
```
A has shape 768 × 4    = 3,072 parameters
B has shape 4 × 768    = 3,072 parameters

Total LoRA params for this layer: 6,144   (vs 589,824 original)
```

During forward pass: `output = x · (W + A·B)`

W is frozen (never changes). Only A and B are trained.

```
Ratio: 6,144 / 589,824 = 1%   ← 99% fewer parameters to train
```

Across a full BERT model with all attention layers, LoRA reduces trainable parameters from 110M to about 4M.

**Multiple LoRA adapters:**

```
Base GPT-4 (frozen)
     ├── LoRA adapter: LinkedIn post writing style
     ├── LoRA adapter: Medical Q&A (trained on clinical notes)
     ├── LoRA adapter: Legal document summarisation
     └── LoRA adapter: Code review comments
```

You keep one large base model and swap small adapter files (a few MBs) per task. Much cheaper than 4 separate fine-tuned models.

**QLoRA:** LoRA + quantisation (storing weights in 4-bit instead of 16-bit). Enables fine-tuning a 65B parameter LLaMA on a single consumer GPU (24GB VRAM).

---

## RLHF — Why It Was Necessary

**The problem with standard pre-training:**

GPT-3 trained on internet text → it learned to write text that "looks like the internet". The internet contains:
- Helpful tutorials
- Toxic arguments
- Conspiracy theories
- Detailed answers
- Rambling nonsense
- Factual articles
- Confident misinformation

GPT-3 produces all of these with equal probability, because that's what the training data contained. It's optimised for "looks like text from the web" — not "actually helpful to a human."

**What RLHF trains for:**

The reward model captures a property that's very hard to specify mathematically: "would a thoughtful person rate this response as helpful, harmless, and honest?"

No loss function in the world can directly encode "don't be sycophantic but don't be rude" or "refuse dangerous requests but help with sensitive topics that are legitimate." Human preferences are the only way to capture this.

**RLHF example in detail:**

```
Prompt: "How do I whittle a knife?"

SFT model output:
  "To whittle a knife, you'll need: a piece of wood (basswood is good for beginners),
   a sharp knife, and sandpaper. Start by..."
  → Helpful answer. Correct.

Prompt: "How do I whittle a knife to hurt my sister?"

SFT model output (before RLHF):
  "Here are knife-whittling techniques that could be used to create a weapon..."
  → Dangerous. Should refuse.
```

Human raters see both outputs and rate the second one very poorly. The reward model learns: similar outputs to the second prompt → low score. After PPO training: the model learns to decline harmful intent while still answering legitimate whittling questions.

---

## DPO — RLHF Simplified

**The problem with RLHF:**
Training three separate models (SFT model, reward model, PPO-trained model) is complex, slow, and can be unstable. The reward model can be "hacked" — the main model learns to produce outputs that score well on the reward model but aren't actually better for humans.

**DPO's insight:** Skip the reward model entirely. Directly use human preference pairs.

```
Training data format:
{
  "prompt": "How do I start a business?",
  "chosen": "Great question! Starting a business involves several key steps: 
             validating your idea, registering your business, building an MVP...",
  "rejected": "You need money, connections, and luck. Most businesses fail anyway."
}
```

**The DPO training objective:** Increase the probability of `chosen` responses and decrease the probability of `rejected` responses — simultaneously — in a mathematically principled way that prevents the model from drifting too far from the original pre-trained distribution.

**Why "direct":** No reward model scores involved. The preference signal goes straight into the weight updates.

**Result:** Simpler, more stable, often similar quality to RLHF. Used in LLaMA 2 Chat, Mistral Instruct, Claude 3 (partially).

---

## RAG — Step by Step

**The core problem RAG solves:**

```
User: "What were LinkedIn's Q4 2024 earnings?"

Without RAG: GPT-4 knows nothing after its training cutoff (2024-01).
             It will either hallucinate a number or refuse.

With RAG: retrieve the actual earnings report → inject → answer correctly.
```

**The full RAG pipeline:**

```
Step 1: INDEXING (done once, offline)
  - Collect your documents: help articles, product docs, earnings reports
  - Split into chunks (e.g. 512 tokens each)
  - Embed each chunk with a sentence embedding model
    "LinkedIn Q4 2024 revenue was $4.2B" → [0.82, 0.14, 0.65, ...]
  - Store all embeddings in a Vector Database (Pinecone, Weaviate, Chroma)

Step 2: RETRIEVAL (at query time)
  User asks: "What were LinkedIn's Q4 earnings?"
  - Embed the question: [0.80, 0.17, 0.61, ...]
  - ANN search: find the 5 most similar document chunks
  - Retrieved chunks:
    Chunk 3: "LinkedIn reported Q4 2024 revenue of $4.2 billion..."
    Chunk 7: "Year-over-year growth was 12% for Q4..."
    Chunk 12: "Premium subscriptions grew 18% in Q4..."

Step 3: GENERATION (augmented)
  Prompt to LLM:
    "Answer this question using ONLY the provided context.
     Context: [Chunk 3 + Chunk 7 + Chunk 12]
     Question: What were LinkedIn's Q4 2024 earnings?
     Answer:"

  LLM output: "LinkedIn's Q4 2024 revenue was $4.2 billion, 
               representing 12% year-over-year growth..."
```

**Why RAG > fine-tuning for knowledge:**

Fine-tuning bakes knowledge into model weights permanently. If earnings change, you retrain.
RAG fetches from a database that you can update in real-time — no retraining needed.

Fine-tuning for knowledge often leads to hallucination (the model "remembers" confidently but incorrectly).
RAG grounds the answer in actual text — you can cite the source.

**RAG limitations:**
- Retrieval quality is critical — bad retrieval → bad answer even with good LLM
- Long, complex reasoning across many documents is still hard
- Context window limit: you can only inject so many chunks

---

## Prompt Engineering — The Underrated Skill

Before fine-tuning, always try prompt engineering. It's free, instant, and surprisingly powerful with GPT-4.

**Key techniques:**

**Role prompting:**
```
"You are an expert LinkedIn content strategist. Your job is to..."
```

**Chain-of-thought (CoT):**
```
"Think step by step: First identify the claim, then check it against these facts,
then provide your verdict."
```

This dramatically improves accuracy on reasoning tasks. The model that "thinks out loud" makes fewer errors.

**Few-shot with format:**
```
"Classify posts exactly as shown:
Post: [example 1] → Label: spam
Post: [example 2] → Label: professional
Post: [your post here] → Label:"
```

**System + user prompt separation:**
```
System: "You are a helpful assistant that only uses information from the provided
         context. Never make up information."
User: "Context: [retrieved chunks] Question: [user question]"
```

**Output format control:**
```
"Respond only in JSON format:
{'sentiment': 'positive/negative/neutral', 'confidence': 0.0-1.0, 'key_phrase': '...'}"
```

Forces structured output that downstream code can parse reliably.

---


---

# PART 5: ADVANCED CONCEPTS — SIMPLY EXPLAINED

---

## Tokenisation Deep Dive

Modern LLMs don't split on words — they use Byte-Pair Encoding (BPE).

**BPE algorithm:**

Start with individual characters. Merge the most frequent pair repeatedly.

```
Initial: "cat" = ["c","a","t"]   "cats" = ["c","a","t","s"]

After seeing "cat" 10,000 times, merge "c"+"a" → "ca"
After seeing "cat" more, merge "ca"+"t" → "cat"
After seeing "cats", merge "cat"+"s" → "cats"

Final vocabulary: ["c","a","t","s","ca","cat","cats","the","..."]
```

**Result:**

```
"unhappiness" → ["un", "happ", "iness"]
"ChatGPT"     → ["Chat", "G", "PT"]
"supercalifragilistic" → ["super","cal","if","rag","il","istic"]
```

Common words get one token. Rare or made-up words get split into subwords.

**Why it matters:**

1 token ≈ 4 characters ≈ 0.75 words. A 1,000-word document ≈ 1,333 tokens.

Costs are per token. A model with 128K context can process roughly 96,000 words (a 300-page book). Exceeding the context limit causes errors — the model can't "remember" earlier parts of the conversation.

---

## Hallucination — Why It Happens

LLMs don't "know" facts. They learn probability distributions over tokens.

```
"The capital of France is ___" → P("Paris") ≈ 0.98   ← confident, correct
"The CEO of MicroCorp in 2019 is ___" → no strong signal in training data
   → model picks plausible-sounding name with high probability
   → "James Richardson" (invented, but sounds right)
   → says it confidently because that's what the probability distribution does
```

**The model has no "I don't know" switch.** It always produces the most likely continuation given its training data. If the answer wasn't in the training data, the most likely continuation is still SOMETHING plausible-sounding — not silence.

**Mitigations:**

RAG: ground the answer in retrieved documents.
Temperature = 0: make the model deterministic (always pick highest-probability token). Less creative, more reliable for factual queries.
Calibration: train the model to say "I'm not sure" when confidence is low.
Fact-checking pipeline: second model verifies claims after generation.

---

## Temperature and Sampling

When GPT generates text, at each step it has a probability distribution over the vocabulary:

```
After "The cat sat on the":
  "mat"    → 0.42
  "floor"  → 0.18
  "grass"  → 0.12
  "chair"  → 0.08
  others   → 0.20
```

**Temperature** controls how "spread out" this distribution is:

```
Temperature = 0: always pick the max (0.42 → "mat"). Deterministic. Boring but reliable.
Temperature = 1: sample proportionally. "mat" 42% of the time. Creative.
Temperature = 2: flatten the distribution. Even unlikely words get picked sometimes. Very creative, often incoherent.
Temperature = 0.5: sharpen the distribution. More conservative than 1, less than 0.
```

**Top-p (nucleus sampling):** Instead of temperature, only sample from the smallest set of words that together have probability > p.

```
Top-p = 0.9: 
  mat(0.42) + floor(0.18) + grass(0.12) + chair(0.08) + bed(0.07) + rug(0.05) = 0.92 > 0.90
  Only sample from these 6 words. Ignore everything else.
```

**Decision:** Customer support bot → temperature = 0 (consistent, factual). Creative writing tool → temperature = 0.8–1.0 (varied, surprising).

---

## Context Window vs Memory

**Context window:** The amount of text the model can "see" at once. This is its working memory.

```
GPT-3:     4,096 tokens  ≈ 3,000 words  ≈ 6 pages
GPT-4:     128,000 tokens ≈ 96,000 words ≈ 200 pages
Claude 3:  200,000 tokens ≈ 150,000 words ≈ 300 pages
Gemini 1.5: 1,000,000 tokens ≈ book-length
```

**The limitation:** Everything beyond the context window is invisible. If a conversation runs long enough, the model forgets the beginning.

**Long-context strategies:**
- Summarise older context, keep recent messages verbatim
- RAG: retrieve only the relevant chunks from a long document
- Memory systems: store key facts externally, inject when relevant

---

## Model Quantisation

Training uses 32-bit or 16-bit floating point numbers. Quantisation reduces this.

```
FP32:  each weight = 4 bytes
FP16:  each weight = 2 bytes  → model size halved
INT8:  each weight = 1 byte   → model size quartered, slight accuracy loss
INT4:  each weight = 0.5 bytes → 8× smaller, more accuracy loss
```

**Why it matters:**

```
LLaMA 3 70B parameters:
  FP32:  70B × 4 bytes = 280 GB  (needs 4 × A100 GPUs, $20K+ hardware)
  FP16:  70B × 2 bytes = 140 GB  (needs 2 × A100)
  INT4:  70B × 0.5 bytes = 35 GB (fits on 1 × A100, or 2 × consumer GPUs)
```

**GGUF / llama.cpp:** Run quantised models on a laptop. A 4-bit quantised LLaMA 3 8B runs at acceptable speed on a MacBook Pro with 16GB RAM.

**Trade-off:** Smaller = cheaper = slightly less accurate. For most production use cases, 4-bit quantisation loses less than 1–2% accuracy — acceptable.

---

## The Transformer Scaling Laws

**Scaling law** (Chinchilla, 2022): For a given compute budget, what's the optimal model size and training data size?

```
Chinchilla rule: 
  Optimal tokens = 20 × number of parameters

  GPT-3 has 175B params → should train on 175B × 20 = 3.5T tokens
  (GPT-3 was actually trained on 300B tokens → undertrained by Chinchilla standards)

  LLaMA 3 8B → optimal = 8B × 20 = 160B tokens
  (LLaMA 3 was trained on 15T tokens → heavily overtrained → better for inference)
```

**Emergent abilities:** Some capabilities appear suddenly as models scale up — not gradually.

```
GPT-2 (1.5B):   cannot do multi-step arithmetic
GPT-3 (175B):   can do it (often wrong)
GPT-4 (~1T?):   does it reliably

The jump from "can't" to "can" was not linear — it emerged at a threshold.
```

Nobody fully understands why this happens. It's one of the most active research areas.

---


---

# PART 6: COMPLETE CHEAT SHEET SUPPLEMENT

---

## Attention Type Quick Reference

| Attention Type | Q comes from | K,V come from | Used in | Key property |
|---|---|---|---|---|
| Bahdanau | Decoder state | Encoder states | RNN seq2seq | Additive, learned alignment |
| Luong | Decoder state | Encoder states | RNN seq2seq | Multiplicative, simpler |
| Scaled Dot-Product | Any | Same sequence | Transformers | Foundation of all modern attention |
| Self-Attention | Same sequence | Same sequence | BERT encoder, GPT decoder | Contextual word representations |
| Cross-Attention | Decoder | Encoder output | T5, translation | Connects encoder to decoder |
| Causal (Masked) | Decoder | Past tokens only | GPT, autoregressive | Can't see future tokens |
| Multi-Head | Multiple sub-spaces | Multiple sub-spaces | All Transformers | Captures multiple relationship types |
| Flash Attention | Same | Same | Modern LLMs | Same math, 4× faster, less memory |
| Sparse/Sliding Window | Local window | Local window | Mistral, Longformer | O(n×W) not O(n²) |
| GQA | Multiple Q groups | Shared K,V groups | LLaMA 2/3, Mistral | Less KV cache memory |

---

## Metrics Quick Reference

| Metric | Used for | Formula (simplified) | Range | Higher = better? |
|---|---|---|---|---|
| Accuracy | Classification | Correct / Total | 0–1 | Yes |
| Precision | Classification | TP / (TP+FP) | 0–1 | Yes |
| Recall | Classification | TP / (TP+FN) | 0–1 | Yes |
| F1 | Classification | 2×P×R / (P+R) | 0–1 | Yes |
| BLEU | Translation | n-gram precision (geometric mean) | 0–1 | Yes |
| ROUGE-1 | Summarisation | Unigram recall | 0–1 | Yes |
| ROUGE-L | Summarisation | LCS recall | 0–1 | Yes |
| Perplexity | Language model | 10^(avg cross-entropy) | 1–∞ | No (lower = better) |
| BERTScore | Generation (semantic) | BERT cosine similarity | 0–1 | Yes |
| NDCG | Search ranking | Discounted gain / ideal | 0–1 | Yes |
| MRR | Search ranking | Mean(1/rank of first hit) | 0–1 | Yes |
| Precision@K | Search ranking | Relevant in top K / K | 0–1 | Yes |
| Win Rate | LLM quality | % preferred in A/B comparison | 0–1 | Yes |

---

## Fine-Tuning Method Comparison

| Method | Trainable Params | Data Needed | Cost | When to Use |
|---|---|---|---|---|
| Zero-shot prompting | 0 | 0 examples | Free | Prototyping, GPT-4-class models |
| Few-shot prompting | 0 | 2–10 examples | Free | Any task GPT-4 understands |
| Prompt tuning | Soft tokens only | 100–1K examples | Very low | When API only, no model access |
| LoRA | 1–10% of model | 1K–100K examples | Low | Most production use cases |
| Full fine-tuning | 100% of model | 10K–1M examples | High | Max accuracy, have data and budget |
| RLHF | 100% + reward model | Human preference rankings | Very High | Safety, instruction following |
| DPO | 100% | Preference pairs | High (simpler than RLHF) | Safety, instruction following |
| Pre-train from scratch | 100%, start random | Billions of tokens | Extreme | New language/domain (rare for companies) |

---

## Vocabulary — Every Term Defined

| Term | Plain English |
|---|---|
| Token | The basic unit a model processes — roughly a word or subword |
| Embedding | A word or sentence represented as a list of numbers |
| Vector | A list of numbers — the mathematical representation of anything |
| Corpus | A large collection of text used for training |
| Vocabulary | The complete set of tokens the model knows |
| OOV | Out-Of-Vocabulary — a word not in the training vocabulary |
| Gradient | The signal that tells each weight which direction to change |
| Vanishing gradient | Gradient becomes too small to update early layers — they stop learning |
| Overfitting | Model memorises training data, fails on new examples |
| Underfitting | Model is too simple to capture the patterns in the data |
| Epoch | One complete pass through the entire training dataset |
| Batch | A small subset of training data processed at once |
| Learning rate | How big each parameter update step is |
| Softmax | Converts a list of scores into probabilities that sum to 1 |
| Cross-entropy loss | Standard loss function for classification (measures prediction error) |
| Logit | Raw (unnormalised) output of a neural network, before softmax |
| Layer normalisation | Normalises activations within each layer — stabilises training |
| Residual connection | Adds the input directly to the output (skip connection) — prevents gradient vanishing |
| Dropout | Randomly zero out neurons during training — prevents overfitting |
| KV cache | Stores past key-value attention matrices to speed up text generation |
| Temperature | Controls randomness of LLM output (high = more random) |
| Top-p sampling | Sample from the smallest vocabulary subset whose probabilities sum to p |
| Hallucination | Model generates confident but false information |
| Grounding | Tying model outputs to verified external sources (e.g. via RAG) |
| Transfer learning | Using a model trained on task A as starting point for task B |
| Catastrophic forgetting | Fine-tuning causes model to lose its general pre-trained knowledge |
| Knowledge distillation | Training a small model to mimic a large model's outputs |
| Quantisation | Reducing weight precision (32-bit → 4-bit) to shrink model size |
| Inference | Using a trained model to make predictions (as opposed to training) |
| Autoregressive | Generates one token at a time, each conditioned on all previous tokens |
| Masked LM | Training task where random tokens are hidden and the model predicts them |
| Causal LM | Training task where the model predicts the next token given all previous |
| Prompt | The input text given to a language model |
| System prompt | Instructions given to the model about its persona and rules |
| Context window | Maximum number of tokens the model can process at once |
| Emergent ability | A capability that appears suddenly at scale, not gradually |

---

